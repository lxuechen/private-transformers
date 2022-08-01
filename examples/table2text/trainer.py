import inspect
import json
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from ml_swissknife import utils
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import is_datasets_available, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_pt_utils import distributed_broadcast_scalars
from transformers.trainer_utils import (EvalPrediction, EvaluationStrategy, IntervalStrategy, PREFIX_CHECKPOINT_DIR,
                                        PredictionOutput, TrainOutput, set_seed)
from transformers.utils import logging

from . import decoding_utils
from .compiled_args import (AuxiliaryArguments, DataTrainingArguments, ModelArguments, PrivacyArguments,
                            TrainingArguments)

logger = logging.get_logger(__name__)


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments`
            with the ``output_dir`` set to a directory named `tmp_trainer` in the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or
            :obj:`eval_dataset`. Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is
            provided, an instance of :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.
        kwargs:
            Deprecated keyword arguments.
    """

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        args: Optional[TrainingArguments] = None,
        model_args: Optional[ModelArguments] = None,
        data_args: Optional[DataTrainingArguments] = None,
        privacy_args: Optional[PrivacyArguments] = None,
        auxiliary_args: Optional[AuxiliaryArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),

        val_dataset: Optional[Dataset] = None,
        generation_stuff: Optional[Dict] = None,
        **kwargs,
    ):
        if args is None:
            logger.info("No `TrainingArguments` passed, using the current path as `output_dir`.")
            args = TrainingArguments("tmp_trainer")
        self.args = args
        self.model_args = model_args
        self.data_args = data_args
        self.privacy_args = privacy_args
        self.auxiliary_args = auxiliary_args

        # Seed must be set before instantiating the model when using model
        set_seed(self.args.seed)
        assert (
            model is not None or model_init is not None
        ), "You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` " \
           "argument."
        assert model_init is None
        self.model = model.to(args.device) if model is not None else None
        self.num_params = sum(
            param.numel() for param in self.model.parameters() if param.requires_grad
        )
        from transformers.modeling_utils import Conv1D
        self.num_non_embedding_params = sum(
            param.numel()
            for module in self.model.modules() if isinstance(module, (nn.LayerNorm, Conv1D))
            for param in module.parameters() if param.requires_grad
        )
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.val_dataset = val_dataset
        self.generation_stuff = generation_stuff
        self.tokenizer = tokenizer
        self.curr_best_eval = 10000000.
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument."
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        self.log_history = []
        if "prediction_loss_only" in kwargs:
            warnings.warn(
                "Passing `prediction_loss_only` as a keyword argument is deprecated and won't be possible in a future "
                "version. Use `args.prediction_loss_only` instead.",
                FutureWarning,
            )
            self.args.prediction_loss_only = kwargs.pop("prediction_loss_only")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

        # Create output directory if needed
        if self.is_world_process_zero():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_torch_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            self.data_collator = self.data_collator.collate_batch
            warnings.warn(
                (
                    "The `data_collator` should now be a simple callable (function, class with `__call__`), classes "
                    + "with a `collate_batch` are deprecated and won't be supported in a future version."
                ),
                FutureWarning,
            )

        self.global_step = None
        self.epoch = None
        self.total_flos = None
        self.hp_search_backend = None
        self.use_tune_checkpoints = False
        if self.args.label_names is None:
            self.args.label_names = (
                ["start_positions, end_positions"]
                if type(self.model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values()
                else ["labels"]
            )

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(self.model.forward)
        signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        signature_columns += ["label", "label_ids"]
        columns = [k for k in signature_columns if k in dataset.column_names]
        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        dset_description = "" if description is None else f"in the {description} set "
        logger.info(
            f"The following columns {dset_description}don't have a corresponding argument in `"
            f"{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        )
        dataset.set_format(type=dataset.format["type"], columns=columns)

    def _get_train_sampler(self, shuffle=True) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        else:
            # Sometimes we don't want to shuffle!
            if shuffle:
                return (
                    RandomSampler(self.train_dataset)
                    if self.args.local_rank == -1
                    else DistributedSampler(self.train_dataset)
                )
            else:
                return SequentialSampler(self.train_dataset)

    def get_train_dataloader(self, train_sampler=None) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if train_sampler is None:
            train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            return None
        elif self.args.local_rank != -1:
            raise ValueError("Multi-gpu and distributed training is currently not supported.")
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.eval_dataset` is a :obj:`torch.utils.data.IterableDataset`, a sequential
        sampler (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               (not any(nd in n for nd in no_decay)) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.
        """
        return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None, **kwargs):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
        """
        if self.args.local_rank != -1 or self.args.n_gpu > 1:
            raise ValueError("Multi-gpu and distributed training is currently not supported.")
        if self.args.fp16:
            raise ValueError("Mixed-precision training is currently not supported.")

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(self.args.seed)
            model = self.model_init()
            self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(num_update_steps_per_epoch * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        # Evaluate before training.
        if self.args.evaluate_before_training:
            self.evaluate(epoch=0)  # No need to report to hp search.

        # --- low rank analysis project ---
        callback = None

        if self.auxiliary_args.orthogonal_projection_path is not None:
            state_dicts = torch.load(self.auxiliary_args.orthogonal_projection_path)
            # Kept on CPU during most of the time of training.
            orthogonal_projection = state_dicts.get("eigenvectors")[:, :self.auxiliary_args.orthogonal_projection_rank]

            def callback(privacy_engine):
                """Orthogonally project flattened `.summed_grad` with projection matrix then fill this back."""
                named_params = privacy_engine.named_params

                # Collect.
                flat_grad = []
                for _, param in named_params:
                    flat_grad.append(param.summed_grad.flatten())
                    param.summed_grad = None  # Save memory.
                flat_grad = torch.cat(flat_grad)

                # Project.
                P = orthogonal_projection  # noqa
                if orthogonal_projection.device != flat_grad.device or orthogonal_projection.dtype != flat_grad.dtype:
                    P = orthogonal_projection.to(flat_grad)  # noqa
                Pt_flat_g = torch.matmul(P.t(), flat_grad)  # noqa
                # Matrix multiplication with very large dimension (millions in this case) results in weird issues.
                # In this case, `torch.matmul` fails due to calling some algo. Resorting to `torch.mm` for now.
                flat_grad = torch.mm(orthogonal_projection, Pt_flat_g[:, None]).squeeze()

                # Redistribute.
                grads = utils.flat_to_shape(flat_tensor=flat_grad, shapes=[param.shape for _, param in named_params])
                for (_, param), grad in utils.zip_(named_params, grads):
                    param.summed_grad = grad

        if self.auxiliary_args.store_grads:
            store_grads_dir = utils.join(self.args.output_dir, 'grad_trajectory')
            utils.makedirs(store_grads_dir, exist_ok=True)
        else:
            store_grads_dir = None
        # ---

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.warning("***** Running training *****")
        logger.warning("  Num examples = %d", self.num_examples(train_dataloader))
        logger.warning("  Num Epochs = %d", num_train_epochs)
        logger.warning("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.warning("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       total_train_batch_size)
        logger.warning("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.warning("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split(os.path.sep)[0])
                if self.args.n_gpu > 1:
                    self.total_flos = getattr(model.module.config, "total_flos", 0)
                else:
                    self.total_flos = getattr(model.config, "total_flos", 0)

                epochs_trained = self.global_step // num_update_steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % (num_update_steps_per_epoch)

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Continuing training from %d non-embedding floating-point operations", self.total_flos)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                self.total_flos = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            # This extra step is crucial. The problem is that the total number of steps in one epoch might
            # not divide the number of accumulation steps, thus the accumulated .summed_grad (.grad) might overflow to
            # the next epoch, causing more gradient signal than there truly is.
            model.zero_grad(set_to_none=True)

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration", disable=disable_tqdm)
            for step, inputs in enumerate(epoch_iterator):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    epoch_pbar.update(1)
                    continue

                losses = self.training_step(model, inputs)
                tr_loss += losses["scalar_loss"]
                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    self.args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)
                ):
                    if self.privacy_args.non_private:
                        # Don't double clip in private learning.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        self.optimizer.step()
                    else:
                        if store_grads_dir is not None:
                            def callback(privacy_engine):
                                named_params = privacy_engine.named_params
                                flat_grad = torch.cat([param.summed_grad.flatten() for _, param in named_params])
                                torch.save(
                                    {"flat_grad": flat_grad.cpu().float()},
                                    utils.join(store_grads_dir, f'global_step_{self.global_step:06d}.ckpt')
                                )

                        vector_loss = losses.get("vector_loss")
                        self.optimizer.step(loss=vector_loss, callback=callback)

                    self.lr_scheduler.step()
                    model.zero_grad(set_to_none=True)

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            self.lr_scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else self.lr_scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar

                        self.log(logs)

                    if (
                        self.args.evaluation_strategy in (EvaluationStrategy.STEPS, IntervalStrategy.STEPS)
                        and self.global_step % self.args.eval_steps == 0
                    ):
                        self.evaluate(epoch=epoch)

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert (
                                model.module is self.model
                            ), f"Module {model.module} should be a reference to self.model"
                        else:
                            assert model is self.model, f"Model {model} should be a reference to self.model"
                        # Save model checkpoint
                        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}"
                        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                        self.store_flos()
                        self.save_model(output_dir)

                        if self.is_world_process_zero():
                            self._rotate_checkpoints(use_mtime=True)
                            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                else:
                    if not self.privacy_args.non_private:
                        self.optimizer.virtual_step(loss=losses.get("vector_loss"))

                epoch_pbar.update(1)
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

            epoch_pbar.close()
            train_pbar.update(1)

            if (
                self.args.evaluation_strategy in (EvaluationStrategy.EPOCH, IntervalStrategy.EPOCH) and
                (epoch + 1) % self.args.eval_epochs == 0
            ):
                metrics = self.evaluate(epoch=epoch)

            if self.args.max_steps is not None and 0 < self.args.max_steps <= self.global_step:
                break

        train_pbar.close()
        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step, metrics=dict())

    def log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
            iterator (:obj:`tqdm`, `optional`):
                A potential tqdm progress bar to write the logs on.
        """
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                logs["total_flos"] = self.total_flos
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        output = {
            **logs,
            **{
                "step": self.global_step,
                'num_params': self.num_params,
                'num_non_embedding_params': self.num_non_embedding_params
            }
        }
        if self.is_world_process_zero():
            self.log_history.append(output)
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.args.device)

        # GPT-2 don't use these; these are mostly for encoder-decoder architectures.
        inputs.pop('src_attn', None)
        inputs.pop('tgt_attn', None)
        inputs.pop('src', None)
        return inputs

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> dict:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)  # (batch_size,).

        vector_loss = loss
        scalar_loss = loss.mean(dim=0) / self.args.gradient_accumulation_steps

        if self.privacy_args.non_private:
            scalar_loss.backward()

        scalar_loss = scalar_loss.detach()
        return dict(vector_loss=vector_loss, scalar_loss=scalar_loss)

    def compute_loss(self, model, inputs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        seq_lens = (shift_labels != -100).sum(dim=1)
        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels, reduction="none")
        loss = loss.sum(dim=1) / seq_lens  # Per token loss.

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        return loss  # (batch_size,)

    def is_local_master(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
        several machines) main process.

        .. warning::

            This method is deprecated, use :meth:`~transformers.Trainer.is_local_process_zero` instead.
        """
        warnings.warn("This method is deprecated, use `Trainer.is_local_process_zero()` instead.", FutureWarning)
        return self.is_local_process_zero()

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
        several machines) main process.
        """
        return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on
        several machines, this is only going to be :obj:`True` for one process).

        .. warning::

            This method is deprecated, use :meth:`~transformers.Trainer.is_world_process_zero` instead.
        """
        warnings.warn("This method is deprecated, use `Trainer.is_world_process_zero()` instead.", FutureWarning)
        return self.is_world_process_zero()

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on
        several machines, this is only going to be :obj:`True` for one process).
        """
        return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_process_zero():
            self._save(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)  # Find the models in `train_dir/checkpoint-k/pytorch_model.bin`
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        json.dump(
            self.log_history, open(os.path.join(output_dir, "log_history.json"), "w"), indent=4, ensure_ascii=False
        )

    def store_flos(self):
        # Storing the number of floating-point operations that went into the model
        if self.total_flos is not None:
            if self.args.local_rank != -1:
                total_flos = distributed_broadcast_scalars([self.total_flos]).sum().item()
            else:
                total_flos = self.total_flos
            if total_flos > 0:
                self.model.config.total_flos = total_flos

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        output_dir_name = os.path.basename(self.args.output_dir)
        checkpoint_prefix = f"{output_dir_name}-{PREFIX_CHECKPOINT_DIR}"

        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(self, log_results=True, epoch=None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            log_results:
                Store the results in `self.log_history` and print to stdout.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """

        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
        eval_output = self.prediction_loop(eval_dataloader, description="Evaluate eval split")

        val_dataloader = self.get_eval_dataloader(self.val_dataset)
        val_output = self.prediction_loop(val_dataloader, description="Evaluate val split")

        train_sampler = self._get_train_sampler(shuffle=False)  # Don't shuffle during evaluation!
        train_dataloader = self.get_train_dataloader(train_sampler=train_sampler)
        train_output = self.prediction_loop(train_dataloader, description="Evaluate train split")

        metrics = {
            "train": train_output.metrics,
            "eval": eval_output.metrics,
            "val": val_output.metrics,
            "epoch": epoch,
            "lr": [pg["lr"] for pg in self.optimizer.param_groups],
        }

        if hasattr(self.optimizer, 'privacy_engine'):
            pe = self.optimizer.privacy_engine
            privacy_metrics = pe.get_privacy_spent(accounting_mode="all", lenient=True)
            privacy_stats = pe.get_training_stats()
            metrics = {**metrics, **privacy_metrics, **privacy_stats}

        # Generate with beam search.
        if not self.args.skip_generation:
            self.generate_and_write_to_file()

        if log_results:
            self.log(metrics)

            # Save log history always! This must appear after the `log_history` is updated.
            json.dump(
                self.log_history,
                open(os.path.join(self.args.output_dir, "log_history.json"), "w"),
                indent=2,
                ensure_ascii=False
            )

        return metrics

    def _get_loader_by_split(self, split):
        if split == "train":
            loader = self.get_train_dataloader()
        else:
            if split == "val":
                loader = self.get_eval_dataloader(self.val_dataset)
            elif split == "eval":
                loader = self.get_eval_dataloader(self.eval_dataset)
            else:
                raise ValueError(f"Unknown split: {split}")
        return loader

    def _get_prompt_dataset_by_split(self, split):
        return {
            "train": self.generation_stuff["train_prompts"],
            "val": self.generation_stuff["val_prompts"],
            "eval": self.generation_stuff["eval_prompts"],
        }[split]

    def generate_and_write_to_file(self, num_generations_to_print=6, **decoding_kwargs):
        # Pass in the additional decoding stuff from `decoding_kwargs`.

        models = (self.model,)
        model_tags = ("model",)
        all_generations = {model_tag: {} for model_tag in model_tags}

        for this_model, this_model_tag in utils.zip_(models, model_tags):
            kwargs = dict(model=this_model, tokenizer=self.tokenizer, device=self.args.device)
            this_generations = all_generations[this_model_tag]

            for split in ("train", "val", "eval"):
                # Don't use the loader to avoid duplicated prompts!
                prompt_dataset = self._get_prompt_dataset_by_split(split)
                if split == "train":  # Don't waste compute on sanity checks.
                    max_generations = self.args.max_generations_train
                elif split in ('val', 'valid'):  # Use val and valid interchangeably.
                    max_generations = self.args.max_generations_valid
                else:
                    max_generations = self.args.max_generations

                full_generations, unstripped_generations, generations, references = decoding_utils.generate(
                    prompt_dataset=prompt_dataset, max_generations=max_generations,
                    **kwargs, **decoding_kwargs
                )
                this_generations[split] = dict(
                    full_generations=full_generations,
                    unstripped_generations=unstripped_generations,
                    generations=generations,
                    references=references,
                )

                def pretty_format(lines):
                    """A useful helper to make printted generationed look nice."""
                    return '\n'.join([repr(line) for line in lines[:num_generations_to_print]])

                # Various visuals.
                print(f" --- split {split} --- ")
                print(f" *** full generations *** ")
                print(pretty_format(full_generations))
                print(f" *** unstripped generations *** ")
                print(pretty_format(unstripped_generations))
                print(f" *** generations *** ")
                print(pretty_format(generations))
                print(f" *** references *** ")
                print(pretty_format(references))
                print(f" *** num generations: {len(generations)}, num references: {len(references)} *** ")

                # Store generations for BLEU.
                counter = self.global_step if self.global_step is not None else -1
                generations_path = os.path.join(
                    self.args.output_dir,
                    f'generations_{this_model_tag}', f'{split}', f'global_step_{counter:08d}.txt'
                )
                os.makedirs(os.path.dirname(generations_path), exist_ok=True)
                with open(generations_path, 'w') as f:
                    f.writelines([line + '\n' for line in generations])
                logger.warning(f"Wrote generations to {generations_path}")

    def prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        assert not getattr(
            self.model.config, "output_attentions", False
        ), "The prediction loop does not work with `output_attentions=True`."
        assert not getattr(
            self.model.config, "output_hidden_states", False
        ), "The prediction loop does not work with `output_hidden_states=True`."

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        self.model.eval()
        models = (self.model,)
        model_tags = ("model",)

        def create_record():
            return dict(
                eval_losses=[], entropy_losses=[], tok_logprobs=[], lin_logprobs=[],
            )

        records = {model_tag: create_record() for model_tag in model_tags}
        preds = label_ids = None

        if self.args.past_index >= 0:
            self._past = None

        def eval_stats(inputs, loss, logits, labels):
            if loss is not None:
                batch_size = inputs['input_ids'].size(0)
                eval_loss = [loss] * batch_size
            else:
                eval_loss = [-1]

            if logits is not None:
                logits = logits[..., :-1, :]
                labels = labels[..., 1:]

                valid_locations = (labels != -100)
                all_log_probs = logits.log_softmax(dim=-1)  # (B, L, V).
                entropy = -(all_log_probs.exp() * all_log_probs).sum(dim=-1)  # (B, L).
                entropy = entropy[valid_locations]

                logprob = F.cross_entropy(logits.permute(0, 2, 1), labels, reduction="none")  # (B, L).
            else:
                entropy, logprob = [-1], [-1]

            return eval_loss, entropy, logprob

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        for batch_idx, inputs in tqdm(enumerate(dataloader), desc=description, disable=disable_tqdm):
            for this_model, this_model_tag in utils.zip_(models, model_tags):
                this_record = records[this_model_tag]
                loss, logits, labels = self.prediction_step(this_model, inputs, prediction_loss_only)
                eval_loss, entropy, logprob = eval_stats(inputs, loss, logits, labels)
                this_record["eval_losses"].extend(eval_loss)
                this_record["entropy_losses"].extend(entropy.tolist())
                this_record["tok_logprobs"].extend(logprob.view(-1).tolist())
                this_record["lin_logprobs"].extend(logprob.sum(dim=-1).view(-1).tolist())

            if 0 < self.args.max_eval_batches <= batch_idx + 1:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # lxuechen: I removed everything regarding distributed training.
        for record_key, record_value in records.items():
            this_record = records[record_key]
            for key, value in this_record.items():
                if isinstance(value, (list, tuple)):
                    this_record[key] = np.mean(value)

        metrics = records

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = all(inputs.get(k) is not None for k in self.args.label_names)
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss
            if has_labels:  # The .mean() is to reduce in case of distributed training
                loss = loss.mean().item()
            logits = outputs.logits

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = tuple(inputs.get(name).detach() for name in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        return loss, logits, labels

    def floating_point_ops(self, inputs: Dict[str, Union[torch.Tensor, Any]]):
        """
        For models that inherit from :class:`~transformers.PretrainedModel`, uses
        that method to compute the number of floating point operations for every backward + forward pass. If using
        another model, either implement such a method in the model or subclass and override this method.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """

        if isinstance(self.model, torch.nn.DataParallel) or isinstance(
            self.model, torch.nn.parallel.DistributedDataParallel
        ):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, "floating_point_ops"):
            return model.floating_point_ops(inputs)

        else:
            return 0
