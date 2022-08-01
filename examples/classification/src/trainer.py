########## The following part is copied from Transformers' trainer (3.4.0) ########## 

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import collections
import gc
import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from ml_swissknife import utils
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import EvaluationStrategy, IntervalStrategy, TrainOutput
from transformers.utils import logging

from .compiled_args import TrainingArguments

_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    pass

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)

if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    pass

if is_ray_available():
    pass

logger = logging.get_logger(__name__)


########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))


def default_dev_objective_key(metrics):
    """Get the key (name) for the specific metric used for dev."""
    keys = (
        "eval_mnli/acc",
        "eval_mnli-mm/acc",
        "eval_f1",
        "eval_mcc",
        "eval_pearson",
        "eval_acc"
    )
    for key in keys:
        if key in metrics:
            return key
    raise Exception("No metric founded for {}".format(metrics))


class Trainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def __init__(self, model_args=None, privacy_args=None, auxiliary_args=None, **kwargs):
        super(Trainer, self).__init__(**kwargs)
        self.privacy_args = privacy_args
        self.model_args = model_args
        self.auxiliary_args = auxiliary_args
        self.scaler = torch.cuda.amp.GradScaler(init_scale=128)

    # --- lxuechen: Not sure why v4.10.0 removed this function...
    def is_local_master(self) -> bool:
        if is_torch_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    # ---

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
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
        return self.optimizer, self.lr_scheduler

    def get_training_setup(self):
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )

            t_total_from_num_train_epochs = (
                int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            )
            assert t_total <= t_total_from_num_train_epochs, (
                "`num_train_epochs` give strict control (since it also controls the noise multiplier), "
                "`max_steps` should yield fewer steps"
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
        return dict(
            train_dataloader=train_dataloader,
            t_total=t_total,
            num_train_epochs=num_train_epochs
        )

    def train(self, model_path=None, dev_objective=None, dev_objective_key=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.local_rank != -1 or self.args.n_gpu > 1:
            raise ValueError("Multi-gpu and distributed training is currently not supported.")
        if self.args.fp16:
            raise ValueError("Mixed-precision training is currently not supported.")

        self.args: TrainingArguments

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = default_dev_objective if dev_objective is None else dev_objective
        self.dev_objective_key = default_dev_objective_key if dev_objective_key is None else dev_objective_key
        # --- lxuechen: Don't use self.state.log_history. Given implementation so convoluted...
        self.log_history = []
        # ---

        # Data loading.
        training_setup = self.get_training_setup()
        train_dataloader = training_setup["train_dataloader"]
        t_total = training_setup["t_total"]
        num_train_epochs = training_setup["num_train_epochs"]

        optimizer, scheduler = self.create_optimizer_and_scheduler(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )
                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0

        # --- low rank analysis project ---
        callback = None  # Default; don't store gradients or perform projection.

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

        # --- in case no training happens ---
        epoch = 0
        metrics = None
        # ---

        if self.args.evaluate_before_training:
            logging_loss_scalar = self.evaluate_and_log(
                tr_loss=tr_loss,
                logging_loss_scalar=logging_loss_scalar,
                scheduler=scheduler,
            )

        train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master())
        for epoch in train_iterator:
            # Clear gradient before entering a new epochs.
            # This is ultra important when using gradient accumulation in privacy training;
            # grads of micro batches could ooze.
            model.zero_grad(set_to_none=True)

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_master())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                losses = self.training_step(model, inputs)
                tr_loss += losses["scalar_loss"]

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps

                    # --- Don't do the update when this is the case. You get bad batch size for privacy ---
                    # len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    # and (step + 1) == len(epoch_iterator)
                    # ---
                ):
                    if self.privacy_args.non_private:
                        # Don't double clip in private learning.
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                    else:
                        if store_grads_dir is not None:
                            def callback(privacy_engine):
                                """Store clipped gradients for spectrum analysis."""
                                named_params = privacy_engine.named_params
                                flat_grad = torch.cat([param.summed_grad.flatten() for _, param in named_params])
                                torch.save(
                                    {"flat_grad": flat_grad.cpu().float()},
                                    utils.join(store_grads_dir, f'global_step_{self.global_step:06d}.ckpt')
                                )

                        vector_loss = losses.get("vector_loss")
                        self.optimizer.step(loss=vector_loss, callback=callback)

                    scheduler.step()
                    model.zero_grad(set_to_none=True)
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    metrics = None
                    if (
                        self.args.evaluation_strategy in (IntervalStrategy.STEPS, EvaluationStrategy.STEPS) and
                        self.global_step % self.args.eval_steps == 0
                    ):
                        logging_loss_scalar = self.evaluate_and_log(
                            tr_loss=tr_loss,
                            logging_loss_scalar=logging_loss_scalar,
                            scheduler=scheduler,
                        )
                else:
                    if not self.privacy_args.non_private:
                        self.optimizer.virtual_step(loss=losses.get("vector_loss"))  # noqa

                if 0 < self.args.max_steps < self.global_step:
                    epoch_iterator.close()
                    break

            if self.args.evaluation_strategy == IntervalStrategy.EPOCH and (epoch + 1) % self.args.eval_epochs == 0:
                logging_loss_scalar = self.evaluate_and_log(
                    tr_loss=tr_loss,
                    logging_loss_scalar=logging_loss_scalar,
                    scheduler=scheduler,
                )

            if 0 < self.args.max_steps < self.global_step:
                train_iterator.close()
                break

            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.evaluate_after_training:
            logger.info("Evaluate after training ends.")
            self.evaluate_and_log(
                tr_loss=tr_loss,
                logging_loss_scalar=logging_loss_scalar,
                scheduler=scheduler,
            )

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, metrics=metrics), self.objective

    def compute_loss(self, model, inputs, return_outputs=False, return_vector_loss=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)  # This should not contain `loss`.
        logits = outputs[0]
        if isinstance(outputs, SequenceClassifierOutput):
            outputs = (logits,)
        loss = F.cross_entropy(logits, labels, reduction="none")  # (batch_size,).
        if not return_vector_loss:
            loss = loss.mean(dim=0)
        return (loss, (loss,) + outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> dict:
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs, return_vector_loss=True)  # (batch_size,).

        vector_loss = loss
        scalar_loss = loss.mean(dim=0) / self.args.gradient_accumulation_steps

        if self.privacy_args.non_private:
            scalar_loss.backward()

        scalar_loss = scalar_loss.detach()
        return dict(vector_loss=vector_loss, scalar_loss=scalar_loss)

    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the 
    logits)
    """

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output

    def evaluate_and_log(
        self,
        tr_loss,
        logging_loss_scalar,
        scheduler,
    ):
        # lxuechen: Defaults to use .eval_dataset, which is set to 'dev'.
        output = self.evaluate()
        metrics = output.metrics

        objective = self.dev_objective(metrics)
        objective_key = self.dev_objective_key(metrics)

        # --- lxuechen: Print the metrics in a pretty format.
        print('metrics: ')
        print(json.dumps(metrics, indent=4))
        print(f'dev objective {objective_key}: {objective}')
        # ---

        if objective > self.objective:
            logger.info("Best dev result: {}".format(objective))
            self.objective = objective
            self.save_model(self.args.output_dir)

        # --- lxuechen: Combine logging and evaluation
        logs = dict(dev=metrics)

        tr_loss_scalar = tr_loss.item()
        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
        # backward compatibility for pytorch schedulers
        logs["learning_rate"] = (
            scheduler.get_last_lr()[0]
            if version.parse(torch.__version__) >= version.parse("1.4")
            else scheduler.get_lr()[0]
        )
        logging_loss_scalar = tr_loss_scalar

        if not self.privacy_args.non_private:
            logs["get_training_stats"] = self.optimizer.get_training_stats()
            logs["privacy_spent"] = self.optimizer.get_privacy_spent(accounting_mode="all", lenient=True)

        logs["epoch"] = self.epoch
        logs["step"] = self.global_step
        self.log_history.append(logs)

        # Write to disk!
        utils.jdump(self.log_history, os.path.join(self.args.output_dir, 'log_history.json'))
        # ---

        # ---
        # Evaluate gradient covariance spectrum for the dimension-dependence analysis project.
        if self.auxiliary_args.eval_spectrum:
            from ..spectrum import spectrum_utils

            def loss_fn(batch, model):
                batch = self._prepare_inputs(inputs=batch)
                return self.compute_loss(
                    model=model, inputs=batch, return_outputs=False, return_vector_loss=True
                )

            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float64)  # Slow but accurate.
            self.model.to(dtype=torch.float64)

            spectrum_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                shuffle=False,  # Must not shuffle.
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

            # No per-sample grads accumulated here, since `model.eval()` called internally.
            spectrum_outputs = spectrum_utils.make_spectrum_lanczos(
                loader=spectrum_loader,
                model=self.model,
                max_batches=self.auxiliary_args.max_spectrum_batches,
                max_lanczos_iter=self.auxiliary_args.max_lanczos_iter,
                loss_fn=loss_fn,
                return_dict=True,
                verbose=True,
            )

            state_dicts = {
                key: value.cpu().float() if torch.is_tensor(value) else value
                for key, value in spectrum_outputs.items()
            }
            utils.tsave(
                state_dicts,
                utils.join(self.args.output_dir, 'spectrum', f'global_step_{self.global_step:06d}.pt')
            )

            torch.set_default_dtype(default_dtype)
            self.model.to(dtype=default_dtype)

            del spectrum_outputs, state_dicts
            gc.collect()
            torch.cuda.empty_cache()
        # ---

        # ---
        # Store grad params.
        state_dicts = dict()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                state_dicts[name] = param.data.cpu().float()
        utils.tsave(
            state_dicts,
            utils.join(self.args.output_dir, 'grad_params', f'global_step_{self.global_step:06d}.pt')
        )
        # ---

        return logging_loss_scalar
