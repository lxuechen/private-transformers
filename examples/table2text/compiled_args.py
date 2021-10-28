"""Compilation of all the arguments."""
from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Optional

import transformers

MODEL_CONFIG_CLASSES = list(transformers.MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from "
                    "scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    static_lm_head: bool = field(default=False)
    static_embedding: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_folder: Optional[str] = field(default=None, metadata={"help": "Path to folder with all the data."})

    # Useful for truncating the dataset.
    max_train_examples: Optional[int] = field(default=sys.maxsize)
    max_valid_examples: Optional[int] = field(default=sys.maxsize)
    max_eval_examples: Optional[int] = field(default=sys.maxsize)

    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The name of the task."}
    )
    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )
    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )
    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special "
                    "tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_len: int = field(default=sys.maxsize)

    def __post_init__(self):
        if self.data_folder is not None:
            logging.warning(f'Overriding dataset paths using those given in `data_folder`')

            if self.task_mode == "e2e":
                self.train_data_file = os.path.join(self.data_folder, 'src1_train.txt')
                self.valid_data_file = os.path.join(self.data_folder, 'src1_valid.txt')
                self.eval_data_file = os.path.join(self.data_folder, 'src1_test.txt')

                self.train_prompt_file = os.path.join(self.data_folder, 'prompts_train.txt')
                self.val_prompt_file = os.path.join(self.data_folder, 'prompts_valid.txt')
                self.eval_prompt_file = os.path.join(self.data_folder, 'prompts_test.txt')

            elif self.task_mode == "dart":
                self.train_data_file = os.path.join(self.data_folder, 'dart-v1.1.1-full-train.json')
                self.valid_data_file = os.path.join(self.data_folder, 'dart-v1.1.1-full-dev.json')
                self.eval_data_file = os.path.join(self.data_folder, 'dart-v1.1.1-full-test.json')

                self.train_prompt_file = os.path.join(self.data_folder, 'prompts_train.txt')
                self.val_prompt_file = os.path.join(self.data_folder, 'prompts_valid.txt')
                self.eval_prompt_file = os.path.join(self.data_folder, 'prompts_test.txt')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_eval_batches: int = field(default=-1, metadata={"help": "Maximum number of evaluation steps to run."})
    max_generations: int = field(default=sys.maxsize)
    max_generations_train: int = field(default=10)
    max_generations_valid: int = field(default=10)
    skip_generation: str = field(default="no")

    ema_model_averaging: str = field(default="no")
    ema_model_gamma: float = field(default=0.99)
    ema_model_start_from: int = field(default=1000)
    lr_decay: str = field(default="yes")
    eval_epochs: int = field(default=10)

    evaluate_during_training: str = field(
        default="yes",
        metadata={"help": "Run evaluation during training at each logging step."},
    )
    evaluate_before_training: str = field(
        default="yes",
        metadata={"help": "Run evaluation before training."},
    )
    save_at_last: str = field(default="no", metadata={"help": "Save at the end of training."})

    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()
        self.skip_generation = self.skip_generation.lower() in ('y', 'yes')
        self.ema_model_averaging = (self.ema_model_averaging.lower() in ('y', 'yes'))
        self.lr_decay = (self.lr_decay.lower() in ('y', 'yes'))
        self.evaluate_during_training = (self.evaluate_during_training in ('y', 'yes'))
        self.evaluate_before_training = (self.evaluate_before_training in ('y', 'yes'))
        self.save_at_last = (self.save_at_last in ('y', 'yes'))


@dataclass
class PrivacyArguments:
    """Arguments for differentially private training."""
    per_example_max_grad_norm: float = field(
        default=.1, metadata={
            "help": "Clipping 2-norm of per-sample gradients."
        }
    )
    noise_multiplier: float = field(
        default=None, metadata={
            "help": "Standard deviation of noise added for privacy; if `target_epsilon` is specified, "
                    "use the one searched based budget"
        }
    )
    target_epsilon: float = field(
        default=None, metadata={
            "help": "Privacy budget; if `None` use the noise multiplier specified."
        }
    )
    target_delta: float = field(
        default=None, metadata={
            "help": "Lax probability in approximate differential privacy; if `None` use 1 / len(train_data)."
        }
    )
    accounting_mode: str = field(
        default="rdp_cks", metadata={"help": "One of `rdp`, `gdp`, `rdp_cks`, `glw`, `all`."}
    )
    non_private: str = field(default="no")
    ghost_clipping: str = field(default="no")

    def __post_init__(self):
        self.non_private = self.non_private.lower() in ('y', 'yes')
        self.ghost_clipping = self.ghost_clipping.lower() in ('y', 'yes')
