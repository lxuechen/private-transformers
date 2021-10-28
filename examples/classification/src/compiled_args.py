from dataclasses import dataclass, field

import transformers


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
    non_private: str = field(
        default="yes", metadata={"help": "Train non-privately if True."}
    )
    accounting_mode: str = field(
        default="rdp_cks", metadata={"help": "One of (`rdp`, `gdp`, `rdp_cks`, `all`)."}
    )
    ghost_clipping: str = field(default="no")

    def __post_init__(self):
        self.non_private = self.non_private.lower() in ('y', 'yes')
        self.ghost_clipping = self.ghost_clipping.lower() in ('y', 'yes')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    eval_epochs: int = field(default=10, metadata={"help": "Evaluate once such epochs"})
    evaluate_before_training: bool = field(default=False, metadata={"help": "Run evaluation before training."})
    lr_decay: str = field(
        default="no", metadata={"help": "Apply the usual linear decay if `yes`, otherwise no deacy."}
    )
    evaluate_test_split: bool = field(default=False, metadata={"help": "Run evaluation on the test split"})

    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()
        self.lr_decay = self.lr_decay.lower() in ('y', 'yes')
