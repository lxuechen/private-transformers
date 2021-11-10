"""Get the best hyper-parameter setting for various tuning modes; also create train directory in a principled manner."""

import logging
import sys

from swissknife import wrapper, utils


def get_best_hyper_params(
    tuning_mode, task_mode, non_private, target_epsilon,
    seed, model_name_or_path, date,
    gpu=None,
    **additional_kwargs,
):
    # Use these configs to
    #   1) directly get the command via `shared.get_command`,
    #   2) directly get `train_dir` (for later processing logs, json).

    if non_private.lower() in ('y', 'yes'):  # If non-private, train with gradient clipping but no noise.
        default_kwargs = dict(
            non_private="no",
            per_example_max_grad_norm=0.1,
            noise_multiplier=0.0,
            target_epsilon=-1,  # Signal that this is non-private.
            per_device_train_batch_size=8,
            per_device_eval_batch_size=10,
            max_eval_batches=100,
        )
    else:
        default_kwargs = dict(
            non_private="no",
            per_example_max_grad_norm=0.1,
            target_epsilon=target_epsilon,
            eval_examples=100,
        )
        if task_mode == "e2e":
            # default when gpu in 3090, titanx, titanxp, titanrtx.
            if gpu in ('a100',):
                if model_name_or_path in ("distilgpt2", "gpt2", 'gpt2-medium'):
                    per_device_train_batch_size = 32
                    per_device_eval_batch_size = 50
                else:
                    per_device_train_batch_size = 16
                    per_device_eval_batch_size = 25
            else:
                if model_name_or_path in ("distilgpt2", "gpt2", 'gpt2-medium'):
                    per_device_train_batch_size = 8
                    per_device_eval_batch_size = 20
                else:
                    per_device_train_batch_size = 4
                    per_device_eval_batch_size = 10
        elif task_mode == "dart":
            # default when gpu in 3090, titanx, titanxp, titanrtx.
            if model_name_or_path in ("distilgpt2", "gpt2", 'gpt2-medium'):
                per_device_train_batch_size = 8
                per_device_eval_batch_size = 10
            else:
                per_device_train_batch_size = 4
                per_device_eval_batch_size = 10

            # TODO: Improve efficiency via tweaking batch size.
            if gpu in ('a100',):
                per_device_train_batch_size *= 2
        else:
            raise ValueError(f'Unknown task_mode: {task_mode}')

        default_kwargs["per_device_train_batch_size"] = per_device_train_batch_size
        default_kwargs["max_eval_batches"] = default_kwargs["eval_examples"] // per_device_eval_batch_size
        default_kwargs["per_device_eval_batch_size"] = per_device_eval_batch_size

    tuning_mode_to_hyper_params = {
        "full": dict(
            train_batch_size=1024,
            learning_rate=2e-3,
            epochs=10,
            lr_decay="no",
            static_lm_head=False,
            static_embedding=False,
        ),
        "scratch": dict(
            train_batch_size=1024,
            learning_rate=1e-3,
            epochs=10,
            lr_decay="no",
            static_lm_head=True,
            static_embedding=True,
        ),
        "prefix": dict(
            train_batch_size=1024,
            learning_rate=1e-3,
            epochs=10,
            lr_decay="no",
            preseqlen=10,
            mid_dim=512,
        ),
        "rgp": dict(
            train_batch_size=512,
            learning_rate=1e-3,
            epochs=10,
            lr_decay="no",
            rank=1,
        ),
        "top2": dict(
            train_batch_size=512,
            learning_rate=5e-4,
            lr_decay="yes",
            epochs=50,
        ),
        "lora": dict(
            train_batch_size=512,
            learning_rate=1e-3,
            lr_decay="no",
            epochs=50,
            rank=4,
        ),
    }
    kwargs = {
        **tuning_mode_to_hyper_params[tuning_mode],
        **default_kwargs,
        **additional_kwargs,

        "seed": seed,
        "model_name_or_path": model_name_or_path,
        "date": date,

        "task_mode": task_mode,
        "tuning_mode": tuning_mode,
    }
    if kwargs["train_batch_size"] < kwargs["per_device_train_batch_size"]:
        kwargs["per_device_train_batch_size"] = kwargs["train_batch_size"]
    kwargs["gradient_accumulation_steps"] = kwargs["train_batch_size"] // kwargs["per_device_train_batch_size"]
    return kwargs


get_best_hparams = get_best_hyper_params  # Alias for backwards compatibility.


def make_train_dir_from_kwargs(
    date,
    task_mode,
    model_name_or_path,
    non_private,
    tuning_mode,
    per_example_max_grad_norm,
    learning_rate,
    train_batch_size,
    epochs,
    target_epsilon,
    lr_decay,
    seed,

    # Reasonable defaults.
    noise_multiplier=-1,
    rank=1,
    mid_dim=512,
    preseqlen=10,
    train_dir=None,
    base_dir=None,
    **kwargs,
):
    if len(kwargs) > 0:
        logging.warning(f"Unknown kwargs: {kwargs}")

    if base_dir is None:
        base_dir = "/nlp/scr/lxuechen/prefixtune"

    # Standardize argument so that directories will be sorted.
    learning_rate_str = wrapper.float2str(learning_rate)
    per_example_max_grad_norm_str = wrapper.float2str(per_example_max_grad_norm)
    noise_multiplier_str = wrapper.float2str(noise_multiplier)
    train_batch_size_str = wrapper.int2str(train_batch_size)
    mid_dim_str = wrapper.int2str(mid_dim)
    preseqlen_str = wrapper.int2str(preseqlen)
    epochs_str = wrapper.int2str(epochs)
    target_epsilon_str = wrapper.int2str(target_epsilon)

    if non_private == "no":
        if train_dir is None:
            train_dir = utils.join(
                base_dir,
                f"date_{date}",
                f"tm_{task_mode}_"
                f"mn_{model_name_or_path}_"
                f"np_{non_private}_"
                f"tm_{tuning_mode}_"
                f"pemgn_{per_example_max_grad_norm_str}_"
                f"nm_{noise_multiplier_str}_"
                f"lr_{learning_rate_str}_"
                f"tbs_{train_batch_size_str}_"
                f"md_{mid_dim_str}_"
                f"psl_{preseqlen_str}_"
                f"e_{epochs_str}_"
                f"te_{target_epsilon_str}_"
                f"r_{rank}_"
                f"lr_decay_{lr_decay}",
                f"{seed}"
            )
    else:
        if train_dir is None:
            train_dir = utils.join(
                base_dir,
                f"date_{date}",
                f"tm_{task_mode}_"
                f"mn_{model_name_or_path}_"
                f"np_{non_private}_"
                f"tm_{tuning_mode}_"
                f"lr_{learning_rate_str}_"
                f"tbs_{train_batch_size_str}_"
                f"md_{mid_dim_str}_"
                f"pql_{preseqlen_str}_"
                f"e_{epochs_str}_"
                f"te_{target_epsilon_str}_"
                f"r_{rank}_"
                f"lr_decay_{lr_decay}",
                f"{seed}"
            )
    assert train_dir is not None
    return train_dir


def get_command(
    seed,
    tuning_mode,
    non_private,
    date=None,  # Always include this so as to not mess up the folders.

    epochs=5,
    train_batch_size=5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=1,
    per_example_max_grad_norm=1.,
    noise_multiplier=None,
    learning_rate=1e-05,
    rank=1,

    # Use 100 for E2E and 120 for DART.
    max_seq_len=None,
    max_generations_train=10,
    max_generations_valid=sys.maxsize,
    max_generations=sys.maxsize,
    skip_generation="no",

    max_train_examples=sys.maxsize,
    max_valid_examples=sys.maxsize,
    max_eval_examples=sys.maxsize,

    eval_steps=100,  # Evaluate every such steps.
    eval_epochs=10,
    max_steps=-1,
    max_eval_batches=-1,
    mid_dim=512,
    preseqlen=10,
    save_steps=500000000,  # Essentially don't save.
    # submit => run on nlp cluster, auto make dir, wraps command with slurm options
    # gvm => run on google cloud virtual machine, auto make dir
    # local => run locally, makes a generic dir at ../test/
    mode="submit",
    model_type="gpt2",
    model_name_or_path="distilgpt2",  # 80+million
    tokenizer_name="gpt2",

    script="table2text.run_language_modeling",
    train_dir=None,
    ema_model_start_from=1000,
    ema_model_averaging="no",
    ghost_clipping="no",
    evaluation_strategy="epoch",
    evaluate_before_training="yes",
    fp16=False,
    static_lm_head=False,
    static_embedding=False,

    # -1 is just a default value.
    target_epsilon=-1,
    target_delta=-1,
    task_mode="e2e",
    lr_decay="yes",
    save_at_last="no",

    data_folder=None,  # Defaults to the clean data (no canary) based on task mode; see the if-else checklist below.

    gpu=None,  # Randomly grab.
    conda_env="lxuechen-private-lm-gen-release",
    priority="standard",
    time="3-0",
    hold_job=True,
    logs=True,
    **kwargs,
):
    if kwargs:
        print(f"unknown kwargs: {kwargs}")

    if mode == wrapper.Mode.submit and date is None:
        raise ValueError(f"`date` cannot be None when submitting.")

    if train_batch_size // per_device_train_batch_size != gradient_accumulation_steps:
        raise ValueError(
            "`train_batch_size`, `per_device_train_batch_size` and `gradient_accumulation_steps`"
            "don't match up"
        )

    # Less than 1 / (2 * dataset size).
    if target_delta < 0:
        if task_mode == "e2e":
            target_delta = 1e-5
        elif task_mode == "webnlg":
            target_delta = 2.5e-5
        elif task_mode == "dart":
            target_delta = 8e-6
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")

    if max_seq_len is None:
        if task_mode == "e2e":
            max_seq_len = 100
        elif task_mode == "dart":
            max_seq_len = 120
        else:
            raise ValueError(f"Unknown task_mode: {task_mode}")

    # Check mode.
    if mode in (wrapper.Mode.gvm, wrapper.Mode.submit):
        train_dir = make_train_dir_from_kwargs(
            date=date,
            task_mode=task_mode,
            model_name_or_path=model_name_or_path,
            non_private=non_private,
            tuning_mode=tuning_mode,
            per_example_max_grad_norm=per_example_max_grad_norm,
            noise_multiplier=noise_multiplier,
            learning_rate=learning_rate,
            train_batch_size=train_batch_size,
            mid_dim=mid_dim,
            preseqlen=preseqlen,
            epochs=epochs,
            target_epsilon=target_epsilon,
            rank=rank,
            lr_decay=lr_decay,
            seed=seed,
        )
    else:
        if train_dir is None:
            # Local debugging.
            train_dir = "/nlp/scr/lxuechen/tests/table2text"

    if data_folder is None:
        if task_mode == "e2e":
            data_folder = "/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data"
        elif task_mode == "webnlg":
            data_folder = "/nlp/scr/lxuechen/data/prefix-tuning/data/webnlg_challenge_2017"
        elif task_mode == "dart":  # dart.
            data_folder = "/nlp/scr/lxuechen/data/prefix-tuning/data/dart"
        else:
            raise ValueError

    logging_dir = train_dir
    command = f'python -m {script} \
        --output_dir {train_dir} \
        --task_mode {task_mode} \
        --model_type {model_type} \
        --model_name_or_path {model_name_or_path} \
        --tokenizer_name {tokenizer_name} \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --per_device_eval_batch_size {per_device_eval_batch_size} \
        --do_train \
        --do_eval \
        --line_by_line \
        --save_steps {save_steps} \
        --save_total_limit 1 \
        --save_at_last {save_at_last} \
        --data_folder {data_folder} \
        --logging_dir {logging_dir} \
        --logging_steps -1 \
        --gradient_accumulation_steps {gradient_accumulation_steps} \
        --learning_rate {learning_rate} \
        --weight_decay 0.0 \
        --seed {seed} \
        --evaluate_during_training "yes" \
        --eval_steps {eval_steps} \
        --eval_epochs {eval_epochs} \
        --non_private {non_private} \
        --cache_dir /nlp/scr/lxuechen/hfcache/control/gpt2/ \
        --max_steps {max_steps} \
        --max_eval_batches {max_eval_batches} \
        --evaluation_strategy {evaluation_strategy} \
        --evaluate_before_training {evaluate_before_training} \
        --per_example_max_grad_norm {per_example_max_grad_norm} \
        --max_seq_len {max_seq_len} \
        --max_generations {max_generations} \
        --max_generations_train {max_generations_train} \
        --max_generations_valid {max_generations_valid} \
        --max_train_examples {max_train_examples} \
        --max_valid_examples {max_valid_examples} \
        --max_eval_examples {max_eval_examples} \
        --ema_model_averaging {ema_model_averaging} \
        --ema_model_start_from {ema_model_start_from} \
        --ghost_clipping {ghost_clipping} \
        --target_delta {target_delta} \
        --target_epsilon {target_epsilon} \
        --overwrite_output_dir \
        --lr_decay {lr_decay} \
        --num_train_epochs {epochs} \
        --skip_generation {skip_generation} '
    if noise_multiplier is not None:
        command += f'--noise_multiplier {noise_multiplier} '
    if fp16 or (isinstance(fp16, str) and fp16.lower() in ('yes', 'y')):
        command += "--fp16 "
    if static_lm_head:
        command += "--static_lm_head "
    if static_embedding:
        command += "--static_embedding "

    if mode == wrapper.Mode.submit:
        command = wrapper.mynlprun_wrapper(
            command,
            train_dir=train_dir,
            gpu=gpu,
            conda_env=conda_env,
            priority=priority,
            time=time,
            hold_job=hold_job,
        )
        command += "\n\n"
    elif logs:
        logs_path = utils.join(train_dir, 'logs.out')
        command += f" > {logs_path} 2>&1 "
        command = f"mkdir -p {train_dir}; \n{command}"
    return command
