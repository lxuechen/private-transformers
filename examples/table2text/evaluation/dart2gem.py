"""Convert the dart reference and generation files to GEM format."""

import logging
import os
import sys
from typing import Optional, Sequence
import uuid

import fire
from swissknife import utils


def convert_ref(
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-test.json",
    out_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/gem-dart-v1.1.1-full-test.json"
):
    """Convert the references to the ref file format needed for GEM-metrics."""
    ref_dict = dict(language="en", values=[])
    data = utils.jload(ref_path)

    for i, example in enumerate(data):
        if len(example['annotations']) == 0:  # Only prompt but with no annotation!
            continue

        targets = [annotation["text"] for annotation in example['annotations']]
        ref_dict["values"].append(
            {
                "target": targets,
                "gem_id": i  # Still the sequential id.
            }
        )
    utils.jdump(ref_dict, out_path)


def convert_gen(
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/dart-v1.1.1-full-test.json",
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0720"
            "/model_name_gpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/generations_model/eval/",
    output_dir=None,
):
    """Convert the generations to be the out file format needed GEM-metrics.

    Outputs to `gen_dir/../../gem_generations_model/eval`
    """
    data = utils.jload(ref_path)

    parpardir = os.path.abspath(os.path.join(gen_dir, os.pardir, os.pardir))
    if output_dir is None:
        output_dir = os.path.join(parpardir, 'gem_generations_model', 'eval')
    os.makedirs(output_dir, exist_ok=True)

    for gen_path in utils.listfiles(gen_dir):
        base_name = os.path.basename(gen_path)
        new_gen_path = os.path.join(output_dir, base_name)

        with open(gen_path, 'r') as f:
            generations = f.readlines()

        gen_dict = dict(language="en", task="table2text", values=[])
        counter = 0  # Index the generation file.
        for idx, example in enumerate(data):
            if len(example['annotations']) == 0:  # Only prompt but with no annotation!
                continue

            gen_dict["values"].append(
                {
                    "generated": generations[counter],
                    "gem_id": idx,
                }
            )
            counter += 1
        assert counter == len(generations)
        utils.jdump(gen_dict, new_gen_path)


def convert_gen_single(gen_path, out_path, ref_path):
    data = utils.jload(ref_path)

    with open(gen_path, 'r') as f:
        generations = f.readlines()

    gen_dict = dict(language="en", task="table2text", values=[])
    counter = 0  # Index the generation file.
    for idx, example in enumerate(data):
        if len(example['annotations']) == 0:  # Only prompt but with no annotation!
            continue

        gen_dict["values"].append(
            {
                "generated": generations[counter],
                "gem_id": idx,
            }
        )
        counter += 1
    assert counter == len(generations)
    utils.jdump(gen_dict, out_path)


def eval_dir(
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/dart/gem-dart-v1.1.1-full-test.json",
    gem_dir="/sailhome/lxuechen/playground/GEM-metrics",
    scratch_dir=None,  # Mess around here.

    global_steps: Optional[Sequence[int]] = None,
    gen_dir="/nlp/scr/lxuechen/prefixtune/date_0720"
            "/model_name_gpt2_nonprivate_no_tuning_mode_scratchtune_per_example_max_grad_norm_0_10000000_noise_multiplier_-1_00000000_learning_rate_0_00050000_train_batch_size_00000512_mid_dim_00000512_preseqlen_00000010_epochs_00000050_target_epsilon_00000008/0/gem_generations_model/eval/",
    img_dir="/nlp/scr/lxuechen/plots/distilgpt2-e2e-nonprivate",
    unwanted_keys=("predictions_file", "N", "references_file"),
    max_files=sys.maxsize,
    # Original list.
    # metric_list=('bleu', 'rouge', "nist", "meteor", "bertscore", "bleurt"),

    # List with `meteor` disabled in GEM; it's strikingly slow, not sure why?!!
    metric_list=('bleu', 'rouge', "nist", "bertscore", "bleurt"),

    python_path="/u/nlp/anaconda/main/anaconda3/envs/lxuechen-gem/bin/python",
):
    assert isinstance(metric_list, (list, tuple))

    if not os.path.exists(gen_dir):
        logging.warning(f"`gen_dir` doesn't exists")
        return

    if global_steps is None:
        import re
        global_steps = []
        for file in utils.listfiles(gen_dir):
            search = re.search(".*global_step_(.*).txt", file)
            if search:
                global_step = int(search.group(1))
                global_steps.append(global_step)
        global_steps.sort()
    global_steps = global_steps[:max_files]
    logging.warning(f"evaluating {len(global_steps)} files")

    for global_step in global_steps:
        gen_path = os.path.join(gen_dir, f"global_step_{global_step:08d}.txt")
        assert os.path.exists(gen_path), f"Failed to find path {gen_path}"
        del gen_path

    logging.warning(f"eval_trajectory for gen_dir {gen_dir}")

    if scratch_dir is None:
        # Ensure there's no corruption across different jobs.
        scratch_dir = f"/nlp/scr/lxuechen/scratch/tmp-{str(uuid.uuid4())}"

    os.makedirs(scratch_dir, exist_ok=True)
    scores = []
    for global_step in global_steps:
        gen_path = os.path.join(gen_dir, f"global_step_{global_step:08d}.txt")
        out_path = os.path.join(scratch_dir, f'global_step_{global_step:08d}.json')
        logging.warning(f'eval for {gen_path}')
        all_metrics = ' '.join(metric_list)
        os.system(
            f'cd {gem_dir}; {python_path} ./run_metrics.py -r {ref_path} {gen_path} '
            f'--metric-list {all_metrics} '
            f'-o {out_path} ; cd -'
        )

        score = utils.jload(out_path)
        scores.append(score)
        del score
    # This code seems to be correct, but don't know why it causes an error...
    # import shutil
    # shutil.rmtree(scratch_dir)

    metrics = list(scores[0].keys())
    for unwanted_key in unwanted_keys:
        if unwanted_key in metrics:
            metrics.remove(unwanted_key)

    for metric in metrics:
        x = global_steps
        y = [score[metric] for score in scores]
        if metric in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
            y = [y_i["fmeasure"] for y_i in y]
        elif metric == "bertscore":
            y = [y_i["f1"] for y_i in y]
        img_path = os.path.join(img_dir, f"{metric}.png")
        utils.plot(
            plots=({'x': x, 'y': y, 'label': metric},),
            options={"xlabel": "steps", "ylabel": "metric"},
            img_path=img_path,
        )
        del img_path

    results_path = os.path.join(img_dir, 'results.json')
    results = {"global_step": global_steps, "score": scores}
    utils.jdump(results, results_path)


def eval_single(
    gen_path, ref_path, out_path, python_path,
    scratch_dir=None,
    gem_dir="/sailhome/lxuechen/playground/GEM-metrics",
    metric_list=('bleu', 'rouge', "nist", "bertscore", "bleurt"),
):
    assert isinstance(metric_list, (list, tuple))

    if scratch_dir is None:
        # Ensure there's no corruption across different jobs.
        scratch_dir = f"/nlp/scr/lxuechen/scratch/tmp-{str(uuid.uuid4())}"

    os.makedirs(scratch_dir, exist_ok=True)
    all_metrics = ' '.join(metric_list)
    os.system(
        f'cd {gem_dir}; '
        f'{python_path} ./run_metrics.py -r {ref_path} {gen_path} '
        f'--metric-list {all_metrics} -o {out_path}; '
        f'cd -'
    )


def main(task="convert_ref", **kwargs):
    if task == "convert_ref":
        convert_ref(**kwargs)
    elif task == "convert_gen":
        convert_gen(**kwargs)
    elif task == "convert_gen_single":
        convert_gen_single(**kwargs)

    elif task == "eval_dir":
        eval_dir(**kwargs)
    elif task == "eval_single":
        eval_single(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
