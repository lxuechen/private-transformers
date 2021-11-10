"""Wrapper command for running e2e-metrics & gem heavy metrics."""
import os

import fire


def evaluate(
    gen_path,
    ref_path="/nlp/scr/lxuechen/data/prefix-tuning/data/e2e_data/clean_references_test.txt",
    # Clone the e2e-metrics repo to this dir if you haven't already: https://github.com/lxuechen/e2e-metrics
    e2e_dir=None,
    skip_coco=False,
    skip_mteval=False,
    out_path=None,
    **kwargs,
):
    """Evaluate a file of generate sentences against references."""
    if e2e_dir is None:
        e2e_dir = os.path.join(os.path.expanduser('~'), 'software', 'e2e-metrics')

    if out_path is None:
        os.system(
            f'cd {e2e_dir}; '
            f'./measure_scores.py {ref_path} {gen_path} '
            f'--skip_coco {skip_coco} --skip_mteval {skip_mteval} --python ; '
            f'cd -'
        )
    else:
        os.system(
            f'cd {e2e_dir}; '
            f'./measure_scores.py {ref_path} {gen_path} '
            f'--skip_coco {skip_coco} --skip_mteval {skip_mteval} --python --out_path {out_path} ; '
            f'cd -'
        )


def main(task="evaluate", **kwargs):
    if task == "evaluate":
        evaluate(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
