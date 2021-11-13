"""Check with different optimizer.

Performance for epsilon = 8:
    maxx: 3.0, maxy: 63.17575925848472
"""

import fire
import numpy as np
from swissknife import utils


def main(base_dir="/Users/xuechenli/Desktop/dump_a100/private-lm/date_111021", seeds=(0, 1, 2), percentage=True):
    x2y = dict()
    for sub_dir in utils.listdir(base_dir, full_path=True):
        vals = []
        for seed in seeds:
            path = utils.join(sub_dir, f'{seed}')
            argparse_path = utils.join(path, 'argparse.json')
            final_results_path = utils.join(path, 'final_results.json')

            if (not utils.pathexists(path) or
                not utils.pathexists(argparse_path) or
                not utils.pathexists(final_results_path)):
                continue

            argparse = utils.jload(argparse_path)
            final_results = utils.jload(final_results_path)
            learning_rate = argparse["learning_rate"]
            test_bleu = final_results["eval"]["model"]["BLEU"]
            if percentage:
                test_bleu *= 100
            vals.append(test_bleu)
        x2y[learning_rate] = np.mean(test_bleu)

    x = []
    y = []
    keys = list(x2y.keys())
    keys.sort()
    for key in keys:
        x.append(key)
        y.append(x2y[key])

    y2x = {y: x for x, y in x2y.items()}
    maxy = max(list(x2y.values()))
    maxx = y2x[maxy]
    print(f'maxx: {maxx}, maxy: {maxy}')

    plots = [dict(x=x, y=y)]
    utils.plot_wrapper(plots=plots)


if __name__ == "__main__":
    fire.Fire(main)
