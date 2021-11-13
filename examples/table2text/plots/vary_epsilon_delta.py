"""
1. Check influence of delta.
2. Check varying epsilon
"""

import fire
from swissknife import utils


def main(
    base_dir="/Users/xuechenli/Desktop/dump_a100/private-lm/date_111221",
    seeds=(0, 1, 2, 3, 4),
    target_epsilons=(3, 8),
    target_deltas=(1e-7, 1e-6, 1e-5),
    percentage=True,
):
    # Vary sigma.
    results = dict()
    for target_epsilon in target_epsilons:
        for target_delta in target_deltas:
            target_epsilon_str = utils.float2str(target_epsilon)
            target_delta_str = utils.float2str(target_delta)
            sub_dir = utils.join(
                base_dir,
                f"tm_e2e_mn_gpt2_np_no_tm_full_pemgn_0_10000000_nm_None_lr_0_00200000_tbs_00001024_md_00000512_psl_00000010_e_00000010_"
                f"te_{target_epsilon_str}_td_{target_delta_str}_r_1_lr_decay_no_optimizer_adam"
            )

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
                test_bleu = final_results["eval"]["model"]["BLEU"]
                if percentage:
                    test_bleu *= 100
                vals.append(test_bleu)
                noise_multiplier = argparse["noise_multiplier"]
                target_delta = argparse["target_delta"]

            results[(target_epsilon, target_delta)] = {
                **utils.confidence_interval(sample=vals),
                "noise_multiplier": noise_multiplier,
            }

    errorbars = [
        dict(
            x=target_deltas,
            yerr=[results[(target_epsilon, target_delta)]['delta'] for target_delta in target_deltas],
            y=[results[(target_epsilon, target_delta)]['mean'] for target_delta in target_deltas],
            label=f"$\epsilon={target_epsilon}$"
        )
        for target_epsilon in target_epsilons
    ]
    utils.plot_wrapper(
        errorbars=errorbars,
    )


if __name__ == "__main__":
    fire.Fire(main)
