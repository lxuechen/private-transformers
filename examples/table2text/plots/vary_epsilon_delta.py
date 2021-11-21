"""
1. Check influence of delta.
2. Check varying epsilon
"""

import fire
from swissknife import utils


def vary_epsilon(
    seeds,
    percentage,
):
    base_dir = "/Users/xuechenli/Desktop/dump_a100/private-lm/date_111321"
    target_epsilons = (0.1, 0.5, 2, 3, 5, 8)

    errorbar = dict(x=target_epsilons, y=[], yerr=[], label="$\delta={10^{-5}}$")
    results = dict()
    for target_epsilon in target_epsilons:
        target_epsilon_str = utils.float2str(target_epsilon)
        sub_dir = utils.join(
            base_dir,
            f"tm_e2e_mn_gpt2_np_no_tm_full_pemgn_0_10000000_nm_None_lr_0_00200000_tbs_00001024_md_00000512_"
            f"psl_00000010_e_00000010_te_{target_epsilon_str}_td_0_00001000_r_1_lr_decay_no_optimizer_adam"
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

        res = utils.single_standard_deviation(sample=vals)
        errorbar['y'].append(res["mean"])
        errorbar['yerr'].append(res['delta'])

    img_path = utils.join('./vary_epsilon')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        errorbars=(errorbar,),
        options=dict(xlabel="$\epsilon$", ylabel="E2E test set BLEU", xscale="linear")
    )


def vary_delta(seeds, percentage):
    base_dir = "/Users/xuechenli/Desktop/dump_a100/private-lm/date_111221"
    target_epsilons = (3, 8)
    target_deltas = (1e-7, 1e-6, 1e-5)

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
                **utils.single_standard_deviation(sample=vals),
                "noise_multiplier": noise_multiplier,
            }

    errorbars = [
        dict(
            x=target_deltas,
            yerr=[results[(target_epsilon, target_delta)]['delta'] for target_delta in target_deltas],
            y=[results[(target_epsilon, target_delta)]['mean'] for target_delta in target_deltas],
            label=f"$\epsilon={target_epsilon}$",
        )
        for target_epsilon in target_epsilons
    ]
    img_path = utils.join('./vary_delta')
    utils.plot_wrapper(
        img_path=img_path,
        suffixes=('.png', '.pdf'),
        errorbars=errorbars,
        options=dict(xlabel="$\delta$", ylabel="E2E test set BLEU", xscale="log")
    )


def main(
    seeds=(0, 1, 2, 3, 4),
    percentage=True,
):
    vary_delta(seeds=seeds, percentage=percentage)
    vary_epsilon(seeds=seeds, percentage=percentage)


if __name__ == "__main__":
    fire.Fire(main)
