from ml_swissknife import utils

if __name__ == "__main__":
    # python -m classification.runs.main
    commands = []
    command = 'python -m classification.numerical \
        --task "pca" \
        --n 2000 \
        --k 1000 \
        --train_dir "/home/t-lc/dump/privlm/rebuttal/run-roberta-base" \
        --num_power_iteration 10'
    commands.append(command)

    command = 'python -m classification.numerical \
        --task "pca" \
        --n 2000 \
        --k 1000 \
        --train_dir "/home/t-lc/dump/privlm/rebuttal/run-roberta-large" \
        --batch_size 40 \
        --num_power_iteration 10'
    commands.append(command)

    utils.gpu_scheduler(commands, excludeID=(0,), log=False)
