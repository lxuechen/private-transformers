# python -m classification.launchers.spectrum_050822
# try single precision
import fire


def _get_cmd(random_init, dump_path):
    return f'''python -m classification.spectrum \
        --model_name_or_path "distilroberta-base" \
        --random_init {random_init} \
        --dump_path {dump_path} \
        --dtype "float32"
        '''


def main():
    cmds = []
    for random_init in (True, False):
        dump_path = f"/mnt/disks/disk-2/dump/spectrum/init_compare_float32/dump_{random_init}.pt"
        cmds.append(_get_cmd(random_init=random_init, dump_path=dump_path))
    from swissknife import utils
    utils.gpu_scheduler(commands=cmds, excludeID=(0, 1), excludeUUID=(0, 1))


if __name__ == "__main__":
    fire.Fire(main)
