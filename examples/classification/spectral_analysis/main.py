import logging

import fire
import torch
import tqdm
from ml_swissknife import utils, numerical_distributed
from torch.utils.data import DataLoader, TensorDataset


def pca(
    # Place where grads are stored and where results will be stored.
    train_dir="/mnt/disks/disk-2/dump/privlm/roberta/sst-2",
    n=2000,  # How many checkpoints?
    k=1000,  # How many eigenvectors?
    num_power_iteration=10,
    batch_size=200,  # Batch size for processing the checkpoints in matmul.
    seed=42,  # Controls randomness in sampling the first vector in orthogonal iteration.
    start_index=0,  # The index of the first checkpoint to be selected.
    eval_steps=5,  # Evaluate PCA accuracy once this many iterations.
    save_steps=5,  # Save eigenvalue and eigenvector tensors once this many iterations.
    disable_tqdm=False,
):
    utils.manual_seed(seed)

    ckpt_dir = utils.join(train_dir, 'grad_trajectory')
    dump_dir = utils.join(train_dir, 'orthproj')

    all_ckpts = utils.all_ckpts(ckpt_dir, sort=True)
    tgt_ckpts = all_ckpts[start_index:start_index + n]
    dataset = torch.stack([
        torch.load(ckpt_path)["flat_grad"] for ckpt_path in tqdm.tqdm(tgt_ckpts, desc="load data")
    ])
    input_mat = DataLoader(dataset=TensorDataset(dataset), batch_size=batch_size)

    def callback(global_step, eigenvalues, eigenvectors):
        if global_step % save_steps == 0:
            utils.tsave(
                dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors),
                utils.join(dump_dir, "all", f"global_step_{global_step:06d}.pt")
            )
            utils.tsave(
                dict(eigenvalues=eigenvalues),
                utils.join(dump_dir, "eigenvalues", f"global_step_{global_step:06d}.evals")
            )
        if global_step % eval_steps == 0:
            err_abs, err_rel = numerical_distributed.check_error(
                input_mat=input_mat, eigenvectors=eigenvectors, disable_tqdm=disable_tqdm
            )
            logging.warning(f"global_step: {global_step}, abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

    numerical_distributed.orthogonal_iteration(
        input_mat=input_mat,
        k=k,
        num_power_iteration=num_power_iteration,
        dtype=torch.get_default_dtype(),
        callback=callback,
        disable_tqdm=disable_tqdm,
    )


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == '__main__':
    fire.Fire(main)
