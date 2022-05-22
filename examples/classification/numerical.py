"""
Numerical algorithms.

Currently, only contains simultaneous iter.
"""
import logging
import math
import sys
from typing import Optional, Tuple

import fire
from swissknife import utils
import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm


def load_data(ckpts_dir, num_ckpts, batch_size):
    all_ckpts = utils.all_ckpts(ckpts_dir, sort=True)[:num_ckpts]
    dataset = torch.stack([
        torch.load(ckpt_path)["flat_grad"]
        for ckpt_path in tqdm.tqdm(all_ckpts, desc="load data")
    ])
    dataset = TensorDataset(dataset)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    return loader


def pca(
    grads_dir="/mnt/disks/disk-2/dump/privlm/roberta/sst-2/grad_trajectory",
    dump_dir="/mnt/disks/disk-2/dump/privlm/roberta/sst-2/orthproj",
    n=2000,
    k=2000,
    num_power_iteration=100,
    batch_size=200,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = load_data(ckpts_dir=grads_dir, num_ckpts=n, batch_size=batch_size)
    get_bases(
        loader=loader,
        k=k,
        num_power_iteration=num_power_iteration, device=device, dump_dir=dump_dir,
    )


def _mem_saving_matmul(
    loader: DataLoader,
    eigenvectors: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    disable_tqdm: bool,
):
    out = torch.zeros_like(eigenvectors)
    nsteps = int(math.ceil(eigenvectors.size(1) / chunk_size))
    for idx in tqdm.tqdm(range(nsteps), desc="matmul", disable=disable_tqdm):
        start_idx = int(idx * chunk_size)
        chunk = eigenvectors[:, start_idx:start_idx + chunk_size].to(device)  # GPU. (p, k1).
        this_out = torch.zeros_like(chunk)  # GPU. (p, k1).
        for (batch,) in loader:
            batch = batch.to(device)
            this_out += torch.mm(batch.T, torch.mm(batch, chunk))
        out[:, start_idx:start_idx + chunk_size] = this_out.cpu()
    return out


def _eigenvectors_to_eigenvalues(
    loader: DataLoader, eigenvectors: torch.Tensor,
    chunk_size: int,  # Number of eigenvalues to process at once.
    device: torch.device,
    disable_tqdm: bool,
):
    nums = []
    dens = []

    nsteps = int(math.ceil(eigenvectors.size(1) / chunk_size))
    for idx in tqdm.tqdm(range(nsteps), desc="evec2eval", disable=disable_tqdm):
        start_idx = int(idx * chunk_size)

        chunk = eigenvectors[:, start_idx: start_idx + chunk_size].to(device)  # (p, ki).
        dens.append((chunk ** 2.).sum(dim=0))

        num = torch.zeros(size=(chunk.size(1),), device=device)
        for (batch,) in loader:
            batch = batch.to(device)  # (nj, p).
            vec = batch @ chunk  # (nj, ki).
            num += (vec ** 2.).sum(dim=0)
        nums.append(num)

    return (torch.cat(nums) / torch.cat(dens)).cpu()


def _check_error(
    loader: DataLoader,
    eigenvectors: torch.Tensor,
    device: Optional[torch.device],
    chunk_size: int,
    disable_tqdm: bool,
) -> Tuple[float, float]:
    """Compare QQ^tA against A."""
    ref_abs = []
    err_abs = []
    for (batch,) in tqdm.tqdm(loader, desc="check error", disable=disable_tqdm):
        batch = batch.to(device)  # (ni, p).
        batch_rec = torch.zeros_like(batch)  # (ni, p).

        nsteps = int(math.ceil(eigenvectors.size(1) / chunk_size))
        for idx in range(nsteps):
            start_idx = int(idx * chunk_size)
            chunk = eigenvectors[:, start_idx: start_idx + chunk_size].to(device)
            batch_rec += torch.mm(chunk, torch.mm(chunk.T, batch.T)).T

        err_abs.append((batch - batch_rec).norm(2))
        ref_abs.append(batch.norm(2))

    ref_abs = torch.stack(ref_abs).norm(2)
    err_abs = torch.stack(err_abs).norm(2)
    err_rel = err_abs / ref_abs

    return err_abs.item(), err_rel.item()


def _orthogonalize(matrix, device, disable_tqdm: bool, chunk_size: int = 100):
    # TODO: Don't put whole matrix on GPU to save memory.
    matrix = matrix.to(device)
    for i in tqdm.tqdm(range(matrix.size(1)), desc="orthogonalize", disable=disable_tqdm):
        # Normalize the ith column.
        col = matrix[:, i: i + 1]  # (p, 1).
        col /= col.norm(2)
        # Remove contribution of this component for remaining columns.
        if i + 1 < matrix.size(1):
            rest = matrix[:, i + 1:]  # (p, r).
            start_idx = 0
            while start_idx < rest.size(1):
                batch = rest[:, start_idx:start_idx + chunk_size]
                # Broadcast, point-wise multiply, and then reduce seems to
                #   suffer from less imprecision than direct matmul or mm.
                batch -= torch.sum(col * batch, dim=0) * col
                start_idx += chunk_size
    return matrix.cpu()


def get_bases(
    loader: DataLoader,
    k: int,
    num_power_iteration=1,
    disable_tqdm=False,
    stop_ratio=.9999,
    dtype=torch.float,
    device=None,
    dump_dir=None,
    chunk_size=100,
):
    """Simultaneous iteration for finding eigenvectors with the largest eigenvalues in absolute value.

    The method is aka subspace iteration or orthogonal iteration.

    Args:
        loader: Dataloader to incrementally load in flat gradients.
        k: Number of principal components to return.
        num_power_iteration: Number of power iterations.
        disable_tqdm: If True, disable progress bar.
        verbose: If True, log the error of QR.
        device: torch.device; defaults to CPU if None.
        dump_dir: Directory to dump the sequence of results.
        stop_ratio: Stop power iteration and return early when
            err_abs > stop_ratio * prev_err_abs.
        dtype: Precision in string format.
        chunk_size: Size of chunks for processing the dimension that loops over eigenvectors.

    Returns:
        eigenvectors: Tensor of selected basis of size (p, k).
        eigenvalues: Tensor of eigenvalues of data.T @ data of size (k,).
    """
    n = sum(batch.size(0) for batch, in loader)
    batch, = next(iter(loader))
    p = batch.size(1)
    k = min(k, p, n)

    prev_err_abs = sys.maxsize
    eigenvectors = torch.randn(size=(p, k), dtype=dtype)

    for global_step in tqdm.tqdm(range(num_power_iteration), desc="power iteration", disable=disable_tqdm):
        matrix = _mem_saving_matmul(
            loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
            device=device, disable_tqdm=disable_tqdm
        )
        eigenvectors = _orthogonalize(
            matrix=matrix,
            device=device, disable_tqdm=disable_tqdm
        )  # (p, k).
        eigenvalues = _eigenvectors_to_eigenvalues(
            loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
            device=device, disable_tqdm=disable_tqdm
        )
        err_abs, err_rel = _check_error(
            loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
            device=device, disable_tqdm=disable_tqdm,
        )
        logging.warning(f"global_step: {global_step}, abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

        if dump_dir is not None:
            utils.tsave(
                dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors, err_abs=err_abs, err_rel=err_rel),
                utils.join(dump_dir, "all", f"global_step_{global_step:06d}.pt")
            )
            utils.tsave(
                dict(eigenvalues=eigenvalues),
                utils.join(dump_dir, "eigenvalues", f"global_step_{global_step:06d}.evals")
            )

        if err_abs > stop_ratio * prev_err_abs:
            logging.warning("Reached breaking condition...")
            break
        prev_err_abs = err_abs

    return eigenvectors, eigenvalues  # noqa


# TODO: Write an eigenvalue test.
def main(task="pca", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("pca",),
        task_callables=(pca,),
        **kwargs,
    )


if __name__ == "__main__":
    # python -m classification.numerical --task "pca" --n 100 --k 100
    fire.Fire(main)
