"""
Numerical algorithms.

Currently only contains QR.

TODO: Move this to swissknife.
"""
import logging

import fire
from swissknife import utils
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dir_, num_ckpts, varname):
    all_ckpts = utils.all_ckpts(dir_, sort=True)[:num_ckpts]
    return torch.stack(
        [
            torch.load(ckpt_path)[varname]
            for ckpt_path in tqdm.tqdm(all_ckpts, desc="load data")
        ]
    )


def qr(grads_dir="/mnt/disks/disk-2/dump/classification/test/grad_trajectory", num_ckpts=1000, varname="flat_grad"):
    data = load_data(dir_=grads_dir, num_ckpts=num_ckpts, varname=varname)
    data = data.to(device)
    Q = get_bases(data=data, k=500, num_power_iteration=2)


def get_bases(data: torch.Tensor, k: int, num_power_iteration=1, save_mem=True, enable_tqdm=True, verbose=True):
    """QR algorithm for finding top-k eigenvectors.

    Args:
        data: Tensor of size (n, p).
        k: Number of principal components to return.
        num_power_iteration: Number of power iterations.
        save_mem: If True, perform matmul in a for loop to save memory.
        enable_tqdm: If True, report progress bar for power iteration.
        verbose: If True, log the error of QR.

    Returns:
        Q: Tensor of selected basis of size (p, k).
        error_rate: Tensor of size (1,) for relative tolerance.
    """
    n, p = data.size()
    k = min(k, p)
    Q = torch.randn(size=(p, k), device=data.device)
    for _ in tqdm.tqdm(range(num_power_iteration), desc="power iter", disable=not enable_tqdm):
        if save_mem:
            R = torch.stack([data @ Q[:, col_idx] for col_idx in range(k)], dim=1)
            Q = torch.stack([data.t() @ R[:, col_idx] for col_idx in range(k)], dim=1)
        else:
            R = torch.matmul(data, Q)  # np, pk -> nk.
            Q = torch.matmul(data.t(), R)  # pn, nk -> pk.
        _orthogonalize(Q)  # pk; orthonormalize the columns.

    if verbose:
        err_abs, err_rel = _check_qr_error(data=data, Q=Q, save_mem=save_mem)
        logging.warning(f"abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

    return Q


def _orthogonalize(matrix):
    """Gram-Schmidt."""
    n, m = matrix.size()
    for i in range(m):
        # Normalize the ith column.
        col = matrix[:, i: i + 1]
        col /= col.norm(2)
        # Remove contribution of this component for remaining columns.
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col


def _check_qr_error(data: torch.Tensor, Q: torch.Tensor, save_mem: bool):
    n, p = data.size()
    _, k = Q.size()
    if save_mem:
        recon = torch.stack([data @ Q[:, col_idx] for col_idx in range(k)], dim=1)  # nk.
        recon = torch.stack([recon @ Q[col_idx, :] for col_idx in range(p)], dim=1)  # np.
    else:
        recon = data @ Q @ Q.T  # np.

    err_abs = (data - recon).norm(2)
    err_rel = err_abs / data.norm(2)
    return err_abs, err_rel


def main(task="qr", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("qr",),
        task_callables=(qr,),
        **kwargs,
    )


if __name__ == "__main__":
    # python -m classification.numerical --task "qr"
    fire.Fire(main)
