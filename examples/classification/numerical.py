"""
"""

import fire
from swissknife import utils
import torch
import tqdm


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


def orthogonalize_(matrix):
    """Gram-Schmidt."""
    n, m = matrix.shape
    for i in range(m):
        # Normalize the ith column.
        col = matrix[:, i: i + 1]
        col /= col.norm(2)
        # Project it on the rest and remove it.
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col


def check_approx_error(L, target):
    """Compute the relative squared error."""
    encode = torch.matmul(target, L)  # n x k
    decode = torch.matmul(encode, L.t())
    error = torch.sum(torch.square(target - decode))
    target = torch.sum(torch.square(target))
    if target.item() == 0:
        return -1
    return error.item() / target.item()


def get_bases(data, k, power_iter=1):
    """QR algorithm for finding top-k eigenvalues.

    Args:
        data: Tensor of size (n, p).
        k: Number of principal components to return.

    Returns:
        L: Tensor of selected basis of size (k, k).
        error_rate: Tensor of size (1,) for relative tolerance.
    """
    n, p = data.size()
    k = min(k, p)
    L = torch.randn(size=(p, k), device=data.device)
    for i in range(power_iter):
        # TODO: Make this more mem efficient.
        R = torch.matmul(data, L)  # np, pk -> nk
        L = torch.matmul(data.t(), R)  # pn, nk -> pk
        orthogonalize_(L)  # pk; orthonormalize the columns.
    error_rate = check_approx_error(L, data)
    return L, error_rate


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
