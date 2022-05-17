"""
Numerical algorithms.

Currently, only contains simultaneous iter.
"""
import gc
import logging
from typing import Optional

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


def qr(
    grads_dir="/mnt/disks/disk-2/dump/classification/test/grad_trajectory",
    dump_dir="/mnt/disks/disk-2/dump/classification/test/orthproj",
    num_ckpts=1000,
    varname="flat_grad",
    num_power_iteration=1,
    k=1000,
):
    data = load_data(dir_=grads_dir, num_ckpts=num_ckpts, varname=varname)
    Q = get_bases(data=data, k=k, num_power_iteration=num_power_iteration, gpu=device, dump_dir=dump_dir)


def _mem_saving_matmul(mat1, mat2, gpu):
    (n, k), (_, m) = mat1.size(), mat2.size()
    mat1_a, mat1_b = torch.split(mat1, split_size_or_sections=2, dim=1)

    out = []
    for mat in (mat1_a, mat1_b):
        mat = mat.to(gpu)
        section_out = []
        for idx in range(m):
            section_out.append(
                (mat @ mat2[:, idx].to(gpu)).cpu()
            )
        out.append(
            torch.stack(section_out, dim=1)
        )

        del mat
        gc.collect()
        torch.cuda.empty_cache()

    return torch.cat(out, dim=0)


def get_bases(data: torch.Tensor, k: int, num_power_iteration=1, save_mem=True, disable_tqdm=False, verbose=True,
              gpu=None, dump_dir=None):
    """Simultaneous iteration for finding eigenvectors with the largest eigenvalues in absolute value.

    The method is aka subspace iteration or orthogonal iteration.

    Args:
        data: Tensor of size (n, p).
        k: Number of principal components to return.
        num_power_iteration: Number of power iterations.
        save_mem: If True, perform matmul in a for loop to save memory.
        disable_tqdm: If True, disable progress bar.
        verbose: If True, log the error of QR.
        gpu: torch.device; defaults to CPU if None.
        dump_dir: Directory to dump the sequence of results.

    Returns:
        eigenvectors: Tensor of selected basis of size (p, k).
        eigenvalues: Tensor of eigenvalues of data.T @ data of size (k,).
    """

    def _rayleigh_quotient(mat: torch.Tensor, vec: torch.Tensor):
        """Compute v^t A^t A v / v^t v."""
        mvp = torch.matmul(mat, vec)
        return (mvp * mvp).sum() / (vec * vec).sum()

    n, p = data.size()
    k = min(k, p, n)
    Q = torch.randn(size=(p, k))

    for global_step in tqdm.tqdm(range(num_power_iteration), desc="power iter", disable=disable_tqdm):
        if save_mem:
            R = _mem_saving_matmul(mat1=data, mat2=Q, gpu=gpu)
            Q = _mem_saving_matmul(mat1=data.T, mat2=R, gpu=gpu)
        else:
            R = torch.matmul(data, Q)  # np, pk -> nk.
            Q = torch.matmul(data.T, R)  # pn, nk -> pk.
        Q = _orthogonalize(matrix=Q, gpu=gpu, disable_tqdm=disable_tqdm)  # pk; orthonormalize the columns.

        eigenvectors = Q
        eigenvalues = torch.stack(
            [_rayleigh_quotient(mat=data.to(gpu), vec=q.to(gpu)) for q in eigenvectors.T],
        ).cpu()

        # TODO: stop when the reconstruction error is stable.
        err_abs, err_rel = _check_qr_error(data=data, Q=Q, save_mem=save_mem, disable_tqdm=disable_tqdm, gpu=gpu)

        if dump_dir is not None:
            utils.makedirs(dump_dir, exist_ok=True)
            dump_path = utils.join(dump_dir, f"global_step_{global_step:06d}.pt")
            torch.save(
                dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors, err_abs=err_abs, err_rel=err_rel),
                dump_path,
            )

            logging.warning(f"abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

    return eigenvectors, eigenvalues  # noqa


def _orthogonalize(matrix, gpu, disable_tqdm: bool):
    """Gram-Schmidt.

    By far the slowest step, since cannot be parallelized.
    """
    n, m = matrix.size()
    matrix = matrix.to(gpu)
    for i in tqdm.tqdm(range(m), desc="orthogonalize", disable=disable_tqdm):
        # Normalize the ith column.
        col = matrix[:, i: i + 1]
        col /= col.norm(2)
        # Remove contribution of this component for remaining columns.
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            rest -= torch.sum(col * rest, dim=0) * col
    matrix = matrix.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return matrix


def _check_qr_error(data: torch.Tensor, Q: torch.Tensor, save_mem: bool, disable_tqdm: bool,
                    gpu: Optional[torch.device]):
    n, p = data.size()
    _, k = Q.size()

    if save_mem:
        data = data.to(gpu, non_blocking=True)
        data_norm = data.norm(2).item()

        iterator = tqdm.tqdm(range(k), desc="check qr:: encode", disable=disable_tqdm)
        data_mul_Q = torch.cat(
            [data.matmul(Q[:, idx][:, None].to(gpu, non_blocking=True)) for idx in iterator],
            dim=1
        )  # nk.

        data = data.cpu()
        data_mul_Q = data_mul_Q.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        Q = Q.to(gpu, non_blocking=True)

        iterator = tqdm.tqdm(range(n), desc="check qr:: decode", disable=disable_tqdm)

        err_abs = torch.sqrt(
            sum(
                [
                    (Q.matmul(data_mul_Q.T[:, idx:idx + 1].to(gpu, non_blocking=True)).squeeze() -
                     data[idx].to(gpu, non_blocking=True)).norm(2) ** 2.
                    for idx in iterator
                ]
            )
        )
        err_rel = err_abs / data_norm

    else:
        recon = data @ Q @ Q.T  # np.

        err_abs = (data - recon).norm(2)
        err_rel = err_abs / data.norm(2)

    return err_abs.item(), err_rel.item()


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
