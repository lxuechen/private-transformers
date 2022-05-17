"""
Numerical algorithms.

Currently, only contains simultaneous iter.
"""
import gc
import logging
import sys
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
    grads_dir="/mnt/disks/disk-2/dump/privlm/roberta/sst-2/grad_trajectory",
    dump_dir="/mnt/disks/disk-2/dump/privlm/roberta/sst-2/orthproj",
    num_ckpts=2000,
    varname="flat_grad",
    num_power_iteration=100,
    k=2000,
):
    data = load_data(dir_=grads_dir, num_ckpts=num_ckpts, varname=varname)
    get_bases(data=data, k=k, num_power_iteration=num_power_iteration, gpu=device, dump_dir=dump_dir)


def _mem_saving_matmul(mat1, mat2, gpu, disable_tqdm: bool, tag=None, chunks1=2, chunks2=100):
    # (n x p) x (p x k); most intensive dimension should be p.
    out = []

    mat1_seq = torch.chunk(mat1, chunks=chunks1, dim=0)
    outer_tag = "outer" if tag is None else f"outer ({tag})"
    outer_iterator = tqdm.tqdm(mat1_seq, desc=outer_tag, disable=disable_tqdm)
    for this_mat1 in outer_iterator:
        this_mat1 = this_mat1.to(gpu)
        section_out = []

        mat2_seq = torch.chunk(mat2, chunks=chunks2, dim=1)
        inner_tag = "inner" if tag is None else f"inner ({tag})"
        inner_iterator = tqdm.tqdm(mat2_seq, desc=inner_tag, disable=disable_tqdm)
        for this_mat2 in inner_iterator:
            section_out.append(
                torch.mm(this_mat1, this_mat2.to(gpu)).cpu()
            )
        out.append(
            torch.cat(section_out, dim=1)
        )

        del this_mat1, this_mat2
        gc.collect()
        torch.cuda.empty_cache()

    out = torch.cat(out, dim=0)
    return out


def get_bases(data: torch.Tensor, k: int, num_power_iteration=1, save_mem=True, disable_tqdm=False, verbose=True,
              gpu=None, dump_dir=None, stop_ratio=.999, dtype="float32"):
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
        stop_ratio: Stop power iteration and return early when
            err_abs > stop_ratio * prev_err_abs.
        dtype: Precision in string format.

    Returns:
        eigenvectors: Tensor of selected basis of size (p, k).
        eigenvalues: Tensor of eigenvalues of data.T @ data of size (k,).
    """

    def _rayleigh_quotient(mat: torch.Tensor, vec: torch.Tensor):
        """Compute v^t A^t A v / v^t v."""
        mvp = torch.matmul(mat, vec)
        return (mvp * mvp).sum() / (vec * vec).sum()

    dtype = utils.get_dtype(dtype)
    data = data.to(dtype)
    n, p = data.size()
    k = min(k, p, n)
    Q = torch.randn(size=(p, k), dtype=dtype)
    prev_err_abs = sys.maxsize

    for global_step in tqdm.tqdm(range(num_power_iteration), desc="power iter", disable=disable_tqdm):
        if save_mem:
            R = _mem_saving_matmul(mat1=data, mat2=Q, gpu=gpu, disable_tqdm=disable_tqdm, tag="R")
            Q = _mem_saving_matmul(mat1=data.T, mat2=R, gpu=gpu, disable_tqdm=disable_tqdm, tag="Q")
        else:
            R = torch.matmul(data, Q)  # np, pk -> nk.
            Q = torch.matmul(data.T, R)  # pn, nk -> pk.
        Q = _msg(matrix=Q, gpu=gpu, disable_tqdm=disable_tqdm)  # pk; orthonormalize the columns.

        data = data.to(gpu)
        eigenvectors = Q
        eigenvalues = torch.stack(
            [_rayleigh_quotient(mat=data, vec=q.to(gpu))
             for q in tqdm.tqdm(eigenvectors.T, desc="eigenvalues")],
        ).cpu()
        data = data.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        err_abs, err_rel = _check_qr_error(data=data, Q=Q, save_mem=save_mem, disable_tqdm=disable_tqdm, gpu=gpu)
        logging.warning(f"abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

        if dump_dir is not None:
            utils.makedirs(dump_dir, exist_ok=True)
            dump_path = utils.join(dump_dir, f"global_step_{global_step:06d}.pt")
            torch.save(
                dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors, err_abs=err_abs, err_rel=err_rel),
                dump_path,
            )

        if err_abs > stop_ratio * prev_err_abs:
            logging.warning("Reached breaking condition...")
            break
        prev_err_abs = err_abs

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
            rest = matrix[:, i + 1:]  # (p, r).
            rest -= torch.mm(rest, torch.mm(rest.T, col))
    matrix = matrix.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return matrix


def _msg(matrix, gpu, disable_tqdm: bool):
    """Modified Gram-Schmidt.

    Much more numerically stable.
    """
    n, m = matrix.size()
    matrix = matrix.to(gpu)

    for i in tqdm.tqdm(range(m), desc="orthogonalize", disable=disable_tqdm):
        qi = matrix[:, i: i + 1]  # (p, 1).
        if i > 0:
            prev = matrix[:, :i]  # (p, r).
            qi -= torch.mm(prev, torch.mm(prev.T, qi))
        qi /= qi.norm(2)

    matrix = matrix.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return matrix


def _check_qr_error(data: torch.Tensor, Q: torch.Tensor, save_mem: bool, disable_tqdm: bool,
                    gpu: Optional[torch.device], chunks=2):
    n, p = data.size()
    _, k = Q.size()

    if save_mem:
        data_mul_Q = _mem_saving_matmul(
            mat1=data, mat2=Q, gpu=gpu, disable_tqdm=disable_tqdm, tag="check qr:: encode"
        )  # (n, k).

        new_data = _mem_saving_matmul(
            mat1=Q, mat2=data_mul_Q.T, gpu=gpu, disable_tqdm=disable_tqdm, tag="check qr:: decode"
        )

        a_chunks = torch.chunk(data, chunks=chunks, dim=0)
        b_chunks = torch.chunk(new_data, chunks=chunks, dim=0)
        err_abs = []
        for ai, bi in tqdm.tqdm(utils.zip_(a_chunks, b_chunks), desc="check qr error chunks", total=chunks):
            ai, bi = ai.to(gpu), bi.to(gpu)
            err_abs.append(
                (ai - bi).norm(2) ** 2
            )
        err_abs = torch.stack(err_abs).sum().sqrt().item()
        err_rel = err_abs / data.to(gpu).norm(2).item()
    else:
        recon = data @ Q @ Q.T  # np.

        err_abs = (data - recon).norm(2)
        err_rel = err_abs / data.norm(2)

    return err_abs.item(), err_rel.item()


def test_qr_decomposition(p=100000, k=100):
    torch.set_default_dtype(torch.float16)
    Q = torch.randn(p, k, device=device) * 3
    P1 = _orthogonalize(Q, gpu=device, disable_tqdm=True)
    P2 = _msg(Q, gpu=device, disable_tqdm=True)
    torch.testing.assert_allclose(P1, P2)
    print('.')


def main(task="qr", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("qr", "test_qr_decomposition"),
        task_callables=(qr, test_qr_decomposition),
        **kwargs,
    )


if __name__ == "__main__":
    # python -m classification.numerical --task "qr"
    # CUDA_VISIBLE_DEVICES=3 python -m classification.numerical --task "test_qr_decomposition"
    fire.Fire(main)
