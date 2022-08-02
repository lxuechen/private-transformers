"""
Numerical algorithms.

Currently, only contains simultaneous iter.
"""
import logging
import math
from typing import Optional, Tuple

import fire
import numpy as np
import torch
import tqdm
from ml_swissknife import utils
from torch.utils.data import DataLoader, TensorDataset


def load_data(ckpts_dir, num_ckpts, start_index, batch_size):
    all_ckpts = utils.all_ckpts(ckpts_dir, sort=True)[start_index:start_index + num_ckpts]
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
    train_dir="/mnt/disks/disk-2/dump/privlm/roberta/sst-2",
    n=2000,
    k=1000,
    num_power_iteration=10,
    batch_size=200,
    seed=42,
    start_index=0,
    chunk_size=100,
    chunk_size_2=10,
):
    utils.manual_seed(seed)
    grads_dir = utils.join(train_dir, 'grad_trajectory')
    dump_dir = utils.join(train_dir, 'orthproj')

    orthogonal_iteration(
        loader=load_data(ckpts_dir=grads_dir, num_ckpts=n, start_index=start_index, batch_size=batch_size),
        k=k,
        num_power_iteration=num_power_iteration,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dump_dir=dump_dir,
        chunk_size=chunk_size,
        chunk_size_2=chunk_size_2,
        dtype=torch.get_default_dtype(),
    )


def _mem_saving_matmul(
    loader: DataLoader,
    eigenvectors: torch.Tensor,
    chunk_size: int,
    device: Optional[torch.device],
    disable_tqdm: bool,
):
    """Compute AQ."""
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


# def _mem_saving_matmul_v2(
#     loader: DataLoader,
#     eigenvectors: torch.Tensor,
#     disable_tqdm: bool,
#     **kwargs,
# ):
#     num_cuda_devices = torch.cuda.device_count()
#     assert num_cuda_devices > 0, "v2 is only supported in distributed settings."
#     devices = tuple(range(num_cuda_devices))
#
#     out = torch.zeros_like(eigenvectors)
#
#     evec_chunks = torch.tensor_split(eigenvectors, len(devices), dim=1)
#     evec_chunks = tuple(evec_chunk.to(device) for evec_chunk, device in utils.zip_(evec_chunks, devices))
#
#     chunk_num_cols = (0,) + tuple(evec_chunk.size(1) for evec_chunk in evec_chunks)
#     chunk_num_cols_cumsum = np.cumsum(chunk_num_cols)
#     chunk_col_ranges = tuple(utils.zip_(chunk_num_cols_cumsum[:-1], chunk_num_cols_cumsum[1:]))
#
#     for chunk, chunk_col_range in utils.zip_(evec_chunks, chunk_col_ranges):
#         this_out = torch.zeros(size=chunk.size())  # GPU. (p, k1).
#         for (batch,) in tqdm.tqdm(loader, desc="batches", disable=disable_tqdm):
#             batch = batch.to(chunk.device, non_blocking=True)
#             this_out += torch.mm(batch.T, torch.mm(batch, chunk)).cpu()
#         out[:, chunk_col_range[0]:chunk_col_range[1]] = this_out.cpu()
#     return out

def _mem_saving_matmul_v2(
    loader: DataLoader,
    eigenvectors: torch.Tensor,
    disable_tqdm: bool,
    **kwargs,
):
    num_cuda_devices = torch.cuda.device_count()
    assert num_cuda_devices > 0, "v2 is only supported in distributed settings."
    devices = tuple(range(num_cuda_devices))

    out = torch.zeros_like(eigenvectors)

    evec_chunks = torch.tensor_split(eigenvectors, len(devices), dim=1)
    evec_chunks = tuple(evec_chunk.to(device) for evec_chunk, device in utils.zip_(evec_chunks, devices))

    chunk_num_cols = (0,) + tuple(evec_chunk.size(1) for evec_chunk in evec_chunks)
    chunk_num_cols_cumsum = np.cumsum(chunk_num_cols)
    chunk_col_ranges = tuple(utils.zip_(chunk_num_cols_cumsum[:-1], chunk_num_cols_cumsum[1:]))

    for (batch,) in tqdm.tqdm(loader, desc="batches", disable=disable_tqdm):
        outs = []
        for chunk in evec_chunks:
            batch = batch.to(chunk.device, non_blocking=True)
            outs.append(torch.mm(batch.T, torch.mm(batch, chunk)))

        outs = [o.cpu() for o in outs]
        for o, chunk_col_range in utils.zip_(outs, chunk_col_ranges):
            out[:, chunk_col_range[0]:chunk_col_range[1]] += o

    return out


def _eigenvectors_to_eigenvalues(
    loader: DataLoader, eigenvectors: torch.Tensor,
    chunk_size: int,  # Number of eigenvalues to process at once.
    device: Optional[torch.device],
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


def _check_error_v2(
    loader: DataLoader,
    eigenvectors: torch.Tensor,
    disable_tqdm: bool,
    **kwargs,
):
    num_cuda_devices = torch.cuda.device_count()
    assert num_cuda_devices > 0, "v2 is only supported in distributed settings."
    devices = tuple(range(num_cuda_devices))

    evec_chunks = torch.tensor_split(eigenvectors, len(devices), dim=1)
    evec_chunks = tuple(evec_chunk.to(device) for evec_chunk, device in utils.zip_(evec_chunks, devices))

    ref_abs = []
    err_abs = []
    for (batch,) in tqdm.tqdm(loader, desc="check error", disable=disable_tqdm):
        batch_recs = []
        for evec_chunk in evec_chunks:
            this_batch = batch.to(evec_chunk.device, non_blocking=True)
            batch_recs.append(
                torch.mm(evec_chunk, torch.mm(evec_chunk.T, this_batch.T)).T
            )

        batch_rec = batch_recs[0]
        for this_batch_rec in batch_recs[1:]:
            batch_rec += this_batch_rec.to(0)
        batch = batch.to(0)

        err_abs.append((batch - batch_rec).norm(2))
        ref_abs.append(batch.norm(2))

    ref_abs = torch.stack(ref_abs).norm(2)
    err_abs = torch.stack(err_abs).norm(2)
    err_rel = err_abs / ref_abs

    return err_abs.item(), err_rel.item()


def _orthogonalize(matrix, device, disable_tqdm: bool, chunk_size_2=10):
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
                batch = rest[:, start_idx:start_idx + chunk_size_2]
                # Broadcast, point-wise multiply, and then reduce seems to
                #   suffer from less imprecision than direct matmul or mm.
                batch -= torch.sum(col * batch, dim=0) * col
                start_idx += chunk_size_2
    return matrix.cpu()


def _orthogonalize_v2(matrix, device, disable_tqdm: bool, chunk_size_2=10):
    # Memory saving: don't store `matrix` on device.
    # Note this is an in-place operation!
    for i in tqdm.tqdm(range(matrix.size(1)), desc="orthogonalize", disable=disable_tqdm):
        # Normalize the ith column.
        col = matrix[:, i: i + 1].to(device)  # (p, 1). copy!
        col /= col.norm(2)
        matrix[:, i: i + 1] = col.cpu()

        # Remove contribution of this component for remaining columns.
        if i + 1 < matrix.size(1):
            rest = matrix[:, i + 1:]  # (p, r).
            start_idx = 0
            while start_idx < rest.size(1):
                batch = rest[:, start_idx:start_idx + chunk_size_2].to(device)  # copy!
                # Broadcast, point-wise multiply, and then reduce seems to
                #   suffer from less imprecision than direct matmul or mm.
                batch -= torch.sum(col * batch, dim=0) * col
                rest[:, start_idx:start_idx + chunk_size_2] = batch.cpu()
                start_idx += chunk_size_2
    return matrix


def _orthogonalize_v3(matrix, device, disable_tqdm: bool, chunk_size_2=20):
    if device.type == "cuda":
        devices = tuple(range(torch.cuda.device_count()))
    else:
        devices = (device,)

    matrix_chunks = torch.tensor_split(matrix, len(devices), dim=1)
    matrix_chunks = tuple(
        matrix_chunk.to(matrix_device) for matrix_chunk, matrix_device in utils.zip_(matrix_chunks, devices)
    )
    chunk_num_cols = (0,) + tuple(matrix_chunk.size(1) for matrix_chunk in matrix_chunks)
    chunk_num_cols_cumsum = np.cumsum(chunk_num_cols)
    chunk_col_ranges = tuple(utils.zip_(chunk_num_cols_cumsum[:-1], chunk_num_cols_cumsum[1:]))

    def col_idx_to_chunk_idx_and_offset(col_idx):
        """Returns the index of the matrix chunk and the offset to index into."""
        # k, offset = col_idx_to_chunk_idx_and_offset(col_idx)
        # col = matrix_chunks[k][offset]
        for k, (head, tail) in enumerate(chunk_col_ranges):
            if head <= col_idx < tail:
                offset = col_idx - head
                return k, offset
        raise ValueError("Should not reach here...")

    def gram_schmidt_helper(col_, rest_):
        col_ = col_.to(rest_)
        start_idx = 0
        while start_idx < rest_.size(1):
            batch = rest_[:, start_idx:start_idx + chunk_size_2]
            batch -= torch.sum(col_ * batch, dim=0) * col_
            start_idx += chunk_size_2

    for i in tqdm.tqdm(range(matrix.size(1)), desc="orthogonalize", disable=disable_tqdm):
        k, offset = col_idx_to_chunk_idx_and_offset(i)
        matrix_chunk = matrix_chunks[k]

        col = matrix_chunk[:, offset:offset + 1]
        col /= col.norm(2)
        if i + 1 < matrix.size(1):
            # current matrix_chunk.
            rest = matrix_chunk[:, offset + 1:]
            gram_schmidt_helper(col, rest)

            # future matrix_chunk.
            for future_matrix_chunk in matrix_chunks[k + 1:]:
                rest = future_matrix_chunk
                gram_schmidt_helper(col, rest)
    return torch.cat(tuple(matrix_chunk.cpu() for matrix_chunk in matrix_chunks), dim=1)


def orthogonal_iteration(
    loader: DataLoader,
    k: int,
    num_power_iteration=1,
    disable_tqdm=False,
    dtype=torch.float,
    device: Optional[torch.device] = None,
    dump_dir=None,
    chunk_size=100,
    chunk_size_2=10,
    eval_steps=5,
):
    """Simultaneous iteration for finding eigenvectors with the largest eigenvalues in absolute value.

    The method is aka subspace iteration or orthogonal iteration.

    WARNING:
        - good reconstruction of the data does not imply converged eigenvalues!

    Args:
        loader: Dataloader to incrementally load in flat gradients.
        k: Number of principal components to return.
        num_power_iteration: Number of power iterations.
        disable_tqdm: If True, disable progress bar.
        device: torch.device; defaults to CPU if None.
        dump_dir: Directory to dump the sequence of results.
        dtype: Precision in string format.
        chunk_size: Size of chunks for processing the dimension that loops over eigenvectors.
        chunk_size_2: Size of chunks for orthogonalization.
        eval_steps: Number of steps before a data reconstruction evaluation.
        use_v2: Use memory saving version of orthogonalization. Very slow due to tensor transfer across devices.

    Returns:
        eigenvectors: Tensor of selected basis of size (p, k).
        eigenvalues: Tensor of eigenvalues of data.T @ data of size (k,).
    """
    n = sum(batch.size(0) for batch, in loader)
    batch, = next(iter(loader))
    p = batch.size(1)
    k = min(k, p, n)
    eigenvectors = torch.randn(size=(p, k), dtype=dtype)  # This step will be very slow for large models.
    orthogonalizer = _orthogonalize_v3

    # err_abs, err_rel = _check_error_v2(
    #     loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
    #     device=device, disable_tqdm=disable_tqdm,
    # )
    # logging.warning(f"before iteration, abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

    for global_step in tqdm.tqdm(range(1, num_power_iteration + 1), desc="power iteration", disable=disable_tqdm):
        matrix = _mem_saving_matmul(
            loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
            device=device, disable_tqdm=disable_tqdm
        )
        # matrix = torch.zeros_like(eigenvectors)
        # eigenvectors = orthogonalizer(
        #     matrix=matrix, chunk_size_2=chunk_size_2,
        #     device=device, disable_tqdm=disable_tqdm
        # )  # (p, k).
        eigenvalues = _eigenvectors_to_eigenvalues(
            loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
            device=device, disable_tqdm=disable_tqdm
        )

        if dump_dir is not None:
            utils.tsave(
                dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors),
                utils.join(dump_dir, "all", f"global_step_{global_step:06d}.pt")
            )
            utils.tsave(
                dict(eigenvalues=eigenvalues),
                utils.join(dump_dir, "eigenvalues", f"global_step_{global_step:06d}.evals")
            )

        if global_step % eval_steps == 0:
            err_abs, err_rel = _check_error(
                loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
                device=device, disable_tqdm=disable_tqdm,
            )
            logging.warning(f"global_step: {global_step}, abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

    return eigenvalues, eigenvectors  # noqa


def test_orthogonal_iteration(n=100, d=20, k=10):
    torch.set_default_dtype(torch.float64)

    features = torch.randn(n, d)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)

    eigenvalues, eigenvectors = orthogonal_iteration(
        loader=loader,
        k=k,
        num_power_iteration=100,
        device=torch.device("cuda" if torch.cuda.is_available() else 'cpu'),
        chunk_size=2,
        dtype=torch.get_default_dtype(),
    )
    eigenvalues_expected, eigenvectors_expected = torch.linalg.eigh(features.T @ features)
    print(eigenvalues)
    print(eigenvalues_expected.flip(dims=(0,)))
    print('---')
    print(eigenvectors)
    print(eigenvectors_expected.flip(dims=(1,)))
    torch.testing.assert_allclose(eigenvalues, eigenvalues_expected.flip(dims=(0,))[:k], atol=1e-4, rtol=1e-4)


def test_mem_saving_matmul(n=1000, d=100):
    torch.set_default_dtype(torch.float64)

    features = torch.randn(n, d)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False)
    Q = torch.randn(d, d)

    matmul = _mem_saving_matmul(
        loader, Q, chunk_size=10, device=None, disable_tqdm=True
    )
    matmul_expected = features.T @ features @ Q
    torch.testing.assert_allclose(matmul, matmul_expected)


def test_mem_saving_matmul_v2(n=1000, d=100):
    torch.set_default_dtype(torch.float64)

    features = torch.randn(n, d)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=min(100, 2), shuffle=False, drop_last=False)
    Q = torch.randn(d, d)

    matmul = _mem_saving_matmul_v2(loader, Q, disable_tqdm=True)
    matmul_expected = features.T @ features @ Q
    torch.testing.assert_allclose(matmul, matmul_expected)


def test__check_error_v2(n=1000, d=10, k=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = torch.randn(n, d)
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False)

    eigenvectors = torch.randn(d, k)
    err1 = _check_error(loader=loader, eigenvectors=eigenvectors, chunk_size=10, device=device, disable_tqdm=False)
    err2 = _check_error_v2(
        loader=loader, eigenvectors=eigenvectors, chunk_size=10, disable_tqdm=False
    )
    torch.testing.assert_close(err1, err2)


def test__orthogonalize_v2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p, k, chunk_size_2 = int(50 * 10 ** 6), 500, 10
    matrix = torch.randn(p, k)

    out2 = _orthogonalize_v2(matrix=matrix.clone(), device=device, disable_tqdm=False, chunk_size_2=chunk_size_2)
    out1 = _orthogonalize(matrix=matrix.clone(), device=device, disable_tqdm=False, chunk_size_2=chunk_size_2)

    torch.testing.assert_close(out1, out2)


def test__orthogonalize_v3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p, k, chunk_size_2 = 10000, 500, 10
    matrix = torch.randn(p, k)

    out2 = _orthogonalize_v3(matrix=matrix.clone(), device=device, disable_tqdm=False, chunk_size_2=chunk_size_2)
    out1 = _orthogonalize(matrix=matrix.clone(), device=device, disable_tqdm=False, chunk_size_2=chunk_size_2)

    torch.testing.assert_close(out1, out2)


def main(task="pca", **kwargs):
    utils.runs_tasks(
        task=task,
        task_names=("pca", "test_orthogonal_iteration", "test_mem_saving_matmul", "test__orthogonalize_v2",
                    "test__orthogonalize_v3", "test__check_error_v2", "test_mem_saving_matmul_v2"),
        task_callables=(pca, test_orthogonal_iteration, test_mem_saving_matmul, test__orthogonalize_v2,
                        test__orthogonalize_v3, test__check_error_v2, test_mem_saving_matmul_v2),
        **kwargs,
    )


if __name__ == "__main__":
    # python -m classification.numerical --task "pca" --n 100 --k 100
    # python -m classification.numerical --task "test_orthogonal_iteration"
    # python -m classification.numerical --task "test_mem_saving_matmul"
    # python -m classification.numerical --task "test__orthogonalize_v2"
    # python -m classification.numerical --task "test__orthogonalize_v3"
    # python -m classification.numerical --task "test__check_error_v2"
    # python -m classification.numerical --task "test_mem_saving_matmul_v2"
    fire.Fire(main)
