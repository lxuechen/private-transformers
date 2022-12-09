# Copyright (c) Xuechen Li. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,  software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,  either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiments ran pre- and post-rebuttals."""
import logging
import os
from typing import Optional

import fire
import torch
import tqdm
from ml_swissknife import utils, numerical_distributed
from torch.utils.data import DataLoader, TensorDataset


def run_save_grads(
    num_train_epochs=60,  # This amounts to 4k updates, roughly.
    model_name_or_path="roberta-base",
    train_dir=None,
    per_device_train_batch_size=25,
):
    if train_dir is None:
        train_dir = utils.join("/mnt/data1/dump/", 'rebuttal_v2', f'run-{model_name_or_path}')
    command = f'''python -m classification.run_wrapper \
        --output_dir {train_dir} \
        --task_name "sst-2" \
        --model_name_or_path "{model_name_or_path}" \
        --attention_only "yes" \
        --static_lm_head "yes" \
        --num_train_epochs {num_train_epochs} \
        --eval_spectrum "no" \
        --non_private "no" \
        --eval_steps 50 \
        --randomly_initialize "no" \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --batch_size 1000 \
        --clipping_mode "default" \
        --store_grads "yes"'''
    os.system(command)


def run_pca(
    # Place where grads are stored and where results will be stored.
    train_dir="/mnt/disks/disk-2/dump/privlm/roberta/sst-2",
    n=2000,  # How many checkpoints?
    k=1000,  # How many eigenvectors?
    num_power_iteration=10,
    batch_size=20,  # Batch size for processing the checkpoints in matmul.
    seed=42,  # Controls randomness in sampling the first vector in orthogonal iteration.
    start_index=0,  # The index of the first checkpoint to be selected.
    eval_steps=5,  # Evaluate PCA accuracy once this many iterations.
    save_steps=5,  # Save eigenvalue and eigenvector tensors once this many iterations.
    disable_tqdm=False,
    dtype="float",  # String repr of dtype.
):
    utils.manual_seed(seed)

    ckpt_dir = utils.join(train_dir, 'grad_trajectory')
    dump_dir = utils.join(train_dir, 'orthproj')

    all_ckpts = utils.all_ckpts(ckpt_dir, sort=True)
    tgt_ckpts = all_ckpts[start_index:start_index + n]
    dataset = torch.stack([
        torch.load(ckpt_path)["flat_grad"] for ckpt_path in tqdm.tqdm(tgt_ckpts, desc="load data")
    ]).to(utils.get_dtype(dtype))
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
        callback=callback,
        disable_tqdm=disable_tqdm,
    )


def run_retrain_single(
    output_dir: str,
    orthogonal_projection_path: str,
    model_name_or_path: str,
    rank: Optional[int] = None,
    seed=42,
):
    cmd = f'''python -m classification.run_wrapper \
        --output_dir {output_dir} \
        --task_name "sst-2" \
        --model_name_or_path {model_name_or_path} \
        --few_shot_type "prompt" \
        --attention_only "yes" \
        --static_lm_head "yes" \
        --per_device_train_batch_size 25 \
        --batch_size 1000 \
        --clipping_mode "default" \
        --num_train_epochs 4 \
        --eval_spectrum "no" \
        --non_private "no" \
        --eval_steps 25 \
        --randomly_initialize "no" \
        --seed {seed} \
        --orthogonal_projection_path {orthogonal_projection_path}'''
    if rank is not None:
        cmd += f' --orthogonal_projection_rank {rank}'
    os.system(cmd)


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
