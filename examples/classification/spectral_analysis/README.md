## Experiments for spectral analysis

Everything below should be run from the `examples/` folder.

1. To run with the geometric median example in the paper, use the following command and supply an `<output_dir>` of your
   choice:

```bash
python -m classification.spectral_analysis.geometric_median --img_dir <output_dir>
```

2. Spectral analysis.

   2.1. To reproduce the spectral analysis experiments, one first need a first round of fine-tuning to collect
   gradients.
   Run the following command with `<train_dir>` of your choice. Note everything down the line about PCA will be stored
   here, so make sure you have enough diskspace! It's perhaps safe to reserve 500G~1T. The spectral analyses are very
   diskspace intensive. Note below `<model_name_or_path>` can be one of `distilroberta-base`, `roberta-base`
   , `roberta-large`.

   ```bash
   CUDA_VISIBLE_DEVICES=0 python -m classification.spectral_analysis.rebuttal_neurips_2022 \
    --task "run_save_grads" \
    --train_dir <train_dir> \
    --model_name_or_path <model_name_or_path>
   ```

   2.2. Now run PCA with orthogonal iteration to extract top eigenvectors. The command below runs PCA based on 4k
   checkpoints (4k gradients stored along the trajectory), and extracts the top 1k eigenvalues and
   eigenvectors. `batch_size` can be set small to save memory (it affects distributed matmul). Note for
   the `roberta-large` experiment, you would likely need several GPUs. For reference, I used 4 A6000 (each with 48G
   VRAM) for that experiment. The code is written in a
   way so that computation can be distributed across many GPUs on a single machine, and should be
   fast with enough accelerators. **`<train_dir>` below must be the same as in the previous command.**

   ```bash
   python -m classification.spectral_analysis.rebuttal_neurips_2022 \
    --task "run_pca" \
    --train_dir <train_dir> \
    --n 4000 \
    --k 1000 \
    --num_power_iteration 10 \
    --batch_size 20
   ```

   2.3. For re-training in subspace, we need to specify to the command the place where the PCA results are stored in
   order to use those. The PCA results will be in `<train_dir>/orthproj/all/`. There will likely be a couple of
   checkpoints in this folder, each of which correspondes to a different iteration of the orthogonal iteration. Now run
   the
   following command. Note that below 1) `<output_dir>` should **not** be the same as `<train_dir>` to avoid
   overwriting previous PCA results, and 2) `<rank>` should be smaller than `k` from the previous command since it's the
   rank of the subspace.

    ```bash
    CUDA_VISIBLE_DEVICES=0 python -m classification.spectral_analysis.rebuttal_neurips_2022 \
      --task "run_retrain_single" \
      --output_dir <output_dir> \
      --orthogonal_projection_path "<train_dir>/orthproj/all/global_step_x.pt" \
      --rank <rank> \
      --model_name_or_path <model_name_or_path>
    ```

## Citation

If you found this codebase useful in your research, please consider citing:

```@misc{li2022when,
  doi = {10.48550/ARXIV.2207.00160},
  url = {https://arxiv.org/abs/2207.00160},
  author = {Li, Xuechen and Liu, Daogao and Hashimoto, Tatsunori and Inan, Huseyin A. and Kulkarni, Janardhan and Lee, Yin Tat and Thakurta, Abhradeep Guha},
  keywords = {Machine Learning (cs.LG), Cryptography and Security (cs.CR), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {When Does Differentially Private Learning Not Suffer in High Dimensions?},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
