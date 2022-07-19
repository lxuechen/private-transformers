## Reproducing results for sentence classification

### Requirements

In addition to requirements of the `private-transformers` package, install additional requirements by running the
following from the `examples` folder of this repo:

```bash
pip install -r classification/requirements.txt
```

This code is tested against `transformers==4.11.3`, but should also work for slightly earlier versions.

### Getting the data

This part of the codebase is adapted from the excellent work
by [[Gao et al., 2021](https://arxiv.org/pdf/2012.15723.pdf)]. We reuse their data pipeline. To obtain the data, run the
following:

```bash
cd data
bash download_dataset.sh
```

This should produce a `data/original` subfolder that contains all the data that we need.

### Running

Use the `run_wrapper.py` script in the folder. This Python script produces a text string for the command and runs it.

Supply at least 2 arguments:

- `--output_dir`: path to a folder where results will be written
- `--task_name`: name of task; one of `sst-2`, `qnli`, `qqp`, `mnli`

For instance, run the following under the `examples/` folder:

```bash
python -m classification.run_wrapper --output_dir <output_dir> --task_name <task_name>
```

The script by default uses ghost clipping, and the micro batch size is tweaked so that things should run smoothly even
on a Titan Xp with 12Gigs of VRAM. For SST-2, the run-time of this script on an RTX 3090 is roughly less than one and a
half hours. Larger datasets take longer to train.

Additional arguments:

- `--target_epsilon`: Target privacy spending
- `--model_name_or_path`: The pretrained model; one of `distilbert-base-uncased`, `bert-base-uncased`
  , `bert-large-uncased`, `distilroberta-base`, `roberta-base`, `roberta-large`
- `--few_shot_type`: Whether to use the generic prompt formatter described in Section 3.2 of our paper. `prompt` is to
  use, `finetune` is to not use.
- `--ghost_clipping`: Whether to use ghost clipping for memory saving; one of `yes`, `no`
  Note keeping other training hyperparameter (e.g., number of training epochs, clipping norm, learning rate) the same,
  things should still work
- `--data_dir`: Path to where data is stored; if data is obtained via the procedure described above, just stick to the
  defaults.

Training on the larger datasets for even more epochs should bring further performance gains.

### Notes

- We have reproduced some results in our paper with the codebase of
  a [concurrent anonymous submission](https://openreview.net/pdf?id=Q42f0dfjECO). Our modified version of their codebase
  is located at [this link](https://github.com/lxuechen/Differentially-Private-Fine-tuning-of-Language-Models). This
  code is modified from their original codebase and only optimizes the dense/linear layers in a Transformer model, and
  hence is not strictly full fine-tuning (since the embedding and LayerNorm layers aren't updated). The main difference
  from their original setup is that we run everything in full precision (i.e., fp32), not mixed-precision.
- We got similar results as those reported in the paper with Opacus, but with the embedding subnetworks (word embedding,
  positional embedding, token type embedding) frozen. Note that unfreezing the embedding subnetwork and plugging such a
  model (from HF) into Opacus would result in errors, due to how HF transformers are implemented.
