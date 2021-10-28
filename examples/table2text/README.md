## Reproducing results for table-to-text generation

### Requirements

In addition to requirements of the `private-transformers` package, install additional requirements by running the
following from the `examples` folder of this repo:

```bash
pip install -r table2text/requirements.txt
```

### Getting the data

We host the datasets for E2E and DART on Google drive at
this [link](https://drive.google.com/file/d/1Re1wyUPtS3IalSsVVJhSg2sn8UNa7DM7/view?usp=sharing). Download and unzip the
folder to a reasonable location. The unzipped folder is named `prefix-tuning`, since it's adapted from data used in the
prefix-tuning paper.

### Running

Use the `run.sh` script in the folder.

Supply at least 3 arguments:

- `output_dir`: path to a folder where results will be written
- `data_folder`: path to the unzipped data folder
- `task_mode`: name of task; one of `e2e` and `dart`

For instance, to fine-tune GPT-2 on E2E at ε = 8, run the following from the `examples` folder of this repo:

```bash
bash table2text/run.sh <output_dir> <data_folder> "e2e"
```

The script by default uses ghost clipping, and the micro batch size is tweaked so that things should run smoothly even
on a Titan Xp with 12Gigs of VRAM. For E2E, the run-time of this script on an RTX 3090 is roughly less than one and a
half hours.

Feel free to toggle other arguments like `target_epsilon` and `model_name_or_path` of the `run.sh` script to use
different privacy levels and models. The other hyperparameters should still mostly work for workloads with varied model
and privacy level.
