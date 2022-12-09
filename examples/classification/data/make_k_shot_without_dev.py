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

"""The datasets in the k-shot folder contain dev.tsv; we make the test set the dev set in the new k-shot.

python -m classification.data.make_k_shot_without_dev
"""
import os

from ml_swissknife import utils

join = os.path.join

base_dir = '/nlp/scr/lxuechen/data/lm-bff/data/k-shot'
new_dir = '/nlp/scr/lxuechen/data/lm-bff/data/k-shot-no-dev'

task_names = ("SST-2", "QNLI", "MNLI", "QQP")
for task_name in task_names:
    folder = join(base_dir, task_name)
    new_folder = join(new_dir, task_name)

    for name in utils.listdir(folder):
        subfolder = join(folder, name)
        new_subfolder = join(new_folder, name)
        os.makedirs(new_subfolder, exist_ok=True)

        train = join(subfolder, 'train.tsv')
        new_train = join(new_subfolder, 'train.tsv')
        os.system(f'cp {train} {new_train}')

        if task_name == "MNLI":
            test = join(subfolder, 'test_matched.tsv')
            new_dev = join(new_subfolder, 'dev_matched.tsv')
            os.system(f'cp {test} {new_dev}')

            test = join(subfolder, 'test_mismatched.tsv')
            new_dev = join(new_subfolder, 'dev_mismatched.tsv')
            os.system(f'cp {test} {new_dev}')
        else:
            test = join(subfolder, 'test.tsv')
            new_dev = join(new_subfolder, 'dev.tsv')
            os.system(f'cp {test} {new_dev}')
