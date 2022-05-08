import torch

task_name2suffix_name = {"sst-2": "GLUE-SST-2", "mnli": "MNLI", "qqp": "QQP", "qnli": "QNLI"}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
true_tags = ('y', 'yes', 't', 'true')
