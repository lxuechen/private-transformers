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

"""
LoRA layers.

This version does not have merged weights for zero latency inference. It makes the code easier to read and maintain.
Adapted from
    https://github.com/microsoft/LoRA
    https://www.microsoft.com/en-us/research/project/dp-transformers/
"""

import torch
import transformers
from torch import nn


class DPMergedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_r=0,
        lora_alpha=1.,
        lora_dropout=0.,
    ):
        super(DPMergedLinear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        if self.lora_r > 0:
            self.lora_A = nn.Linear(in_features=in_features, out_features=lora_r, bias=False)
            self.lora_B = nn.Linear(in_features=lora_r, out_features=out_features, bias=False)
            self.scaling = self.lora_alpha / lora_r
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        result = self.linear(x)
        if self.lora_r > 0:
            after_dropout = self.lora_dropout(x)
            after_A = self.lora_A(after_dropout)
            after_B = self.lora_B(after_A)
            result += after_B * self.scaling
        return result

    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.lora_r > 0:
            self.lora_A.reset_parameters()
            self.lora_B.weight.data.zero_()

    @staticmethod
    def from_transformers_conv1d(
        original_layer,
        lora_r=0,
        lora_alpha=1.,
        lora_dropout=0.,
    ) -> "DPMergedLinear":
        lora_layer = DPMergedLinear(
            in_features=original_layer.weight.shape[0],
            out_features=original_layer.weight.shape[1],
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(original_layer.weight.device)
        lora_layer.linear.weight.data.copy_(original_layer.weight.T.data)
        lora_layer.linear.bias.data.copy_(original_layer.bias.data)
        return lora_layer


def convert_gpt2_attention_to_lora(
    model: transformers.GPT2PreTrainedModel,
    lora_r=0,
    lora_alpha=1.,
    lora_dropout=0.,
) -> transformers.GPT2PreTrainedModel:
    if not isinstance(model, transformers.GPT2PreTrainedModel):
        raise TypeError("Requires a GPT2 model")

    if not hasattr(model, "h") and hasattr(model, "transformer"):
        transformer = model.transformer
    else:
        transformer = model

    for h_i in transformer.h:
        new_layer = DPMergedLinear.from_transformers_conv1d(
            original_layer=h_i.attn.c_attn,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        h_i.attn.c_attn = new_layer

    return model


def mark_only_lora_as_trainable(model: torch.nn.Module) -> None:
    model.requires_grad_(True)
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
