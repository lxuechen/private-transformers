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

import transformers
from ml_swissknife import utils


class BackwardHookMode(metaclass=utils.ContainerMeta):
    ghost_norm = "ghost_norm"
    ghost_grad = "ghost_grad"
    default = "default"


class ClippingMode(metaclass=utils.ContainerMeta):
    default = "default"  # Global fixed.
    ghost = "ghost"  # Global fixed clipping with ghost clipping.
    per_layer = "per_layer"  # Per layer fixed clipping.
    per_layer_percentile = "per_layer_percentile"  # Clip gradient per-layer based on gradient norm percentile.


class AccountingMode(metaclass=utils.ContainerMeta):
    rdp = "rdp"
    glw = "glw"
    all_ = "all"


SUPPORTED_TRANSFORMERS = (
    transformers.models.openai.modeling_openai.OpenAIGPTLMHeadModel,
    transformers.models.openai.modeling_openai.OpenAIGPTDoubleHeadsModel,
    transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
    transformers.models.gpt2.modeling_gpt2.GPT2DoubleHeadsModel,
    transformers.models.bert.modeling_bert.BertForSequenceClassification,
    transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification,
    transformers.models.albert.modeling_albert.AlbertForSequenceClassification,
    transformers.models.bart.modeling_bart.BartForConditionalGeneration,
    transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
    transformers.models.opt.modeling_opt.OPTForCausalLM,
    transformers.models.vit.modeling_vit.ViTForImageClassification,
    transformers.models.deit.modeling_deit.DeiTForImageClassification,
    transformers.models.beit.modeling_beit.BeitForImageClassification,
)
