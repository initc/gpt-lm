# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for wrapping BertModel."""

import torch
import torch.nn.functional as F

from .modeling import BertConfig
from .modeling import BertForPreTraining
from .modeling import BertLayerNorm
import pdb

def get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0}
    for module_ in module.modules():
        if isinstance(module_, (BertLayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


class BertModel(torch.nn.Module):

    def __init__(self, tokenizer, args):
        super(BertModel, self).__init__()
        if args.load_model:
            self.model = BertForPreTraining.load_model(
                args.model_config, fp32_embedding=args.fp32_embedding, fp32_layernorm=args.fp32_layernorm, fp32_tokentypes=args.fp32_tokentypes, layernorm_epsilon=args.layernorm_epsilon)
        else:
            self.model = BertForPreTraining.build_model(args.model_config, fp32_embedding=args.fp32_embedding, fp32_layernorm=args.fp32_layernorm,fp32_tokentypes=args.fp32_tokentypes,layernorm_epsilon=args.layernorm_epsilon)

    def forward(self, input_tokens, token_type_ids=None, task_type_ids=None,
                attention_mask=None, past=None, checkpoint_activations=False, clm=False):
        return self.model(
            input_tokens, token_type_ids, task_type_ids, attention_mask, past=past,
            checkpoint_activations=checkpoint_activations, clm=clm)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination=destination, prefix=prefix,
                                     keep_vars=keep_vars)

    def reorder_encoder_out(self, encoder_out, new_order):

        if encoder_out['input_tokens'] is not None:
            encoder_out['input_tokens'] = \
                encoder_out['input_tokens'].index_select(0, new_order)

        if "task_type_ids" in encoder_out and encoder_out['task_type_ids'] is not None:
            encoder_out['task_type_ids'] = \
                encoder_out['task_type_ids'].index_select(0, new_order)

        if "token_type_ids" in encoder_out and encoder_out['token_type_ids'] is not None:
            encoder_out['token_type_ids'] = \
                encoder_out['token_type_ids'].index_select(0, new_order)

        return encoder_out

    def get_normalized_probs(self, net_output, log_probs=False):
        if log_probs:
            return F.log_softmax(net_output, dim=-1)
        else:
            return F.softmax(net_output, dim=-1)