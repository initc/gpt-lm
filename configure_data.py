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

"""parses arguments and preps data loader"""

import os
import torch
import data_utils

from data_utils.datasets import CLMDataset, CLMTaskDataset
from data_utils.datasets import FuseDataset, FuseSampler
from data_utils.tokenization import BertWordPieceTokenizer
from data_utils.dataset_finetune import QAClmDataset


class DataConfig:

    def __init__(self, defaults={}):
        super(DataConfig, self).__init__()
        self.defaults = defaults

    def apply(self, args):
        print('configuring data')
        self.apply_defaults(args)
        return make_loaders(args)

    def set_defaults(self, **kwargs):
        for k, v in kwargs.items():
            self.defaults[k] = v

    def apply_defaults(self, args):
        for k, v in self.defaults.items():
            k = k.replace('-', '_')
            if not hasattr(args, k):
                setattr(args, k, v)


def make_data_loader(dataset, batch_size, args):
    shuffle = args.shuffle
    if shuffle:
        sampler = data_utils.samplers.RandomSampler(dataset, replacement=True,
                                                    num_samples=batch_size * args.train_iters)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    world_size = args.world_size
    rank = args.rank
    distributed = world_size > 1
    drop_last = distributed

    if distributed:
        batch_sampler = data_utils.samplers.DistributedBatchSampler(sampler,
                                                                    batch_size,
                                                                    drop_last,
                                                                    rank,
                                                                    world_size)
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                      batch_size,
                                                      drop_last)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    return data_loader


def make_tfrecord_loaders(args):
    """Load train/val/test dataset from shuffled TFRecords"""

    import data_utils.tf_dl
    data_set_args = {'batch_size': args.batch_size,
                     'max_seq_len': args.seq_length,
                     'max_preds_per_seq': args.max_preds_per_seq,
                     'train': True,
                     'num_workers': max(args.num_workers, 1),
                     'seed': args.seed + args.rank + 1,
                     'threaded_dl': args.num_workers > 0
                     }
    train = data_utils.tf_dl.TFRecordDataLoader(args.train_data,
                                                **data_set_args)
    data_set_args['train'] = False
    if args.eval_seq_length is not None:
        data_set_args['max_seq_len'] = args.eval_seq_length
    if args.eval_max_preds_per_seq is not None:
        data_set_args['max_preds_per_seq'] = args.eval_max_preds_per_seq
    valid = None
    if args.valid_data is not None:
        valid = data_utils.tf_dl.TFRecordDataLoader(args.valid_data,
                                                    **data_set_args)
    test = None
    if args.test_data is not None:
        test = data_utils.tf_dl.TFRecordDataLoader(args.test_data,
                                                   **data_set_args)
    tokenizer = data_utils.make_tokenizer(args.tokenizer_type,
                                          train,
                                          args.tokenizer_path,
                                          args.vocab_size,
                                          args.tokenizer_model_type,
                                          cache_dir=args.cache_dir)

    return (train, valid, test), tokenizer


def make_loaders_2(args):
    tokenizer = BertWordPieceTokenizer("bert-base-chinese", cache_dir="temp_cache_dir")
    if args.no_nsp:
        train, valid_dataset = FuseDataset.load_dataset_no_nsp(tokenizer, args)
    else:
        train, valid_dataset = FuseDataset.load_dataset(tokenizer, args)
    print("| Load train dataset :{}".format(len(train)))
    for d in valid_dataset:
        print("| Load valid dataset :{}".format(len(d)))
    train_simpler = FuseSampler(train, args.world_size, args.rank)
    train = torch.utils.data.DataLoader(train, batch_sampler=train_simpler, collate_fn=train.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    print("| After train batch size {}".format(len(train)))
    valids = []
    for data in valid_dataset:
        s = FuseSampler(data, args.world_size, args.rank)
        l = torch.utils.data.DataLoader(data, batch_sampler=s, collate_fn=data.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
        print("| After valid batch size {}".format(len(l)))
        valids.append(l)

    return (train, valids), tokenizer


def only_gpt_loader(args):
    tokenizer = BertWordPieceTokenizer("bert-base-chinese", cache_dir="temp_cache_dir")
    datapath = args.data
    train_prefix = args.train_prefix
    valid_prefix = args.valid_prefix
    train_data = os.path.join(datapath, train_prefix)
    valid_datas = [os.path.join(datapath, prefix) for prefix in valid_prefix.split(",")]

    train_data = CLMDataset(train_data, tokenizer, args.train_batch, args.max_tokens, world_size=args.world_size,
                            max_lens=args.max_lens, no_cache=args.no_cache, drop_first_token=args.drop_first_token)
    print("| Load train dataset :{}".format(len(train_data)))
    train_simpler = FuseSampler(train_data, args.world_size, args.rank)
    train = torch.utils.data.DataLoader(train_data, batch_sampler=train_simpler, collate_fn=train_data.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    print("| After train batch size {}".format(len(train)))
    valids = []
    for data in valid_datas:
        d = CLMDataset(data, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size,
                       max_lens=args.max_lens, no_cache=args.no_cache, drop_first_token=args.drop_first_token)
        print("| Load valid dataset :{}".format(len(d)))
        simpler = FuseSampler(d, args.world_size, args.rank)
        d = torch.utils.data.DataLoader(d, batch_sampler=simpler, collate_fn=d.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
        print("| After valid batch size {}".format(len(d)))
        valids.append(d)
    return (train, valids), tokenizer


def load_fine_tune_qa_data(args):
    tokenizer = BertWordPieceTokenizer("bert-base-chinese", cache_dir="temp_cache_dir")
    datapath = args.data
    train_prefix = args.train_prefix
    valid_prefix = args.valid_prefix
    train_data = os.path.join(datapath, train_prefix)
    valid_datas = [os.path.join(datapath, prefix) for prefix in valid_prefix.split(",")]

    train_data = QAClmDataset(train_data, tokenizer, args.train_batch, args.max_tokens, world_size=args.world_size, max_lens=args.max_lens, no_cache=args.no_cache, use_token_type=args.use_token_type, use_task_embedding=args.use_task_embedding)
    print("| Load train dataset :{}".format(len(train_data)))
    train_simpler = FuseSampler(train_data, args.world_size, args.rank)
    train = torch.utils.data.DataLoader(train_data, batch_sampler=train_simpler, collate_fn=train_data.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    print("| After train batch size {}".format(len(train)))
    valids = []
    for data in valid_datas:
        d = QAClmDataset(data, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size ,max_lens=args.max_lens, no_cache=args.no_cache, use_token_type=args.use_token_type, use_task_embedding=args.use_task_embedding)
        print("| Load valid dataset :{}".format(len(d)))
        simpler = FuseSampler(d, args.world_size, args.rank)
        d = torch.utils.data.DataLoader(d, batch_sampler=simpler, collate_fn=d.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
        print("| After valid batch size {}".format(len(d)))
        valids.append(d)
    return (train, valids), tokenizer


def multi_task_loader(args):
    tokenizer = BertWordPieceTokenizer("bert-base-chinese", cache_dir="temp_cache_dir")
    datapath = args.data
    train_prefix = args.train_prefix
    valid_prefix = args.valid_prefix
    train_data = os.path.join(datapath, train_prefix)
    valid_datas = [os.path.join(datapath, prefix) for prefix in valid_prefix.split(",")]

    train_data = CLMTaskDataset(train_data, tokenizer, args.train_batch, args.max_tokens, world_size=args.world_size,max_lens=args.max_lens, no_cache=args.no_cache, use_cls_special=args.use_cls_special)
    print("| Load train dataset :{}".format(len(train_data)))
    train_simpler = FuseSampler(train_data, args.world_size, args.rank)
    train = torch.utils.data.DataLoader(train_data, batch_sampler=train_simpler, collate_fn=train_data.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    print("| After train batch size {}".format(len(train)))
    valids = []
    for data in valid_datas:
        d = CLMTaskDataset(data, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size, max_lens=args.max_lens, no_cache=args.no_cache, use_cls_special=args.use_cls_special)
        print("| Load valid dataset :{}".format(len(d)))
        simpler = FuseSampler(d, args.world_size, args.rank)
        d = torch.utils.data.DataLoader(d, batch_sampler=simpler, collate_fn=d.collate,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
        print("| After valid batch size {}".format(len(d)))
        valids.append(d)
    return (train, valids), tokenizer


def make_fine_tune(args):
    if args.qa_style_data:
        return load_fine_tune_qa_data(args)


def make_loaders(args):
    """makes training/val/test"""
    if args.fine_tune:
        return make_fine_tune(args)
    if args.multi_doc:
        return multi_task_loader(args)
    if args.only_gpt:
        return only_gpt_loader(args)
    return make_loaders_2(args)


def configure_data():
    """add cmdline flags for configuring datasets"""
    # These are options that are used by data_utils, but are either
    # deprecated or not meant to be exposed to the command line user.
    # These options are intneded to be set in code by specific scripts.
    defaults = {
        'world_size': 1,
        'rank': -1,
        'persist_state': 0,
        'lazy': False,
        'transpose': False,
        'data_set_type': 'supervised',
        'seq_length': 256,
        'eval_seq_length': 256,
        'samples_per_shard': 100
    }

    return DataConfig(defaults=defaults)
