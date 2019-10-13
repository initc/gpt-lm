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
"""dataset objects for jsons, csvs, and BERT datasets"""
import os
import torch
import random
from data_utils.utils import numpy_seed
import itertools

from torch.utils import data
import numpy as np

from data_utils import indexed_dataset


class CLMTaskDataset(data.Dataset):

    def __init__(self, path, tokenizer, batch_size, max_tokens, world_size=1, max_lens=510, seed=512, no_cache=False, use_cls_special=False):
        self.sizes = None
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.world_size = world_size
        self.max_lens = max_lens
        self.seed = seed + 20
        self.no_cache = no_cache
        self.use_cls_special = use_cls_special
        self.data = self.read_data(path)
        self.padding = tokenizer.pad()
        self.frozen_indices = self.init_data_indices()
        self.frozen_batch = self.fill_batch(list(self.batch_by_size()))

    @property
    def supports_prefetch(self):
        if self.data.supports_prefetch:
            return True
        else:
            return False

    def prefetch(self, indices):
        if self.supports_prefetch:
            self.data.prefetch(indices)
            print("| Fetch all data into memory")

    def fill_batch(self, batches):
        size = len(batches)
        rng = random.Random(self.seed)
        rng.shuffle(batches)
        fill_num = self.world_size - size % self.world_size
        if fill_num == self.world_size:
            return batches
        batches.extend(batches[-fill_num:])
        assert len(batches) % self.world_size == 0
        return batches

    def index_len(self):
        return len(self.data)

    def read_data(self, path):
        if self.no_cache:
            data = indexed_dataset.IndexedDataset(path, fix_lua_indexing=True)
        else:
            data = indexed_dataset.IndexedCachedDataset(path, fix_lua_indexing=True)
        self.sizes = data.sizes
        return data

    def init_data_indices(self):
        return self.ordered_indices()

    def filter_by_size(self, indices):
        ignore = []
        to_samll = []
        for ind in indices:
            size = self.num_tokens(ind)
            if size > self.max_lens:
                ignore.append(ind)
            elif size < 10:
                to_samll.append(ind)
            else:
                yield ind
        if len(ignore) > 0:
            print(
                "WARNING: {} samples have invalid sizes and will be skipped, max-qa-len={}, first few sample ids={}".format(
                    len(ignore), self.max_lens, ignore[:10]))
        if len(to_samll) > 0:
            print(
                "WARNING: {} samples have invalid sizes and will be skipped, max-qa-len={}, first few sample ids={}".format(
                    len(to_samll), 10, to_samll[:10]))

    def __len__(self):
        return len(self.frozen_indices)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        with numpy_seed(self.seed):
            indices = np.random.permutation(len(self.data))
        lens = []
        for inx_a in indices:
            lens.append(self.num_tokens(inx_a))
        indices = indices[np.argsort(np.array(lens), kind='mergesort')]
        indices = list(self.filter_by_size(indices))
        return indices

    def num_tokens(self, idx):
        return self.sizes[idx]

    def batch_by_size(self):
        max_tokens = self.max_tokens
        indices = self.frozen_indices
        batch = []
        tokens = 0

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if num_tokens > max_tokens:
                return True
            # if len(batch) == self.batch_size:
            #     return True
            return False

        for idx in indices:
            idx_max_token = self.num_tokens(idx)
            tokens = max(tokens, idx_max_token)
            if is_batch_full(tokens * (len(batch) + 1)):
                yield batch
                batch = []
                tokens = 0
            batch.append(idx)
        if len(batch) >= 1:
            yield batch

    def __getitem__(self, idx):
        tokens = self.data[idx].tolist()
        task = tokens[0]
        tokens = tokens[1:]
        token_sep = [self.tokenizer.sep()]
        if self.use_cls_special:
            token_cls = [self.tokenizer.cls_style(task)]
        else:
            token_cls = [self.tokenizer.cls()]
        lm_labels = tokens + token_sep
        tokens = token_cls + tokens
        assert len(tokens) == len(lm_labels)
        return {"id": int(idx),
                "tokens": tokens,
                "lm_labels": lm_labels,
                "task": task,
                "model-type": "CLM"
                }

    def collate(self, samples):
        padding = self.tokenizer.pad()
        tensor_size = max([len(candidate["tokens"]) for candidate in samples])
        batch = len(samples)

        input_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        attention_mask = torch.LongTensor(batch, tensor_size).fill_(0)
        lm_labels = torch.LongTensor(batch, tensor_size).fill_(-1)
        nsp_labels = torch.LongTensor(batch).fill_(-1)
        task_type_ids = None
        if not self.use_cls_special:
            task_type_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        id_ = torch.LongTensor(batch).fill_(padding)
        ntokens = 0
        for i, candidates in enumerate(samples):
            assert candidates["model-type"].lower() == "clm"
            tokens = candidates["tokens"]
            lm_tokens = candidates["lm_labels"]
            lens = len(tokens)

            input_ids[i, :lens] = torch.Tensor(tokens)
            lm_labels[i, :lens] = torch.Tensor(lm_tokens)
            if not self.use_cls_special:
                task_type_ids[i] = candidates["task"]
            attention_mask[i, :lens] = 1

            ntokens += lens
            id_[i] = candidates["id"]
            id_[i] = candidates["id"]
        batch = {
            'id': id_,
            'nsentences': batch,
            "ntokens": ntokens,
            'net_input': {
                'input_tokens': input_ids,
                'task_type_ids': task_type_ids,
                'attention_mask': attention_mask,
                'clm': True
            },
            'target': lm_labels,
            'nsp_labels': nsp_labels
        }

        return batch


class MLMDataset(data.Dataset):

    def __init__(self, path, tokenizer, batch_size, max_tokens, world_size=1, max_lens=510, seed=512, mask_lm_prob=0.15, max_preds_per_seq=80, no_cache=False, drop_first_token=False, use_task_embed=False):
        self.a_size = None
        self.b_size = None
        self.tokenizer = tokenizer
        self.vocab_words = list(self.tokenizer.text_token_vocab.values())
        self.world_size = world_size
        self.mask_lm_prob = mask_lm_prob
        self.max_preds_per_seq = max_preds_per_seq
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.max_lens = max_lens
        self.no_cache = no_cache
        self.drop_token = drop_first_token
        self.use_task_embed = use_task_embed
        self.seed = seed
        self.data = self.read_data(path)
        self.sizes = self.data.sizes
        self.padding = tokenizer.pad()
        self.frozen_indices = self.init_data_indices()
        self.frozen_batch = self.fill_batch(list(self.batch_by_size()))

    @property
    def supports_prefetch(self):
        if self.data.supports_prefetch:
            return True
        else:
            return False

    def prefetch(self, indices):
        if self.supports_prefetch:
            self.data.prefetch(indices)
            print("| Fetch all data into memory")

    def fill_batch(self, batches):
        b_size = len(batches)
        rng = random.Random(self.seed)
        rng.shuffle(batches)
        fill_num = self.world_size - b_size % self.world_size
        if fill_num == self.world_size:
            return batches
        batches.extend(batches[-fill_num:])
        assert len(batches) % self.world_size == 0
        return batches

    def index_len(self):
        return len(self.sizes)

    def read_data(self, path):
        if self.no_cache:
            data = indexed_dataset.IndexedDataset(path, fix_lua_indexing=True)
        else:
            data = indexed_dataset.IndexedCachedDataset(path, fix_lua_indexing=True)
        return data

    def init_data_indices(self):
        return self.ordered_indices()

    def filter_by_size(self, indices):
        ignore = []
        error = []
        for ind in indices:
            size = self.num_tokens(ind)
            if size > self.max_lens:
                ignore.append(ind)
            else:
                yield ind
        if len(ignore) > 0:
            print(
                "WARNING: {} samples have invalid sizes and will be skipped, max-qa-len={}, first few sample ids={}".format(
                    len(ignore), self.max_lens, ignore[:10]))
        if len(error) > 0:
            print(
                "WARNING: {} samples have too small size and will be skipped, first few sample ids={}".format(
                    len(error), error[:10]))

    def __len__(self):
        return len(self.frozen_indices)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        with numpy_seed(self.seed):
            indices = np.random.permutation(len(self.data))
        lens = []
        for inx_a in indices:
            lens.append(self.num_tokens(inx_a))
        indices = indices[np.argsort(np.array(lens), kind='mergesort')]
        indices = list(self.filter_by_size(indices))
        return indices

    def num_tokens(self, idx):
        return self.sizes[idx]

    def batch_by_size(self):
        max_tokens = self.max_tokens
        # n_gpu = self.n_gpu
        indices = self.frozen_indices
        batch = []
        tokens = 0

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if num_tokens > max_tokens:
                return True
            # if len(batch) == self.batch_size:
            #     return True
            return False

        for idx in indices:
            idx_max_token = self.num_tokens(idx)
            tokens = max(tokens, idx_max_token)
            if is_batch_full(tokens * (len(batch) + 1)):
                yield batch
                batch = []
                tokens = 0
            batch.append(idx)
        if len(batch) >= 1:
            yield batch

    def __getitem__(self, idx):
        rng = random.Random(int(idx))
        tokens = self.data[idx].tolist()
        if self.use_task_embed:
            assert not self.drop_token
        task_type = None
        if self.use_task_embed:
            task_type = tokens[0]
            tokens = tokens[1:]
        if self.drop_token:
            tokens = tokens[1:]

        tokens, mask, mask_labels = self.create_masked_lm_predictions(tokens, self.mask_lm_prob,
                                                                      self.max_preds_per_seq,
                                                                      self.vocab_words, rng)
        assert len(tokens) == len(mask_labels)
        return {"id": int(idx),
                "tokens": tokens,
                "lm_labels": mask_labels,
                "ntokens": sum(mask),
                "task_type": task_type,
                "model-type": "MLM"
                }

    def create_masked_lm_predictions(self, tokens, mask_lm_prob, max_preds_per_seq, vocab_words, rng):
        token_sep = [self.tokenizer.sep()]
        token_cls = [self.tokenizer.cls()]
        tokens = token_cls + tokens + token_sep
        cand_indices = [idx for idx in range(1, len(tokens) - 1)]
        rng.shuffle(cand_indices)
        num_to_predict = min(max_preds_per_seq, max(1, int(round(len(tokens) * mask_lm_prob))))
        mask = [0] * len(tokens)
        mask_labels = [-1] * len(tokens)
        for idx in cand_indices[:num_to_predict]:
            mask[idx] = 1
            label = self.mask_token(idx, tokens, vocab_words, rng)
            mask_labels[idx] = label
        return tokens, mask, mask_labels

    def mask_token(self, idx, tokens, vocab_words, rng):
        label = tokens[idx]
        if rng.random() < 0.8:
            new_label = self.tokenizer.mask()
        else:
            if rng.random() < 0.5:
                new_label = label
            else:
                new_label = rng.choice(vocab_words)
        tokens[idx] = new_label
        return label

    def collate(self, samples):
        padding = self.tokenizer.pad()
        tensor_size = max([len(candidate["tokens"]) for candidate in samples])
        batch = len(samples)

        input_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        task_type_ids = None
        if self.use_task_embed:
            task_type_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        attention_mask = torch.LongTensor(batch, tensor_size).fill_(0)
        lm_labels = torch.LongTensor(batch, tensor_size).fill_(-1)
        nsp_labels = torch.LongTensor(batch).fill_(-1)

        id_ = torch.LongTensor(batch).fill_(padding)
        ntokens = 0
        for i, candidates in enumerate(samples):
            assert candidates["model-type"].lower() == "mlm"
            tokens = candidates["tokens"]
            task_type = candidates["task_type"]
            lm_tokens = candidates["lm_labels"]
            lens = len(tokens)

            input_ids[i, :lens] = torch.Tensor(tokens)
            if self.use_task_embed:
                task_type_ids[i] = task_type
            lm_labels[i, :lens] = torch.Tensor(lm_tokens)
            attention_mask[i, :lens] = 1
            ntokens += candidates['ntokens']
            id_[i] = candidates["id"]
        batch = {
            'id': id_,
            'nsentences': batch,
            "ntokens": ntokens,
            'net_input': {
                'input_tokens': input_ids,
                'task_type_ids': task_type_ids,
                'attention_mask': attention_mask,
                'clm': False
            },
            'target': lm_labels,
            'nsp_labels': nsp_labels
        }

        return batch


class CLMDataset(data.Dataset):

    def __init__(self, path, tokenizer, batch_size, max_tokens, world_size=1, max_lens=510, seed=512, no_cache=False,
                 drop_first_token=False):
        self.sizes = None
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.world_size = world_size
        self.max_lens = max_lens
        self.seed = seed + 20
        self.no_cache = no_cache
        self.drop_token = drop_first_token
        self.data = self.read_data(path)
        self.padding = tokenizer.pad()
        self.frozen_indices = self.init_data_indices()
        self.frozen_batch = self.fill_batch(list(self.batch_by_size()))

    @property
    def supports_prefetch(self):
        if self.data.supports_prefetch:
            return True
        else:
            return False

    def prefetch(self, indices):
        if self.supports_prefetch:
            self.data.prefetch(indices)
            print("| Fetch all data into memory")

    def fill_batch(self, batches):
        size = len(batches)
        rng = random.Random(self.seed)
        rng.shuffle(batches)
        fill_num = self.world_size - size % self.world_size
        if fill_num == self.world_size:
            return batches
        batches.extend(batches[-fill_num:])
        assert len(batches) % self.world_size == 0
        return batches

    def index_len(self):
        return len(self.data)

    def read_data(self, path):
        if self.no_cache:
            data = indexed_dataset.IndexedDataset(path, fix_lua_indexing=True)
        else:
            data = indexed_dataset.IndexedCachedDataset(path, fix_lua_indexing=True)
        self.sizes = data.sizes
        return data

    def init_data_indices(self):
        return self.ordered_indices()

    def filter_by_size(self, indices):
        ignore = []
        to_samll = []
        for ind in indices:
            size = self.num_tokens(ind)
            if size > self.max_lens:
                ignore.append(ind)
            elif size < 10:
                to_samll.append(ind)
            else:
                yield ind
        if len(ignore) > 0:
            print(
                "WARNING: {} samples have invalid sizes and will be skipped, max-qa-len={}, first few sample ids={}".format(
                    len(ignore), self.max_lens, ignore[:10]))
        if len(to_samll) > 0:
            print(
                "WARNING: {} samples have invalid sizes and will be skipped, min-qa-len={}, first few sample ids={}".format(
                    len(to_samll), 10, to_samll[:10]))

    def __len__(self):
        return len(self.frozen_indices)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        with numpy_seed(self.seed):
            indices = np.random.permutation(len(self.data))
        lens = []
        for inx_a in indices:
            lens.append(self.num_tokens(inx_a))
        indices = indices[np.argsort(np.array(lens), kind='mergesort')]
        indices = list(self.filter_by_size(indices))
        return indices

    def num_tokens(self, idx):
        return self.sizes[idx]

    def batch_by_size(self):
        max_tokens = self.max_tokens
        indices = self.frozen_indices
        batch = []
        tokens = 0

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if num_tokens > max_tokens:
                return True
            # if len(batch) == self.batch_size:
            #     return True
            return False

        for idx in indices:
            idx_max_token = self.num_tokens(idx)
            tokens = max(tokens, idx_max_token)
            if is_batch_full(tokens * (len(batch) + 1)):
                yield batch
                batch = []
                tokens = 0
            batch.append(idx)
        if len(batch) >= 1:
            yield batch

    def __getitem__(self, idx):
        tokens = self.data[idx].tolist()
        if self.drop_token:
            tokens = tokens[1:]
        token_sep = [self.tokenizer.sep()]
        token_cls = [self.tokenizer.cls()]
        lm_labels = tokens + token_sep
        tokens = token_cls + tokens
        assert len(tokens) == len(lm_labels)
        return {"id": int(idx),
                "tokens": tokens,
                "lm_labels": lm_labels,
                "model-type": "CLM"
                }

    def collate(self, samples):
        padding = self.tokenizer.pad()
        tensor_size = max([len(candidate["tokens"]) for candidate in samples])
        batch = len(samples)

        input_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        attention_mask = torch.LongTensor(batch, tensor_size).fill_(0)
        lm_labels = torch.LongTensor(batch, tensor_size).fill_(-1)
        nsp_labels = torch.LongTensor(batch).fill_(-1)

        id_ = torch.LongTensor(batch).fill_(padding)
        ntokens = 0
        for i, candidates in enumerate(samples):
            assert candidates["model-type"].lower() == "clm"
            tokens = candidates["tokens"]
            lm_tokens = candidates["lm_labels"]
            lens = len(tokens)

            input_ids[i, :lens] = torch.Tensor(tokens)
            lm_labels[i, :lens] = torch.Tensor(lm_tokens)
            attention_mask[i, :lens] = 1

            ntokens += lens
            id_[i] = candidates["id"]
            id_[i] = candidates["id"]
        batch = {
            'id': id_,
            'nsentences': batch,
            "ntokens": ntokens,
            'net_input': {
                'input_tokens': input_ids,
                'attention_mask': attention_mask,
                'clm': True
            },
            'target': lm_labels,
            'nsp_labels': nsp_labels
        }

        return batch


class BertPairDataset(data.Dataset):

    def __init__(self, path, tokenizer, batch_size, max_tokens, world_size=1, max_lens=510, seed=512, mask_lm_prob=0.15,
                 max_preds_per_seq=80, no_cache=False):
        self.a_size = None
        self.b_size = None
        self.tokenizer = tokenizer
        self.vocab_words = list(self.tokenizer.text_token_vocab.values())
        self.world_size = world_size
        self.mask_lm_prob = mask_lm_prob
        self.max_preds_per_seq = max_preds_per_seq
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.max_lens = max_lens
        self.no_cache = no_cache
        self.seed = seed
        self.data = self.read_data(path)
        self.padding = tokenizer.pad()
        self.frozen_indices = self.init_data_indices()
        # if self.data[0].supports_prefetch and self.data[1].supports_prefetch:
        #     self.data[0].prefetch(self.frozen_indices)
        #     self.data[1].prefetch(self.frozen_indices)
        self.frozen_batch = self.fill_batch(list(self.batch_by_size()))

    @property
    def supports_prefetch(self):
        if self.data[0].supports_prefetch and self.data[1].supports_prefetch:
            return True
        else:
            return False

    def prefetch(self, indices):
        if self.supports_prefetch:
            self.data[0].prefetch(indices)
            self.data[1].prefetch(indices)
            print("| Fetch all data into memory")

    def fill_batch(self, batches):
        b_size = len(batches)
        rng = random.Random(self.seed)
        rng.shuffle(batches)
        fill_num = self.world_size - b_size % self.world_size
        if fill_num == self.world_size:
            return batches
        batches.extend(batches[-fill_num:])
        assert len(batches) % self.world_size == 0
        return batches

    def index_len(self):
        return len(self.a_size)

    def read_data(self, path):
        A_path = path + "-A"
        B_path = path + "-B"
        if self.no_cache:
            A_data = indexed_dataset.IndexedDataset(A_path, fix_lua_indexing=True)
            B_data = indexed_dataset.IndexedDataset(B_path, fix_lua_indexing=True)
        else:
            A_data = indexed_dataset.IndexedCachedDataset(A_path, fix_lua_indexing=True)
            B_data = indexed_dataset.IndexedCachedDataset(B_path, fix_lua_indexing=True)
        self.a_size = A_data.sizes
        self.b_size = B_data.sizes
        return (A_data, B_data)

    def init_data_indices(self):
        return self.ordered_indices()

    def filter_by_size(self, indices):
        ignore = []
        error = []
        for ind in indices:
            size = self.num_tokens(ind)
            a_size = self.a_size[ind]
            b_size = self.b_size[ind]
            if size > self.max_lens:
                ignore.append(ind)
            elif a_size <= 1 or b_size < 1:
                error.append(ind)
            else:
                yield ind
        if len(ignore) > 0:
            print(
                "WARNING: {} samples have invalid sizes and will be skipped, max-qa-len={}, first few sample ids={}".format(
                    len(ignore), self.max_lens, ignore[:10]))
        if len(error) > 0:
            print(
                "WARNING: {} samples have too small size and will be skipped, first few sample ids={}".format(
                    len(error), error[:10]))

    def __len__(self):
        return len(self.frozen_indices)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
            on this order."""
        with numpy_seed(self.seed):
            indices = np.random.permutation(len(self.data[0]))
        lens = []
        for inx_a in indices:
            lens.append(self.num_tokens(inx_a))
        indices = indices[np.argsort(np.array(lens), kind='mergesort')]
        indices = list(self.filter_by_size(indices))
        return indices

    def num_tokens(self, idx):
        return self.a_size[idx] + self.b_size[idx]

    def batch_by_size(self):
        max_tokens = self.max_tokens
        # n_gpu = self.n_gpu
        indices = self.frozen_indices
        batch = []
        tokens = 0

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if num_tokens > max_tokens:
                return True
            # if len(batch) == self.batch_size:
            #     return True
            return False

        for idx in indices:
            idx_max_token = self.num_tokens(idx)
            tokens = max(tokens, idx_max_token)
            if is_batch_full(tokens * (len(batch) + 1)):
                yield batch
                batch = []
                tokens = 0
            batch.append(idx)
        if len(batch) >= 1:
            yield batch

    def __getitem__(self, idx):
        rng = random.Random(int(idx))
        tokens_a = self.data[0][idx].tolist()
        tokens_b = self.data[1][idx].tolist()
        type = tokens_a[0]
        tokens_a = tokens_a[1:]
        tokens, token_type, mask, mask_labels = self.create_masked_lm_predictions(tokens_a, tokens_b, type,
                                                                                  self.mask_lm_prob,
                                                                                  self.max_preds_per_seq,
                                                                                  self.vocab_words, rng)
        assert len(tokens) == len(token_type)
        return {"id": int(idx),
                "tokens": tokens,
                "tokens_type": token_type,
                "lm_labels": mask_labels,
                "ntokens": sum(mask),
                "data-type": type,
                "model-type": "MLM"
                }

    def create_masked_lm_predictions(self, a, b, type, mask_lm_prob, max_preds_per_seq, vocab_words, rng):
        token_sep = [self.tokenizer.sep()]
        token_cls = [self.tokenizer.cls()]
        tokens = token_cls + a + token_sep + b + token_sep
        type_a = [0] * len(a)
        if type == 0 or type == 1:
            type_b = [1] * len(b)
        else:
            type_b = [0] * len(b)
        token_type = [type_a[0]] + type_a + [type_a[0]] + type_b + [type_b[0]]
        assert len(token_type) == len(tokens)
        cand_indices = [idx + 1 for idx in range(len(a))] + [idx + 2 + len(a) for idx in range(len(b))]
        rng.shuffle(cand_indices)
        num_to_predict = min(max_preds_per_seq, max(1, int(round(len(tokens) * mask_lm_prob))))
        mask = [0] * len(tokens)
        mask_labels = [-1] * len(tokens)
        for idx in cand_indices[:num_to_predict]:
            mask[idx] = 1
            label = self.mask_token(idx, tokens, vocab_words, rng)
            mask_labels[idx] = label
        return tokens, token_type, mask, mask_labels

    def mask_token(self, idx, tokens, vocab_words, rng):
        label = tokens[idx]
        if rng.random() < 0.8:
            new_label = self.tokenizer.mask()
        else:
            if rng.random() < 0.5:
                new_label = label
            else:
                new_label = rng.choice(vocab_words)
        tokens[idx] = new_label
        return label

    def collate(self, samples):
        padding = self.tokenizer.pad()
        tensor_size = max([len(candidate["tokens"]) for candidate in samples])
        batch = len(samples)

        input_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        token_type_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        attention_mask = torch.LongTensor(batch, tensor_size).fill_(0)
        lm_labels = torch.LongTensor(batch, tensor_size).fill_(-1)
        nsp_labels = torch.LongTensor(batch).fill_(-1)

        id_ = torch.LongTensor(batch).fill_(padding)
        ntokens = 0
        for i, candidates in enumerate(samples):
            assert candidates["model-type"].lower() == "mlm"
            tokens = candidates["tokens"]
            types = candidates["tokens_type"]
            lm_tokens = candidates["lm_labels"]
            lens = len(tokens)

            input_ids[i, :lens] = torch.Tensor(tokens)
            token_type_ids[i, :lens] = torch.Tensor(types)
            lm_labels[i, :lens] = torch.Tensor(lm_tokens)
            attention_mask[i, :lens] = 1
            nsp_labels[i] = candidates["data-type"]

            ntokens += candidates['ntokens']
            id_[i] = candidates["id"]
        batch = {
            'id': id_,
            'nsentences': batch,
            "ntokens": ntokens,
            'net_input': {
                'input_tokens': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'clm': False
            },
            'target': lm_labels,
            'nsp_labels': nsp_labels
        }

        return batch

    def get_dummy_batch(self):

        bsz = 2
        src_len = 20
        return self.collate([
            {
                'id': i,
                'Q': torch.Tensor(src_len).uniform_(2, 20).long().tolist(),
                'A': torch.Tensor(src_len).uniform_(2, 20).long().tolist(),
                'style': 0
            }
            for i in range(bsz)
        ])


class GPTDataset(data.Dataset):

    def __init__(self, path, tokenizer, batch_size, max_tokens, world_size=1, max_lens=510, seed=512, no_cache=False):
        self.a_size = None
        self.b_size = None
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.world_size = world_size
        self.max_lens = max_lens
        self.seed = seed
        self.no_cache = no_cache
        self.data = self.read_data(path)
        self.padding = tokenizer.pad()
        self.frozen_indices = self.init_data_indices()
        self.frozen_batch = self.fill_batch(list(self.batch_by_size()))

    @property
    def supports_prefetch(self):
        if self.data[0].supports_prefetch and self.data[1].supports_prefetch:
            return True
        else:
            return False

    def prefetch(self, indices):
        if self.supports_prefetch:
            self.data[0].prefetch(indices)
            self.data[1].prefetch(indices)
            print("| Fetch all data into memory")

    def fill_batch(self, batches):
        b_size = len(batches)
        rng = random.Random(self.seed)
        rng.shuffle(batches)
        fill_num = self.world_size - b_size % self.world_size
        if fill_num == self.world_size:
            return batches
        batches.extend(batches[-fill_num:])
        assert len(batches) % self.world_size == 0
        return batches

    def index_len(self):
        return len(self.a_size)

    def read_data(self, path):
        A_path = path + "-A"
        B_path = path + "-B"
        if self.no_cache:
            A_data = indexed_dataset.IndexedDataset(A_path, fix_lua_indexing=True)
            B_data = indexed_dataset.IndexedDataset(B_path, fix_lua_indexing=True)
        else:
            A_data = indexed_dataset.IndexedCachedDataset(A_path, fix_lua_indexing=True)
            B_data = indexed_dataset.IndexedCachedDataset(B_path, fix_lua_indexing=True)
        self.a_size = A_data.sizes
        self.b_size = B_data.sizes
        return (A_data, B_data)

    def init_data_indices(self):
        return self.ordered_indices()

    def filter_by_size(self, indices):
        ignore = []
        for ind in indices:
            size = self.num_tokens(ind)
            if size > self.max_lens:
                ignore.append(ind)
            else:
                yield ind
        if len(ignore) > 0:
            print(
                "WARNING: {} samples have invalid sizes and will be skipped, max-qa-len={}, first few sample ids={}".format(
                    len(ignore), self.max_lens, ignore[:10]))

    def __len__(self):
        return len(self.frozen_indices)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        with numpy_seed(self.seed):
            indices = np.random.permutation(len(self.data[0]))
        lens = []
        for inx_a in indices:
            lens.append(self.num_tokens(inx_a))
        indices = indices[np.argsort(np.array(lens), kind='mergesort')]
        indices = list(self.filter_by_size(indices))
        return indices

    def num_tokens(self, idx):
        return self.a_size[idx] + self.b_size[idx]

    def batch_by_size(self):
        max_tokens = self.max_tokens
        indices = self.frozen_indices
        batch = []
        tokens = 0

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if num_tokens > max_tokens:
                return True
            # if len(batch) == self.batch_size:
            #     return True
            return False

        for idx in indices:
            idx_max_token = self.num_tokens(idx)
            tokens = max(tokens, idx_max_token)
            if is_batch_full(tokens * (len(batch) + 1)):
                yield batch
                batch = []
                tokens = 0
            batch.append(idx)
        if len(batch) >= 1:
            yield batch

    def __getitem__(self, idx):
        tokens_a = self.data[0][idx].tolist()
        tokens_b = self.data[1][idx].tolist()
        token_sep = [self.tokenizer.sep()]
        token_cls = [self.tokenizer.cls()]
        tokens = token_cls + tokens_a + token_sep + tokens_b
        lm_labels = tokens_a + token_sep + tokens_b + token_sep
        assert len(tokens) == len(lm_labels)
        return {"id": int(idx),
                "tokens": tokens,
                "lm_labels": lm_labels,
                "model-type": "CLM"
                }

    def collate(self, samples):
        padding = self.tokenizer.pad()
        tensor_size = max([len(candidate["tokens"]) for candidate in samples])
        batch = len(samples)

        input_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        attention_mask = torch.LongTensor(batch, tensor_size).fill_(0)
        lm_labels = torch.LongTensor(batch, tensor_size).fill_(-1)
        nsp_labels = torch.LongTensor(batch).fill_(-1)

        id_ = torch.LongTensor(batch).fill_(padding)
        ntokens = 0
        for i, candidates in enumerate(samples):
            assert candidates["model-type"].lower() == "clm"
            tokens = candidates["tokens"]
            lm_tokens = candidates["lm_labels"]
            lens = len(tokens)

            input_ids[i, :lens] = torch.Tensor(tokens)
            lm_labels[i, :lens] = torch.Tensor(lm_tokens)
            attention_mask[i, :lens] = 1

            ntokens += lens
            id_[i] = candidates["id"]
            id_[i] = candidates["id"]
        batch = {
            'id': id_,
            'nsentences': batch,
            "ntokens": ntokens,
            'net_input': {
                'input_tokens': input_ids,
                'attention_mask': attention_mask,
                'clm': True
            },
            'target': lm_labels,
            'nsp_labels': nsp_labels
        }

        return batch


class FuseDataset(data.Dataset):

    def __init__(self, dataload_a, dataload_b, world_size=1, seed=1):
        self.dataload_a = dataload_a
        self.dataload_b = dataload_b
        self.world_size = world_size
        self.lens_a = dataload_a.index_len()
        self.init_seed = seed + 256
        self.fuse_lens = len(dataload_a) + len(dataload_b)
        self.frozen_batch = self.init_batch()

    @property
    def supports_prefetch(self):
        if self.dataload_a.supports_prefetch and self.dataload_b.supports_prefetch:
            return True
        else:
            return False

    def prefetch(self, indices):
        if not self.dataload_a.supports_prefetch or not self.dataload_b.supports_prefetch:
            return
        a_indices = []
        b_indices = []
        for idx in indices:
            if idx >= self.lens_a:
                idx -= self.lens_a
                b_indices.append(idx)
            else:
                a_indices.append(idx)
        self.dataload_a.prefetch(a_indices)
        self.dataload_b.prefetch(b_indices)

    def init_batch(self):
        rng = random.Random(self.init_seed)
        batch_a = self.dataload_a.frozen_batch
        batch_b = self.dataload_b.frozen_batch
        rng.shuffle(batch_a)
        rng.shuffle(batch_b)
        lens_a = self.lens_a
        fake_b = []
        for b in batch_b:
            fake_b.append([i + lens_a for i in b])
        batch_a.extend(fake_b)
        return self.shuffle_by_size(batch_a, rng)

    def shuffle_by_size(self, batches, rng):
        assert len(batches) % self.world_size == 0
        rand_size = len(batches) // self.world_size
        indices = list(range(rand_size))
        rng.shuffle(indices)
        new_batches = []
        for idx in indices:
            new_batches.extend(batches[idx * self.world_size:(idx + 1) * self.world_size])
        assert len(new_batches) == len(batches)
        return new_batches

    @classmethod
    def load_dataset(self, tokenizer, args):
        train_path = args.data
        train_prefix = args.train_prefix
        valid_prefix = args.valid_prefix
        max_lens = args.max_lens
        valid_data = []
        for valid_file in valid_prefix.split(","):
            path = os.path.join(train_path, valid_file)
            if "MLM" in valid_file:
                valid_data.append(
                    BertPairDataset(path, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size,
                                    max_lens=max_lens, mask_lm_prob=args.mask_lm_prob,
                                    max_preds_per_seq=args.max_preds_per_seq, no_cache=args.no_cache))
            else:
                valid_data.append(
                    GPTDataset(path, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size,
                               max_lens=max_lens, no_cache=args.no_cache))
        train_mlm = os.path.join(train_path, train_prefix + "MLM")
        train_clm = os.path.join(train_path, train_prefix + "CLM")
        mlm_data = BertPairDataset(train_mlm, tokenizer, args.train_batch, args.max_tokens, world_size=args.world_size,
                                   max_lens=max_lens, mask_lm_prob=args.mask_lm_prob,
                                   max_preds_per_seq=args.max_preds_per_seq, no_cache=args.no_cache)
        clm_data = GPTDataset(train_clm, tokenizer, args.train_batch, args.max_tokens, world_size=args.world_size,
                              max_lens=max_lens, no_cache=args.no_cache)
        train_loader = FuseDataset(mlm_data, clm_data)
        return train_loader, valid_data

    @classmethod
    def load_dataset_no_nsp(self, tokenizer, args):
        # pdb.set_trace()
        print("| Load dataset with no nsp loss")
        use_task_embed = args.use_task_embed
        if use_task_embed:
            print("| Training with task embedding")
        else:
            print("| Training with no task embedding")
        train_path = args.data
        train_prefix = args.train_prefix
        valid_prefix = args.valid_prefix
        max_lens = args.max_lens
        valid_data = []
        for valid_file in valid_prefix.split(","):
            path = os.path.join(train_path, valid_file)
            if "MLM" in valid_file:
                valid_data.append(
                    MLMDataset(path, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size,
                               max_lens=max_lens, mask_lm_prob=args.mask_lm_prob,
                               max_preds_per_seq=args.max_preds_per_seq, no_cache=args.no_cache,
                               drop_first_token=args.drop_first_token, use_task_embed=args.use_task_embed))
            else:
                if use_task_embed:
                    valid_data.append(
                        CLMTaskDataset(path, tokenizer, args.train_batch, args.max_tokens, world_size=args.world_size,
                                       max_lens=args.max_lens, no_cache=args.no_cache))
                else:
                    valid_data.append(
                        CLMDataset(path, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size,
                                   max_lens=max_lens, no_cache=args.no_cache, drop_first_token=args.drop_first_token))
        train_mlm = os.path.join(train_path, train_prefix + "MLM")
        train_clm = os.path.join(train_path, train_prefix + "CLM")
        mlm_data = MLMDataset(train_mlm, tokenizer, args.train_batch, args.max_tokens, world_size=args.world_size,
                              max_lens=max_lens, mask_lm_prob=args.mask_lm_prob,
                              max_preds_per_seq=args.max_preds_per_seq, no_cache=args.no_cache,
                              drop_first_token=args.drop_first_token, use_task_embed=args.use_task_embed)
        if use_task_embed:
            clm_data = CLMTaskDataset(train_clm, tokenizer, args.train_batch, args.max_tokens,
                                      world_size=args.world_size,
                                      max_lens=args.max_lens, no_cache=args.no_cache)
        else:
            clm_data = CLMDataset(train_clm, tokenizer, args.valid_batch, args.max_tokens, world_size=args.world_size,
                                  max_lens=max_lens, no_cache=args.no_cache, drop_first_token=args.drop_first_token)
        train_loader = FuseDataset(mlm_data, clm_data)
        return train_loader, valid_data

    def __len__(self):
        return len(self.frozen_batch)

    def __getitem__(self, idx):
        if idx >= self.lens_a:
            idx -= self.lens_a
            return self.dataload_b[idx]
        else:
            return self.dataload_a[idx]

    def collate(self, samples):
        data_type = samples[0]["model-type"]
        if data_type.lower() == "mlm":
            return self.dataload_a.collate(samples)
        else:
            return self.dataload_b.collate(samples)


class FuseSampler(object):

    def __init__(self, dataset, num_shards, shard_id):
        if num_shards > 1 and shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')
        if num_shards == 1:
            shard_id = 0
        batches = dataset.frozen_batch
        self._sharded_len = len(batches) // num_shards
        # iterable = batches[:self._sharded_len*num_shards]
        iterable = batches[shard_id::num_shards]
        indices = []
        for batch in iterable:
            indices.extend(batch)
        dataset.prefetch(indices)

        self.itr = [item[1] for item in list(itertools.zip_longest(
            range(self._sharded_len),
            iterable,
        ))]

    def __len__(self):
        return self._sharded_len

    def __iter__(self):

        return iter(self.itr)

    def __next__(self):

        return next(self.itr)[1]
