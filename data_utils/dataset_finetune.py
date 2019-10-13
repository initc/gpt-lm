import os
import torch
import random
from data_utils.utils import numpy_seed
import itertools

from torch.utils import data
import numpy as np

from data_utils import indexed_dataset


class QAClmDataset(data.Dataset):

    def __init__(self, path, tokenizer, batch_size, max_tokens, world_size=1, max_lens=510, seed=512, no_cache=False, use_token_type=False, use_task_embedding=False):
        self.q_size = None
        self.a_size = None
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.world_size = world_size
        self.max_lens = max_lens
        self.seed = seed
        self.no_cache = no_cache
        self.use_token_type = use_token_type
        self.use_task_embedding = use_task_embedding
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
        return len(self.q_size)

    def read_data(self, path):
        q_path = path + "-Q"
        a_path = path + "-A"
        if self.no_cache:
            q_data = indexed_dataset.IndexedDataset(q_path, fix_lua_indexing=True)
            a_data = indexed_dataset.IndexedDataset(a_path, fix_lua_indexing=True)
        else:
            q_data = indexed_dataset.IndexedCachedDataset(q_path, fix_lua_indexing=True)
            a_data = indexed_dataset.IndexedCachedDataset(a_path, fix_lua_indexing=True)
        self.q_size = q_data.sizes
        self.a_size = a_data.sizes
        return q_data, a_data

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
        return self.q_size[idx] + self.a_size[idx]

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
            if len(batch) == self.batch_size:
                return True
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
        tokens_q = self.data[0][idx].tolist()
        tokens_a = self.data[1][idx].tolist()
        begin_token = [self.tokenizer.style(tokens_q[0])]
        tokens_q = tokens_q[1:]
        token_sep = [self.tokenizer.sep()]
        token_cls = [self.tokenizer.cls()]

        tokens = token_cls + tokens_q + begin_token + tokens_a
        lm_labels = tokens_q + begin_token + tokens_a + token_sep
        token_type = None
        if self.use_token_type:
            token_type = [0]*(len(tokens_q)+1) + [1]*(len(tokens_a)+1)
        assert len(tokens) == len(lm_labels)
        return {"id": int(idx),
                "tokens": tokens,
                "lm_labels": lm_labels,
                "model-type": "CLM",
                "token_type": token_type
                }

    def collate(self, samples):
        padding = self.tokenizer.pad()
        tensor_size = max([len(candidate["tokens"]) for candidate in samples])
        batch = len(samples)

        input_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        attention_mask = torch.LongTensor(batch, tensor_size).fill_(0)
        lm_labels = torch.LongTensor(batch, tensor_size).fill_(-1)
        token_type_ids = None
        if self.use_token_type:
            token_type_ids = torch.LongTensor(batch, tensor_size).fill_(padding)
        task_type_ids = None
        if self.use_task_embedding:
            task_type_ids = torch.LongTensor(batch, tensor_size).fill_(0)
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
            if self.use_token_type:
                token_type_ids[i, :lens] = torch.Tensor(candidates["token_type"])
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
                'token_type_ids': token_type_ids,
                'task_type_ids': task_type_ids,
                'clm': True
            },
            'target': lm_labels,
            'nsp_labels': nsp_labels
        }

        return batch
