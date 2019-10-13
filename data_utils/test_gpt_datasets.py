import os
import torch
import random
from data_utils.utils import numpy_seed

from torch.utils import data
import numpy as np

from data_utils import indexed_dataset


class CLMTaskDataset(data.Dataset):

    def __init__(self, path, tokenizer, batch_size, max_tokens, world_size=1, max_lens=510, seed=512, no_cache=False, ):
        self.sizes = None
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.world_size = world_size
        self.max_lens = max_lens
        self.seed = seed + 20
        self.no_cache = no_cache
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