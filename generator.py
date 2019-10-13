# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from model import BertModel
from data_utils.tokenization import BertWordPieceTokenizer

from torch.serialization import default_restore_location

from data_utils.utils import move_to_cuda
import pdb

def read_file(path):
    with open(path, encoding="utf-8") as f:
        for l in f:
            if not l.strip():
                continue
            yield l.strip()


def sample_sequence(model, tokenizer, length, context=None, temperature=1, top_k=0, device='cuda', sample=False):

    input_tokens = context["input_tokens"]
    output = input_tokens
    past = None
    eos_id = tokenizer.sep()
    with torch.no_grad():
        model.eval()
        for i in range(length):
            logits, _, past = model(input_tokens=input_tokens, clm=True, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            # if sample:
            #     prev = torch.multinomial(log_probs, num_samples=1)
            # else:
            #     _, prev = torch.topk(log_probs, k=1, dim=-1)
            _, prev = torch.topk(log_probs, k=top_k, dim=-1)
            next_id = prev[0][2].item()
            pdb.set_trace()
            if next_id == eos_id:
                break
            input_tokens = prev
            output = torch.cat((output, prev), dim=1)
    return output[0, 1:].tolist()


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)


def convert_content(tokenizer, text):
    input_ids = torch.LongTensor([[tokenizer.cls(
    )]+tokenizer.convert_text_to_ids(text)])
    return {
        'input_tokens': input_ids
    }


def convert_model(state_dict):
    new_dict = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        new_dict[key] = value
    return new_dict


def generate(model, tokenizer, device, data_text, sample=True, top_k=5, beam_size=6, outlens=30):
    # device = model.device
    result = []
    with torch.no_grad():
        model.eval()
        for text in read_file(data_text):
            context = convert_content(tokenizer, text=text)
            context = move_to_cuda(context, device)
            out = sample_sequence(model, tokenizer, outlens, context=context, temperature=1, top_k=top_k, device=device, sample=True)
            out = tokenizer.convert_ids_to_text(out)
            out = out.replace("##", "")
            result.append(out)
    print(result)


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-config", type=str, default="openai-gpt",
                        help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--outlens", type=int, default=30)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--checkpoints", type=str)
    parser.add_argument("--data", type=str, default="file")

    args = parser.parse_args()
    args.load_model = True

    model = BertModel(None, args)
    state_dict = convert_model(torch.load(args.checkpoints)['sd'])
    model.load_state_dict(state_dict)
    model.to(args.device)
    tokenizer = BertWordPieceTokenizer("bert-base-chinese", cache_dir="temp_cache_dir")
    generate(model, tokenizer, args.device, args.data, sample=True, top_k=5, beam_size=6, outlens=30)

if __name__ == "__main__":
    main()











