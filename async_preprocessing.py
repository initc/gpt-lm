from multiprocessing import Pool
import torch.utils
import torch.utils.data
from data_utils import indexed_dataset
import torch
import os
import re
import pdb

from data_utils.tokenization import BertWordPieceTokenizer

key_word = {
    "…":"...",
    "—":"-",
    "“":"\"",
    "”":"\"",
    "‘":"'",
    "’":"'"
}


SPECIAL_SIGNAL = "./';,\(\)\"\"'~`''“”《》<>"


def cut_sentence(paragraph):
    paragraph = paragraph.replace("  ", "")
    sentences = re.split('(。|！|\!|？|\?)',paragraph)         # 保留分割符
    if len(sentences) == 1:
        return [sentences[0]]
    new_sents = []
    for i in range(int(len(sentences)/2)):
        sent = sentences[2*i] + sentences[2*i+1]
        if len(new_sents) != 0 and (sent[0] in SPECIAL_SIGNAL or len(new_sents[-1]) < 20):
            new_sents[-1] += sent
        else:
            new_sents.append(sent)
    sent = sentences[-1]
    if len(sentences) % 2 == 1 and len(sent) > 0:
        if len(new_sents) != 0 and (sent[0] in SPECIAL_SIGNAL or len(new_sents[-1]) < 20):
            new_sents[-1] += sent
        else:
            new_sents.append(sent)
    return new_sents


def replace_text(text):
    for key,value in key_word.items():
        text = re.sub(key, value, text)
    return text


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def read_split(
        filename, tokenizer, worker_id, num_workers, type_doc, min_lens=10
    ):
    with open(filename, 'r') as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = offset + chunk_size
        f.seek(offset)
        if offset > 0:
            safe_readline(f)  # drop first incomplete line
        result = []
        line = f.readline()
        while line:
            line = replace_text(line)
            ids = tokenizer.convert_text_to_ids(line)
            ids = ids[:509]
            if len(ids) >= min_lens:
                    ids = [type_doc]+ids
                    result.append(ids)
            if f.tell() > end:
                break
            line = f.readline()
    return result


def main_multi_task(args):
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # parser.add_argument("--tokenizer", type=str, help="where to load vocabulary")
    parser.add_argument("--data", type=str)
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--prefix", type=str, default="train")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(args)


    tokenizer = BertWordPieceTokenizer("bert-base-chinese", cache_dir="temp_cache_dir")
    
    data_bin = os.path.join(args.out, "{}-CLM.bin".format(args.prefix))
    data_idx = os.path.join(args.out, "{}-CLM.idx".format(args.prefix))
    data_ds = indexed_dataset.IndexedDatasetBuilder(data_bin)

    def comsume(worker_result):
        for ids in worker_result:
            data_ds.add_item(torch.IntTensor(ids)
            )
    pool = Pool(processes=args.workers)
    worker_result = []

    for i in range(args.workers):
        w = pool.apply_async(
                    read_split,
                    (
                        args.data,
                        tokenizer,
                        i,
                        args.workers,
                        0,
                        10
                    ),
                    callback=comsume
                )
        worker_result.append(w)
    pool.close()
    pool.join()

    data_ds.finalize(data_idx)
    print("| write data into {}".format(args.outs))


if __name__ == "__main__":
    import sys
    main_multi_task(sys.argv[1:])
