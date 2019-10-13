from werkzeug.serving import run_simple
from flask import Flask, request
import json
import time

from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn.functional as F

from model import BertModel
from data_utils.tokenization import BertWordPieceTokenizer

from model.gpt_beam_search import SequenceGenerator
from utils import move_to_cuda
import pdb


def sample_sequence(model, tokenizer, length, context=None, temperature=1, temperature_lens=0, top_k=0, top_k_lens=200, top_p=0, punishRate=0.8, device='cuda', sample=False, eos=None):
    # pdb.set_trace()
    input_tokens = context["input_tokens"]
    task_type_ids = context["task_type_ids"] if "task_type_ids" in context and context["task_type_ids"] is not None else None
    token_type_ids = context["token_type_ids"] if "token_type_ids" in context and context["token_type_ids"] is not None else None
    task_type = task_type_ids[0, -1].item() if task_type_ids is not None else None
    token_type = token_type_ids[0, -1].item() if token_type_ids is not None else None
    output = input_tokens
    past = None
    eos_id = tokenizer.sep() if eos is None else eos
    unk = tokenizer.unk()
    punishDup = torch.ones(len(tokenizer)).to(input_tokens.device)
    with torch.no_grad():
        model.eval()
        cur_temperature = temperature[0]
        cur_temperature_lens = temperature_lens[0]
        cur_index = 0

        top_k_index = 0
        cur_top_k = top_k[0]
        cur_top_k_lens = top_k_lens[0]
        for i in range(length):
            # pdb.set_trace()
            logits, _, past = model(input_tokens=input_tokens, token_type_ids=token_type_ids, task_type_ids=task_type_ids, clm=True, past=past)
            if i >= cur_temperature_lens and cur_index < len(temperature_lens)-1:
                cur_index += 1
                cur_temperature = temperature[cur_index]
                cur_temperature_lens = temperature_lens[cur_index]

            if i < cur_temperature_lens:
                logits = logits[:, -1, :] / cur_temperature
            else:
                logits = logits[:, -1, :]
            logits[:, unk] = -float("inf")
            if i >= cur_top_k_lens and top_k_index < len(top_k)-1:
                top_k_index += 1
                cur_top_k = top_k[top_k_index]
                cur_top_k_lens = top_k_lens[top_k_index]
            if i < cur_top_k_lens:
                logits = top_k_logits(logits, k=cur_top_k)
            else:
                logits = top_k_logits(logits, k=1)
            if top_p > 0:
                logits = top_p_logits(logits, top_p)
            log_probs = F.softmax(logits, dim=-1)
            # punish duplicate
            log_probs *= punishDup
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            next_id = prev[0][0].item()
            if next_id == eos_id:
                break
            punishDup[next_id] *= punishRate
            input_tokens = prev
            task_type_ids = torch.LongTensor([task_type]).type_as(input_tokens).expand_as(input_tokens) if task_type is not None else None
            token_type_ids = torch.LongTensor([token_type]).type_as(input_tokens).expand_as(
                input_tokens) if token_type is not None else None
            output = torch.cat((output, prev), dim=1)
    return output[0, 1:].tolist()


def lm_bert(model, context, tokenizer):
    logits, _, past = model(input_tokens=context["input_tokens"])
    mask = tokenizer.mask()
    out = []
    for ins, outs in zip(context["input_tokens"][0].tolist(), logits[0]):
        if ins != mask:
            out.append(ins)
        else:
            outs = F.softmax(outs, dim=-1)
            _, prev = torch.topk(outs, k=1, dim=-1)
            out.append(prev[0].item())
    return out


def top_p_logits(logits, top_p, threshold=-float('Inf'), filter_value=-float("inf")):
    assert top_p > 0
    assert logits.size(0) == 1
    # Compute cumulative probabilities of sorted tokens
    sorted_logits, sorted_indices = torch.sort(logits[0], descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Back to unsorted indices and set them to -infinity
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[0][indices_to_remove] = filter_value

    indices_to_remove = logits[0] < threshold
    logits[0][indices_to_remove] = filter_value
    return logits

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
    eos_id = tokenizer.sep()
    if "[unused10]" in text:
        eos_id = tokenizer.sep1()
        text = text.split("[unused10]")
        ids = []
        for t in text:
            ids.extend(tokenizer.convert_text_to_ids(t)+[eos_id])
        if ids[-2]==eos_id:
            ids.pop()
        ids = [tokenizer.cls()]+ids
    else:
        ids = [tokenizer.cls()] + tokenizer.convert_text_to_ids(text)
    input_ids = torch.LongTensor([ids])

    return {
        'input_tokens': input_ids
    }, eos_id


def convert_task_content(tokenizer, text, type):
    type = int(type)
    ids = [tokenizer.cls()] + tokenizer.convert_text_to_ids(text)
    input_ids = torch.LongTensor([ids])
    task_type_ids = torch.LongTensor([type]).expand_as(input_ids)

    return {
        'input_tokens': input_ids,
        'task_type_ids': task_type_ids
    }


def convert_qa_content(tokenizer, text, type):
    type = int(type)
    begin = [tokenizer.style(type)]
    ids = [tokenizer.cls()] + tokenizer.convert_text_to_ids(text)+begin
    # print(ids)
    input_ids = torch.LongTensor([ids])
    task_type_ids = torch.LongTensor([0]).expand_as(input_ids)
    token_type_ids = torch.LongTensor([0]*(input_ids.size(1)-1)+[1]).expand_as(input_ids)

    return {
        'input_tokens': input_ids,
        'task_type_ids': task_type_ids,
        'token_type_ids': token_type_ids
    }


def convert_model(state_dict):
    new_dict = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = "model." + key
        new_dict[key] = value
    return new_dict


import logging
logger = logging.getLogger()

app = Flask(__name__)


def model_init(app):
    ArgsSet = type('ArgsSet',(object,),{})
    client = ArgsSet()
    parser = ArgumentParser()
    parser.add_argument("--model-config", type=str, default="openai-gpt",
                        help="Path, url or short name of the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                        else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--outlens", type=int, default=30)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--fuse-checkpoints", type=str)
    parser.add_argument("--gpt-checkpoints", type=str)
    parser.add_argument("--qa-style-checkpoints", type=str)
    parser.add_argument("--multi-task", type=str)
    parser.add_argument("--split-sentence-with-task-embedding-checkpoints", type=str)
    parser.add_argument("--special-cls-checkpoints", type=str)

    parser.add_argument("--port", type=int, default=8866)

    args = parser.parse_args()
    args.load_model = True
    args.fp32_embedding = False
    args.fp32_layernorm = False
    args.fp32_tokentypes = False
    args.layernorm_epsilon = 1e-12

    fuse_model = BertModel(None, args)
    state_dict = convert_model(torch.load(args.fuse_checkpoints)['sd'])
    fuse_model.load_state_dict(state_dict)
    fuse_model.to(args.device)
    fuse_model.eval()
    print("| Load model from {}".format(args.fuse_checkpoints))

    gpt = BertModel(None, args)
    state_dict = convert_model(torch.load(args.gpt_checkpoints)['sd'])
    gpt.load_state_dict(state_dict)
    gpt.to(args.device)
    gpt.eval()
    tokenizer = BertWordPieceTokenizer("bert-base-chinese", cache_dir="temp_cache_dir")
    print(" Load model from {}".format(args.gpt_checkpoints))

    # Load bert checkpoints
    args.load_model = False
    args.fp32_embedding = False
    args.fp32_layernorm = False
    args.fp32_tokentypes = False
    args.layernorm_epsilon = 1e-12
    bert = BertModel(None, args)
    bert.to(args.device)
    bert.eval()

    client.tokenizer = tokenizer
    client.fuse_model = fuse_model
    client.fuse_beam = SequenceGenerator(fuse_model, tokenizer, beam_size=args.beam, max_lens=args.outlens)
    client.gpt =gpt
    client.gpt_beam = SequenceGenerator(gpt, tokenizer, beam_size=args.beam, max_lens=args.outlens)
    client.bert = bert
    client.device = args.device
    client.port = args.port
    client.generator = sample_sequence

    # multi task model

    multi_task = BertModel(None, args)
    state_dict = convert_model(torch.load(args.multi_task)['sd'])
    print("| Load model from {}".format(args.multi_task))
    multi_task.load_state_dict(state_dict)
    multi_task.to(args.device)
    multi_task.eval()
    client.multi_task_model = multi_task
    client.multi_task_beam = SequenceGenerator(multi_task, tokenizer, beam_size=args.beam, max_lens=args.outlens)

    # qa style model
    qa_style = BertModel(None, args)
    state_dict = convert_model(torch.load(args.qa_style_checkpoints)['sd'])
    qa_style.load_state_dict(state_dict)
    qa_style.to(args.device)
    qa_style.eval()
    print(" Load model from {}".format(args.qa_style_checkpoints))
    client.qa_task_model = qa_style

    # special cls tokens
    special_cls_model = BertModel(None, args)
    special_cls_model.eval()
    state_dict = convert_model(torch.load(args.special_cls_checkpoints)['sd'])
    special_cls_model.load_state_dict(state_dict)
    special_cls_model.to(args.device)
    special_cls_model.eval()
    print(" Load model from {}".format(args.special_cls_checkpoints))
    client.special_cls_model = special_cls_model
    client.special_beam = SequenceGenerator(special_cls_model, tokenizer, beam_size=args.beam, max_lens=args.outlens)

    # split sentence model with task embedding
    split_sentence_model = BertModel(None, args)
    split_sentence_model.eval()
    state_dict = convert_model(torch.load(args.split_sentence_with_task_embedding_checkpoints)['sd'])
    split_sentence_model.load_state_dict(state_dict)
    split_sentence_model.to(args.device)
    split_sentence_model.eval()
    print(" Load model from {}".format(args.split_sentence_with_task_embedding_checkpoints))
    client.split_sentence_model = split_sentence_model
    client.split_sentence_beam = SequenceGenerator(split_sentence_model, tokenizer, beam_size=args.beam, max_lens=args.outlens)

    return client

client = model_init(app)


@app.route('/lm', methods=['POST'])
def score():
    content = request.get_json(silent=True, force=True)
    if "data" in content:
        length = content.get("length", 30)
        eos_id = client.tokenizer.sep()
        is_beam = content.get("beam", -1)
        repeat = content.get("repeat", 1)
        model_type = content.get("model", "gpt")
        response = OrderedDict()
        if model_type == "gpt":
            model = client.gpt
        else:
            model = client.fuse_model
        begin_time = time.time()
        logger.error("Score message...")
        with torch.no_grad():
            text = content["data"]
            response["user-query"] = text
            context, eos_id = convert_content(client.tokenizer, text=text)
            ids_length = context["input_tokens"].size(1)
            context = move_to_cuda(context, client.device)
            reply = []
            for i in range(repeat):
                if model_type != "bert":
                    out = client.generator(model, client.tokenizer, length, context=context, temperature=1, top_k=5, device=client.device, sample=True, eos=eos_id)
                else:
                    out = lm_bert(client.bert, context, client.tokenizer)
                    # content["ids"] = context["input_tokens"][0].tolist()
                out = out[ids_length-1:]
                out = client.tokenizer.convert_ids_to_text(out)
                out = out.replace("##", "")
                reply.append(out)
            if len(reply) == 1:
                reply = reply[0]

            beam_out = None
            if is_beam != -1:
                    if model_type == "gpt":
                        client.gpt_beam.beam_size = is_beam
                        client.gpt_beam.max_lens = length
                        client.gpt_beam.eos = eos_id
                        beam_model = client.gpt_beam
                    else:
                        client.fuse_beam.beam_size = is_beam
                        client.fuse_beam.max_lens = length
                        client.fuse_beam.eos = eos_id
                        beam_model = client.fuse_beam
                    beam_out = beam_model.generate_response(context)
            response["sampling-response"] = reply
            if beam_out is not None:
                response["beam-response"] = client.tokenizer.convert_ids_to_text(beam_out).replace("##", "")
        interval = time.time() - begin_time
        logger.error("elapsed time = %s", interval)
        response["interval"] = interval
    return json.dumps(response, ensure_ascii=False)


@app.route('/qa', methods=['POST'])
def multi_task():
    content = request.get_json(silent=True, force=True)
    if "data" in content:
        length = content.get("length", 30)
        is_beam = content.get("beam", -1)
        task_type = content.get("type", 0)
        if task_type not in [0, 1, 2]:
            task_type = 0
        repeat = content.get("repeat", 1)
        response = OrderedDict()
        begin_time = time.time()
        logger.error("user message...")
        text = content["data"]
        logger.error(text)
        with torch.no_grad():
            response["user-query"] = text
            context = convert_task_content(client.tokenizer, text, task_type)
            ids_length = context["input_tokens"].size(1)
            context = move_to_cuda(context, client.device)
            reply = []
            for i in range(repeat):
                out = client.generator(client.multi_task_model, client.tokenizer, length, context=context, temperature=1, top_k=5, device=client.device, sample=True)
                out = out[ids_length-1:]
                out = client.tokenizer.convert_ids_to_text(out)
                out = out.replace("##", "")
                reply.append(out)
            if len(reply) == 1:
                reply = reply[0]

            beam_out = None
            if is_beam != -1:
                    client.multi_task_beam.beam_size = is_beam
                    client.multi_task_beam.max_lens = length
                    beam_out = client.multi_task_beam.generate_response(context)
            response["sampling-response"] = reply
            if beam_out is not None:
                response["beam-response"] = client.tokenizer.convert_ids_to_text(beam_out).replace("##", "")
        interval = time.time() - begin_time
        logger.error("elapsed time = %s", interval)
        response["interval"] = interval
    return json.dumps(response, ensure_ascii=False)


@app.route('/style_qa', methods=['POST'])
def style_qa_task():
    content = request.get_json(silent=True, force=True)
    if "data" in content:
        length = content.get("length", 30)
        # is_beam = content.get("beam", -1)
        task_type = content.get("type", 0)
        if task_type not in [0, 1]:
            task_type = 0
        repeat = content.get("repeat", 1)
        response = OrderedDict()
        begin_time = time.time()
        logger.error("user message...")
        text = content["data"]
        logger.error(text)
        with torch.no_grad():
            response["user-query"] = text
            context = convert_qa_content(client.tokenizer, text, task_type)
            ids_length = context["input_tokens"].size(1)
            context = move_to_cuda(context, client.device)
            reply = []
            for i in range(repeat):
                out = client.generator(client.qa_task_model, client.tokenizer, length, context=context, temperature=1, top_k=5, device=client.device, sample=True)
                out = out[ids_length-1:]
                out = client.tokenizer.convert_ids_to_text(out)
                out = out.replace("##", "")
                reply.append(out)
            if len(reply) == 1:
                reply = reply[0]
            response["sampling-response"] = reply
        interval = time.time() - begin_time
        logger.error("elapsed time = %s", interval)
        response["interval"] = interval
    return json.dumps(response, ensure_ascii=False)


def convert_special_task(tokenizer, text, type, special_cls=False):
    type = int(type)
    token_cls = [tokenizer.cls_style(type)] if special_cls else [tokenizer.cls()]
    begin_ids = [tokenizer.sep1()]

    if "[unused10]" in text:
        eos_id = tokenizer.sep1()
        text = text.split("[unused10]")
        ids = []
        for t in text:
            ids.extend(tokenizer.convert_text_to_ids(t)+[eos_id])
        if ids[-2]==eos_id:
            ids.pop()
        ids = token_cls+ids
    else:
        ids = token_cls + tokenizer.convert_text_to_ids(text) + begin_ids

    input_ids = torch.LongTensor([ids])
    task_type_ids = torch.LongTensor([type]).expand_as(input_ids) if not special_cls else None

    return {
        'input_tokens': input_ids,
        'task_type_ids': task_type_ids
    }


@app.route('/split_lm', methods=['POST'])
def split_lm():
    content = request.get_json(silent=True, force=True)
    error_message = {}
    if "data" in content:
        length = content.get("length", 30)
        is_beam = content.get("beam", -1)
        beam_k = content.get("beam-k", 1)
        if beam_k > is_beam:
            beam_k = is_beam
        sample_k = content.get("top-k", [5])
        top_k_lens = content.get("top-k-lens", [200])
        top_p = content.get("top-p", 1)
        task_type = content.get("type", 0)
        punishRate = content.get("punishRate", 1)
        temperature = content.get("temperature", [1])
        temperature_lens = content.get("temperature-lens", [0])
        if not isinstance(sample_k, list) or not isinstance(top_k_lens, list):
            error_message["error"] = "top-k expect list type, not {}".format(type(sample_k))
            return json.dumps(error_message, ensure_ascii=False)
        elif len(sample_k) != len(top_k_lens):
            error_message["error"] = "top-k lens not equal to top-k-lens"
            return json.dumps(error_message, ensure_ascii=False)
        if not isinstance(temperature, list) or not isinstance(temperature_lens, list):
            error_message["error"] = "temperature expect list type, not {}".format(type(temperature))
            return json.dumps(error_message, ensure_ascii=False)
        elif len(temperature) != len(temperature_lens):
            error_message["error"] = "temperature lens not equal to temperature_lens"
            return json.dumps(error_message, ensure_ascii=False)
        special_cls = content.get("special-cls", False)
        # pdb.set_trace()
        if task_type not in [0, 1, 2]:
            task_type = 0
        repeat = content.get("repeat", 1)
        response = OrderedDict()
        begin_time = time.time()
        logger.error("user message...")
        text = content["data"]
        logger.error(text)
        with torch.no_grad():
            response["user-query"] = text
            content = convert_special_task(client.tokenizer, text, task_type, special_cls)

            ids_length = content["input_tokens"].size(1)
            context = move_to_cuda(content, client.device)
            reply = []
            for i in range(repeat):
                if special_cls:
                    out = client.generator(client.special_cls_model, client.tokenizer, length, context=context, temperature=temperature, temperature_lens=temperature_lens, top_k=sample_k, top_k_lens=top_k_lens, top_p=top_p, punishRate=punishRate, device=client.device, sample=True)
                else:
                    out = client.generator(client.split_sentence_model, client.tokenizer, length, context=context, temperature=temperature, top_p=top_p, top_k_lens=top_k_lens, temperature_lens=temperature_lens, top_k=sample_k, punishRate=punishRate, device=client.device, sample=True)
                out = out[ids_length-1:]
                out = client.tokenizer.convert_ids_to_text(out)
                out = out.replace("##", "")
                reply.append(out)
            if len(reply) == 1:
                reply = reply[0]

            beam_out = None
            if is_beam != -1:
                if special_cls:
                    client.special_beam.beam_size = is_beam
                    client.special_beam.max_lens = length
                    beam_out = client.special_beam.generate_response(context, temperature=1, temperature_lens=0, beam_k=beam_k)
                else:
                    client.split_sentence_beam.beam_size = is_beam
                    client.split_sentence_beam.max_lens = length
                    beam_out = client.split_sentence_beam.generate_response(context, temperature=1, temperature_lens=0, beam_k=beam_k)
            response["sampling-response"] = reply
            if beam_out is not None:
                if not isinstance(beam_out[0], list):
                    response["beam-response"] = client.tokenizer.convert_ids_to_text(beam_out).replace("##", "")
                else:
                    response["beam-response"] = [client.tokenizer.convert_ids_to_text(r).replace("##", "") for r in beam_out]
        interval = time.time() - begin_time
        logger.error("elapsed time = %s", interval)
        response["interval"] = interval
    return json.dumps(response, ensure_ascii=False)


@app.route('/generator', methods=['POST'])
def generator():
    content = request.get_json(silent=True, force=True)
    error_message = {}
    if "data" in content:
        length = content.get("length", 30)
        min_length = content.get("min-length", 3)
        is_beam = content.get("beam", -1)
        beam_k = content.get("beam-k", 1)
        if beam_k > is_beam:
            beam_k = is_beam
        sample_k = content.get("top-k", [5])
        top_k_lens = content.get("top-k-lens", [200])
        top_p = content.get("top-p", 5)
        task_type = content.get("type", 0)
        punishRate = content.get("punishRate", 1)
        temperature = content.get("temperature", [1])
        temperature_lens = content.get("temperature-lens", [0])
        if not isinstance(sample_k, list) or not isinstance(top_k_lens, list):
            error_message["error"] = "top-k expect list type, not {}".format(type(sample_k))
            return json.dumps(error_message, ensure_ascii=False)
        elif len(sample_k) != len(top_k_lens):
            error_message["error"] = "top-k lens not equal to top-k-lens"
            return json.dumps(error_message, ensure_ascii=False)
        if not isinstance(temperature, list) or not isinstance(temperature_lens, list):
            error_message["error"] = "temperature expect list type, not {}".format(type(temperature))
            return json.dumps(error_message, ensure_ascii=False)
        elif len(temperature)!=len(temperature_lens):
            error_message["error"] = "temperature lens not equal to temperature_lens"
            return json.dumps(error_message, ensure_ascii=False)
        special_cls = content.get("special-cls", False)
        # pdb.set_trace()
        if task_type not in [0, 1, 2]:
            task_type = 0
        repeat = content.get("repeat", 1)
        response = OrderedDict()
        begin_time = time.time()
        logger.error("user message...")
        text = content["data"]
        logger.error(text)
        with torch.no_grad():
            response["user-query"] = text
            content = convert_special_task(client.tokenizer, text, task_type, special_cls)
            sep1 = client.tokenizer.sep1()
            ids_length = content["input_tokens"].size(1)
            context = move_to_cuda(content, client.device)
            reply = []
            while True:
                if special_cls:
                    out = client.generator(client.special_cls_model, client.tokenizer, length, context=context, temperature=temperature, temperature_lens=temperature_lens, top_k=sample_k, top_k_lens=top_k_lens, top_p=top_p, punishRate=punishRate, device=client.device, sample=True)
                else:
                    out = client.generator(client.split_sentence_model, client.tokenizer, length, context=context, temperature=temperature, temperature_lens=temperature_lens, top_k=sample_k, top_k_lens=top_k_lens, top_p=top_p, punishRate=punishRate, device=client.device, sample=True)
                out = out[ids_length-1:]
                if sep1 in out:
                    out = out[:out.index(sep1)]
                out_lens = len(out)
                if out_lens >= length or out_lens <= min_length:
                    continue
                out = client.tokenizer.convert_ids_to_text(out)
                out = out.replace("##", "")
                reply = [out]
                break
            if len(reply) == 1:
                reply = reply[0]

            beam_out = None
            if is_beam != -1:
                if special_cls:
                    client.special_beam.beam_size = is_beam
                    client.special_beam.max_lens = length
                    beam_out = client.special_beam.generate_response(context, temperature=temperature, temperature_lens=temperature_lens, beam_k=beam_k)
                else:
                    client.split_sentence_beam.beam_size = is_beam
                    client.split_sentence_beam.max_lens = length
                    beam_out = client.split_sentence_beam.generate_response(context, temperature=temperature, temperature_lens=temperature_lens, beam_k=beam_k)
            response["sampling-response"] = reply
            if beam_out is not None:
                if not isinstance(beam_out[0], list):
                    response["beam-response"] = client.tokenizer.convert_ids_to_text(beam_out).replace("##", "")
                else:
                    response["beam-response"] = [client.tokenizer.convert_ids_to_text(r).replace("##", "") for r in beam_out]
        interval = time.time() - begin_time
        logger.error("elapsed time = %s", interval)
        response["interval"] = interval
    return json.dumps(response, ensure_ascii=False)


if __name__ == '__main__':
    print("Serving start on {} .".format(client.port))
    # app.run(host='0.0.0.0', port=client.args.port)
    run_simple('127.0.0.1', client.port, app)