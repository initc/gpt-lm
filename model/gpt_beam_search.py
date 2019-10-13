# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import numpy as np
import torch

import model.search_utils as search
import pdb

class SequenceGenerator(object):
    def __init__(
        self, models, tokenizer, beam_size, max_lens, temperature=1.0, eos_ids=None
    ):
        if not isinstance(models, list):
            models = [models]
        self.models = models
        self.tokenizer = tokenizer
        self.eos = self.tokenizer.eos() if eos_ids is None else eos_ids
        self.start_idx = None
        self.minlen = 2
        self.pad = self.tokenizer.pad()
        self.unk = self.tokenizer.unk()
        self.vocab_size = len(self.tokenizer)
        self.normalize_scores = True
        self.len_penalty = 1
        self.beam_size = beam_size
        self.max_lens = max_lens
        self.temperature = temperature
        self.search = search.BeamSearch(self.tokenizer)

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def generate_response(
        self, net_input, temperature=1, temperature_lens=0, beam_k=1
    ):

        token_ids = self.generate(
            net_input, self.beam_size, self.max_lens, temperature=temperature, temperature_lens=temperature_lens, beam_k=beam_k
        )
        return token_ids

    def generate(self, encoder_input, beam_size, max_lens, temperature=1, temperature_lens=0, beam_k=1):

        with torch.no_grad():
            token_ids = self.generate_beam(encoder_input, beam_size, max_lens, temperature=temperature, temperature_lens=temperature_lens, beam_k=beam_k)
            return token_ids

    def generate_beam(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None, temperature=1, temperature_lens=0, beam_k=1):

        return self._generate(encoder_input, beam_size, maxlen, prefix_tokens, temperature=temperature, temperature_lens=temperature_lens, beam_k=beam_k)

    def _generate(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None, temperature=1, temperature_lens=0, beam_k=1):
        """See generate"""
        src_tokens = encoder_input["input_tokens"]

        new_order = torch.zeros(beam_size)
        new_order = new_order.to(src_tokens.device).long()
        encoder_input = self.models[0].reorder_encoder_out(encoder_input, new_order)
        src_tokens = encoder_input["input_tokens"]

        scores = src_tokens.data.new(beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()

        finalized = []
        finished = [False]
        # worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]

        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished():

            assert len(finalized) <= beam_size
            if len(finalized) == beam_size:
                # if self.stop_early or step == maxlen or unfinalized_scores is None:
                return True

            return False

        def finalize_hypos(step, beam_idx, eos_scores, unfinalized_scores=None):

            assert beam_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, beam_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos

            pos_scores = scores.index_select(0, beam_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(beam_idx.tolist(), eos_scores.tolist())):

                def get_hypo():

                    return {
                        'tokens': tokens_clone[i][:-1],
                        'score': score,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized) < beam_size:
                    finalized.append(get_hypo())

            if not finished[0] and is_finished():
                finished[0] = True
            return finished[0]

        input_ids = encoder_input["input_tokens"]
        task_type_ids = encoder_input["task_type_ids"] if "task_type_ids" in encoder_input and encoder_input["task_type_ids"] is not None else None
        task_type = task_type_ids[0,-1].item() if task_type_ids is not None else None

        prev_input_ids = input_ids
        prev_task_type_ids = task_type_ids
        pasts = [None] * len(self.models)

        reorder_state = None

        for step in range(maxlen + 1):

            if reorder_state is not None:
                for past in pasts:
                    if past is not None:
                        for i,p in enumerate(past):
                            past[i] = past[i].index_select(1, reorder_state)
            lprobs, pasts = self._decode(step, input_tokens=prev_input_ids, task_type_ids=prev_task_type_ids, pasts=pasts, temperature=temperature, temperature_lens=temperature_lens)

            prev_task_type_ids = torch.LongTensor([[task_type]]).to(lprobs.device).repeat((beam_size,1)) if task_type is not None else None
            # qsj
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] = -math.inf  #
            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)

            eos_idx = buffer('eos_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                # self.search.set_src_lengths(src_lengths)
                cand_scores, cand_indices, cand_beams = self.search.step(
                    step,
                    lprobs.view(-1, self.vocab_size),
                    scores.view(beam_size, -1)[:, :step],
                )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_idx),
                )
                finalize_hypos(step, eos_idx, eos_scores)
                assert finished[0]==True
                break

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_beams[:beam_size],
                    mask=eos_mask[:beam_size],
                    out=eos_idx,
                )
                if eos_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:beam_size],
                        mask=eos_mask[:beam_size],
                        out=eos_scores,
                    )
                    finalize_hypos(step, eos_idx, eos_scores, cand_scores)

            # 提前结束
            if finished[0]:
                break
            assert step < maxlen

            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(0)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            # 去除了eos后，最靠前的K个
            torch.topk(
                active_mask, k=beam_size, dim=0, largest=False,
                out=(_ignore, active_hypos)
            )

            active_beam_idx = buffer('active_beam_idx')
            torch.gather(
                cand_beams, dim=0, index=active_hypos,
                out=active_beam_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=0, index=active_hypos,
                out=scores[:, step].view(beam_size),
            )

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_beam_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=0, index=active_hypos,
                out=tokens_buf.view(beam_size, -1)[:, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_beam_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=0, index=active_hypos,
                out=scores_buf.view(beam_size, -1)[:, step],
            )

            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores

            prev_input_ids = tokens[:,step+1:step+2]
            reorder_state = active_beam_idx

        # sort by score descending
        sorted_tokens = sorted(finalized, key=lambda r: r['score'], reverse=True)
        finalized = [r["tokens"].tolist() for r in sorted_tokens[:beam_k]]
        if len(finalized) == 1:
            return finalized[0]

        return finalized

    def _decode(self, step, input_tokens, task_type_ids=None, pasts=None, temperature=1, temperature_lens=0):
        if len(self.models) == 1:
            assert len(pasts) == 1
            probs, past = self._decode_one(step, self.models[0], input_tokens=input_tokens, task_type_ids=task_type_ids, past=pasts[0], log_probs=True, temperature=temperature, temperature_lens=temperature_lens)
            return probs, [past]
        log_probs = []
        new_pasts = []
        for model, past in zip(self.models, pasts):
            probs, past = self._decode_one(step, model, input_tokens=input_tokens, task_type_ids=task_type_ids, past=past, log_probs=True, temperature=temperature, temperature_lens=temperature_lens)
            new_pasts.append(past)
            log_probs.append(probs)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        return avg_probs, new_pasts

    def _decode_one(self, step, model, input_tokens, task_type_ids, past, log_probs, temperature=1, temperature_lens=0):
        with torch.no_grad():
            decoder_out, _, past = model(input_tokens=input_tokens, task_type_ids=task_type_ids, past=past, clm=True)
            probs = decoder_out[:, -1, :]
            if temperature_lens < step:
                probs /= temperature

        probs = model.get_normalized_probs(probs, log_probs=log_probs)
        return probs, past

