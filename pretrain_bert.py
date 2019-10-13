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

"""Pretrain BERT"""

import os
import random
import math
import numpy as np
import torch

from arguments import get_args
from configure_data import configure_data
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import BertModel
from model import get_params_for_weight_decay_optimization
from model import DistributedDataParallel as DDP
from optim import Adam
from utils import Timers, save_checkpoint, load_checkpoint, check_checkpoint, move_to_cuda
import pdb


def get_model(tokenizer, args):
    """Build the model."""

    print('building BERT model ...')
    model = BertModel(tokenizer, args)
    print(' > number of parameters: {}'.format(
        sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        print("fp16 mode")
        model = FP16_Module(model)
        if args.fp32_embedding:
            model.module.model.bert.embeddings.word_embeddings.float()
            model.module.model.bert.embeddings.position_embeddings.float()
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_tokentypes:
            model.module.model.bert.embeddings.token_type_embeddings.float()
        if args.fp32_layernorm:
            for name, _module in model.named_modules():
                if 'LayerNorm' in name:
                    _module.float()
    # Wrap model for distributed training.
    if args.world_size > 1:
        model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    layers = model.model.bert.encoder.layer
    pooler = model.model.bert.pooler
    lmheads = model.model.cls.predictions
    nspheads = model.model.cls.seq_relationship
    embeddings = model.model.bert.embeddings
    param_groups = []
    param_groups += list(get_params_for_weight_decay_optimization(layers))
    param_groups += list(get_params_for_weight_decay_optimization(pooler))
    param_groups += list(get_params_for_weight_decay_optimization(nspheads))
    param_groups += list(get_params_for_weight_decay_optimization(embeddings))
    param_groups += list(get_params_for_weight_decay_optimization(
        lmheads.transform))
    param_groups[1]['params'].append(lmheads.bias)

    # Use Adam.
    optimizer = Adam(param_groups,
                     lr=args.lr, weight_decay=args.weight_decay)

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters * args.epochs
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step)

    return lr_scheduler


def setup_model_and_optimizer(args, tokenizer):
    """Setup model and optimizer."""

    model = get_model(tokenizer, args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
    args.continue_train = False

    check_checkpoint(model, optimizer, lr_scheduler, args)

    if args.load is not None and not args.continue_train:
        print("| Resume checkpoints from {}".format(args.load))
        epoch, i, total_iters = load_checkpoint(model, optimizer,
                                                lr_scheduler, args)
        if args.resume_dataloader:
            args.epoch = epoch
            args.mid_epoch_iters = i
            args.total_iters = total_iters

    

    return model, optimizer, lr_scheduler, criterion


def forward_step(data, model, tokenizer, criterion, args):
    """Forward step."""

    sample = move_to_cuda(data, torch.cuda.current_device())
    output, nsp, past = model(**sample["net_input"])
    nsp_labels = sample["nsp_labels"]
    target = sample["target"]
    nsp_loss = criterion(nsp.view(-1, 3).contiguous().float(),
                         nsp_labels.view(-1).contiguous())
    losses = criterion(output.view(-1, tokenizer.num_tokens).contiguous().float(),
                       target.contiguous().view(-1).contiguous())
    # pdb.set_trace()

    return losses, nsp_loss, sample["nsentences"], sample["ntokens"]


def backward_step(optimizer, model, lm_loss, nsp_loss, batch_size, batch_tokens, args):
    """Backward step."""

    # Total loss.
    loss = lm_loss / batch_tokens + nsp_loss / batch_size

    # Backward pass.
    optimizer.zero_grad()
    if args.fp16:
        optimizer.backward(loss, update_master_grads=False)
    else:
        loss.backward()
    # Reduce across processes.
    lm_loss_reduced = lm_loss
    nsp_loss_reduced = nsp_loss
    if args.world_size > 1:
        batch_size = torch.Tensor([batch_size]).to(lm_loss.device)
        batch_tokens = torch.Tensor([batch_tokens]).to(lm_loss.device)
        reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1), batch_size, batch_tokens))

        torch.distributed.all_reduce(reduced_losses.data)
        # reduced_losses.data = reduced_losses.data / args.world_size
        model.allreduce_params(reduce_after=False,
                               fp32_allreduce=args.fp32_allreduce)
        lm_loss_reduced = reduced_losses[0]
        nsp_loss_reduced = reduced_losses[1]
        batch_size = reduced_losses[2].item()
        batch_tokens = reduced_losses[3].item()

    # Update master gradients.
    if args.fp16:
        optimizer.update_master_grads()

    # Clipping gradients helps prevent the exploding gradient.
    if args.clip_grad > 0:
        if not args.fp16:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced, nsp_loss_reduced, batch_size, batch_tokens


def train_step(input_data, model, tokenizer, criterion, optimizer, lr_scheduler, args):
    """Single training step."""
    # Forward model for one step.
    lm_loss, nsp_loss, batch_size, batch_tokens = forward_step(input_data, model, tokenizer, criterion, args)

    # Calculate gradients, reduce across processes, and clip.
    lm_loss_reduced, nsp_loss_reduced, batch_size, batch_tokens = backward_step(optimizer, model, lm_loss,
                                                                                nsp_loss, batch_size, batch_tokens,
                                                                                args)

    # Update parameters.
    optimizer.step()

    # Update learning rate.
    skipped_iter = 0
    if not (args.fp16 and optimizer.overflow):
        lr_scheduler.step()
    else:
        skipped_iter = 1

    return lm_loss_reduced, nsp_loss_reduced, skipped_iter, batch_size, batch_tokens


def train_epoch(epoch, model, tokenizer, optimizer, train_data, val_data,
                lr_scheduler, criterion, timers, args):
    """Train one full epoch."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_nsp_loss = 0.0

    # Iterations.
    max_iters = len(train_data)
    iteration = 0
    update_num = 0
    total_tokens = 0
    total_batch = 0
    skipped_iters = 0
    data_iterator = iter(train_data)

    def comsume_data(times):
        for i in range(times):
            next(data_iterator)

    if args.resume_dataloader:
        iteration = args.mid_epoch_iters
        comsume_data(iteration)
        args.resume_dataloader = False
        lr_scheduler.step(max_iters * (epoch-1) + iteration)

    # Data iterator.
    timers('interval time').start()
    while iteration < max_iters:

        lm_loss, nsp_loss, skipped_iter, batch_size, batch_tokens = train_step(next(data_iterator), model, tokenizer, criterion,optimizer, lr_scheduler, args)
        update_num += 1
        skipped_iters += skipped_iter
        iteration += 1
        args.cur_iteration = iteration

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float().item()
        total_nsp_loss += nsp_loss.data.detach().float().item()
        if nsp_loss != 0.0:
            total_batch += batch_size
        total_tokens += batch_tokens
        if total_batch < 1:
            total_batch = 1
        # Logging.
        if iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_nsp_loss = total_nsp_loss / total_batch
            avg_lm_loss = total_lm_loss / total_tokens
            elapsed_time = timers('interval time').elapsed()
            log_string = ' epoch{:2d} |'.format(epoch)
            log_string += ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            max_iters)
            log_string += ' lm loss {:.3f} |'.format(avg_lm_loss)
            log_string += ' lm ppl {:.3f} |'.format(math.exp(avg_lm_loss))
            log_string += ' nsp loss {:.3f} |'.format(avg_nsp_loss)
            log_string += ' batch size {} |'.format(batch_size)
            log_string += ' learning rate {:.7f} |'.format(learning_rate)
            log_string += ' tpi (ms): {:.2f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            if args.fp16:
                log_string += ' loss scale {:.3f} |'.format(
                    optimizer.loss_scale)
            print(log_string, flush=True)
            if iteration % args.valid_interval == 0:
                lm_loss, nsp_loss = evaluate(val_data, model, tokenizer, criterion, args)
                val_loss = lm_loss + nsp_loss
                print('-' * 100)
                print('| end of epoch {:3d}  | valid loss {:.3f} | '
                      'valid LM Loss {:.3f} | valid LM PPL {:.3f} | valid NSP Loss {:.3f}'.format(
                    epoch, val_loss, lm_loss, math.exp(lm_loss), nsp_loss))
                print('-' * 100)
                if args.save:
                    checkpoints_path = "checkpoints_{}_{}.pt".format(epoch, iteration)
                    save_checkpoint(checkpoints_path, epoch, iteration, model,
                                    optimizer, lr_scheduler, args)
                    checkpoints_path = "checkpoints-last.pt"
                    save_checkpoint(checkpoints_path, epoch, iteration, model,
                                    optimizer, lr_scheduler, args)
                if val_loss < evaluate.best_val_loss:
                    evaluate.best_val_loss = val_loss
                    if args.save:
                        best_path = 'checkpoints-best.pt'
                        print('saving best model to:',
                              os.path.join(args.save, best_path))
                        save_checkpoint(best_path, epoch, iteration, model,
                                        optimizer, lr_scheduler, args)

    if args.save:
        final_path = 'checkpoints_{}.pt'.format(epoch)
        print('saving final epoch model to:', os.path.join(args.save, final_path))
        save_checkpoint(final_path, epoch + 1, 0, model, optimizer, lr_scheduler, args)
        cur_path = 'checkpoints-last.pt'
        save_checkpoint(cur_path, epoch + 1, 0, model, optimizer, lr_scheduler, args)

        lm_loss, nsp_loss = evaluate(val_data, model, tokenizer, criterion, args)
        val_loss = lm_loss + nsp_loss
        if val_loss < evaluate.best_val_loss:
            evaluate.best_val_loss = val_loss
            if args.save:
                best_path = 'checkpoints-best.pt'
                print('saving best model to:',
                      os.path.join(args.save, best_path))
                save_checkpoint(best_path, epoch+1, 0, model,
                                optimizer, lr_scheduler, args)

    return iteration, skipped_iters


def evaluate(data_source, model, tokenizer, criterion, args):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0
    total_nsp_loss = 0
    total_batch_size = 0
    total_batch_tokens = 0
    for data_loader in data_source:
        local_lm_loss = 0
        local_batch_tokens = 0
        max_iters = len(data_loader)
        with torch.no_grad():
            data_iterator = iter(data_loader)
            iteration = 0
            while iteration < max_iters:
                # Forward evaluation.
                lm_loss, nsp_loss, batch_size, batch_tokens = forward_step(next(data_iterator), model, tokenizer,criterion, args)
                # Reduce across processes.
                if isinstance(model, DDP):
                    batch_size = torch.Tensor([batch_size]).to(lm_loss.device)
                    batch_tokens = torch.Tensor([batch_tokens]).to(lm_loss.device)
                    reduced_losses = torch.cat((lm_loss.view(1), nsp_loss.view(1), batch_size, batch_tokens))
                    torch.distributed.all_reduce(reduced_losses.data)
                    # reduced_losses.data = reduced_losses.data / args.world_size
                    lm_loss = reduced_losses[0]
                    nsp_loss = reduced_losses[1]
                    batch_size = reduced_losses[2].item()
                    batch_tokens = reduced_losses[3].item()
                if lm_loss == 0.0:
                    batch_size = 0
                total_lm_loss += lm_loss.data.detach().float().item()
                total_nsp_loss += nsp_loss.data.detach().float().item()
                local_lm_loss += lm_loss.data.detach().float().item()
                local_batch_tokens += batch_tokens
                total_batch_size += batch_size
                total_batch_tokens += batch_tokens
                iteration += 1
        local_lm_loss /= local_batch_tokens
        print('| LOCAL valid LM Loss {:.3f} | valid LM PPL {:.3f}'.format(local_lm_loss, math.exp(local_lm_loss)))

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= total_batch_tokens
    total_nsp_loss /= total_batch_size
    return total_lm_loss, total_nsp_loss


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    if args.world_size > 1:
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size, rank=args.rank,
            init_method=init_method)
        suppress_output(args.rank == 0)


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


def main():
    """Main training program."""

    print('Pretrain BERT model')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    set_random_seed(args.seed)
    print(args)
    # Data stuff.
    data_config = configure_data()
    data_config.set_defaults(data_set_type='BERT', transpose=False)
    (train_data, val_data), tokenizer = data_config.apply(args)

    args.train_iters = len(train_data)
    evaluate.best_val_loss = float("inf")

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler, criterion = setup_model_and_optimizer(
        args, tokenizer)
    # evaluate(val_data, model, tokenizer, criterion, args)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        total_iters = 0
        skipped_iters = 0
        start_epoch = 1
        best_val_loss = float('inf')
        # Resume data loader if necessary.
        if args.resume_dataloader:
            start_epoch = args.epoch
            total_iters = args.total_iters
        # For all epochs.
        for epoch in range(start_epoch, args.epochs + 1):
            timers = Timers()
            # if args.shuffle:
            #     train_data.batch_sampler.sampler.set_epoch(epoch + args.seed)
            timers('epoch time').start()
            iteration, skipped = train_epoch(epoch, model, tokenizer, optimizer,
                                             train_data, val_data, lr_scheduler,
                                             criterion, timers, args)
            elapsed_time = timers('epoch time').elapsed()
            total_iters += iteration
            skipped_iters += skipped
            lm_loss, nsp_loss = evaluate(val_data, model, tokenizer, criterion, args)
            val_loss = lm_loss + nsp_loss
            print('-' * 100)
            print('| end of epoch {:3d} | time: {:.3f}s | valid loss {:.3f} | '
                  'valid LM Loss {:.3f} | valid LM PPL {:.3f} | valid NSP Loss {:.3f}'.format(
                epoch, elapsed_time, val_loss, lm_loss, math.exp(lm_loss), nsp_loss))
            print('-' * 100)
            if val_loss < evaluate.best_val_loss:
                evaluate.best_val_loss = val_loss
                if args.save:
                    best_path = 'checkpoints-best.pt'
                    print('saving best model to:',
                          os.path.join(args.save, best_path))
                    save_checkpoint(best_path, epoch + 1, 0, model, optimizer, lr_scheduler, args)
    except KeyboardInterrupt:
        print('-' * 100)
        print('Exiting from training early')
        if args.save:
            cur_path = 'checkpoints-last.pt'
            print('saving current model to:',
                  os.path.join(args.save, cur_path))
            save_checkpoint(cur_path, epoch, args.cur_iteration, model, optimizer, lr_scheduler, args)
        exit()


if __name__ == "__main__":
    main()
