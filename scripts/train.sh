#!/bin/bash

RANK=0
WORLD_SIZE=1
CUDA_VISIBLE_DEVICES=0

DATA="tmp_data/data-bin"
MODEL_CONFIG="model-config/"
# LOAD_MODEL="checkpoints/multi-task-gpt/checkpoints-best.pt"
SAVE_PT="checkpoints/gpt-test"

# --load-model \
# --strict \
# --fix-model-state \
# --load  $LOAD_MODEL \
python pretrain_bert.py \
    --model-config  $MODEL_CONFIG\
    --data   $DATA\
    --train-prefix train-CLM \
    --valid-prefix valid-CLM \
    --valid-interval 5000 \
    --max-tokens 1000 \
    --max-lens 512 \
    --train-batch 4 \
    --valid-batch 4  \
    --log-interval 1 \
    --multi-doc \
    --no-cache \
    --use-cls-special \
    --epochs 30 \
    --save $SAVE_PT \
    --save-optim \
    --tokenizer-type BertWordPieceTokenizer \
    --cache-dir model-config/cache_dir \
    --tokenizer-model-type bert-base-chinese \
    --vocab-size 21128 \
    --max-preds-per-seq 80 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --num-layers 12 \
    --hidden-size 768 \
    --intermediate-size 3072 \
    --num-attention-heads 12 \
    --hidden-dropout 0.1 \
    --attention-dropout 0.1 \
    --lr 2.25e-4 \
    --lr-decay-style linear \
    --warmup 0.02 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-workers 2 \

