#!/bin/bash

# number of gpu in single node
WORLD_SIZE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
# number of node
NNODES=1
# master node
NODE_RANK=0

DATA="tmp_data/data-bin"
MODEL_CONFIG="model-config/"
# LOAD_MODEL="checkpoints/multi-task-gpt/checkpoints-best.pt"
SAVE_PT="checkpoints/gpt-test"

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    pretrain_bert.py \
    # --load-model \
    # --strict \
    # --fix-model-state \
    # --load  $LOAD_MODEL \
    --model-config  $MODEL_CONFIG\
    --data   $DATA\
    --train-prefix train-CLM \
    --valid-prefix valid-CLM \
    --valid-interval 5000 \
    --max-tokens 4000 \
    --max-lens 512 \
    --train-batch 4 \
    --valid-batch 4
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

