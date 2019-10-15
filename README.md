# Setup
当前库支持 python3.6，其他的python版本没有经过测试。

你需要安装最新的pytorch。

然后：

```
pip install -r requirements.txt
```


# Usage

## 准备你的训练数据

在代码主目录下，提供了两个小的测试文件，tmp_data/train.txt和tmp_data/valid.txt,分别对应着训练数据和验证数据。你可以把这两个文件替换成你要训练的数据。

然后运行下面的脚本：

`bash scripts/make_data.sh`

你可以看到在tmp_data下有一个data-bin文件夹，处理好的数据存放在此文件夹下。

## GPT training

我们提供了两个脚本，他们会自动存checkpoints，并且可以随时停止，然后下次重新训练的时候自动加载，继续训练。

### 单卡训练

`bash scripts/train.sh`

`data`是存放数据的路径，`train-prefix`是训练数据文件的前缀，`valid-interval`指定了多久update去验证一次，并且 会存入checkpoints，`max-tokens`指定一个 gpu中一个batch可以存放多少 token，`log-interval`指定 多少update打印日志，`save`存放checkpoints的路径。

```
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
```

### 多卡GPU训练
`bash scripts/distributed_train.sh`

`WORLD_SIZE`指定GPU的个数，`MASTER_ADDR 和 MASTER_PORT`通讯协议，`NNODES`多机训练，一般设为1


