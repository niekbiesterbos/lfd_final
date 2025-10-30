#!/usr/bin/env bash
set -euo pipefail

# # SVM baseline
python src/svm_baseline.py \
  --train data/train.tsv --dev data/dev.tsv --test data/test.tsv \
  --outdir artifacts/svm --seed 42 --ngram-max 2 --min-df 2

# # LSTM baseline
python src/lstm.py \
  --train data/train.tsv --dev data/dev.tsv --test data/test.tsv \
  --outdir artifacts/lstm --seed 42 \
  --emb-path assets/glove.6B.100d.txt  \
  --hidden 128 --layers 1 --dropout 0.2 --batch-size 64 --epochs 10 \
  --max-len 64 --min-freq 2 --freeze-emb

# Transformer fine-tuning 
python src/finetune_transformer.py \
  --train data/train.tsv --dev data/dev.tsv --test data/test.tsv \
  --outdir artifacts/transformer --seed 42 \
  --model-name microsoft/deberta-v3-base --epochs 3 --lr 2e-5 --batch-size 32 --max-len 128
