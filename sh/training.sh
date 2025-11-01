#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

echo "Starting WordSaladChopper probe training..."

python src/train_prober.py \
    --config configs/train.yaml

echo "Training completed!"
