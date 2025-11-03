#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

echo "Starting WordSaladChopper probe training..."

python src/train_prober.py \
    --config configs/train.yaml

echo "Training completed!"
