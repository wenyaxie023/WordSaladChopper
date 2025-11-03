#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

METHOD="wsc"
DTYPE="bfloat16"
DATASET_LIST=("math500")
PROBER_PATH=prober/DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
echo "Starting WordSaladChopper generation..."

for DATASET in "${DATASET_LIST[@]}"; do
    python src/generate.py \
        --prober-path $PROBER_PATH \
        --dataset $DATASET \
        --method $METHOD \
        --dtype $DTYPE \
        --temperature 0.6 \
        --top-p 0.95 \
        --model $MODEL_NAME
done