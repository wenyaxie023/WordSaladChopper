#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

METHOD="wsc"
DTYPE="bfloat16"
DATASET_LIST=("math500")
PROBER_PATH=<prober_path>
echo "Starting WordSaladChopper generation..."

for DATASET in "${DATASET_LIST[@]}"; do
    python src/generate.py \
        --prober-path $PROBER_PATH \
        --dataset $DATASET \
        --method $METHOD \
        --dtype $DTYPE \
        --temperature 0.6 \
        --top-p 0.95
done