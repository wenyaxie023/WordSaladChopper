#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

METHOD="wsc"
DTYPE="bfloat16"
DATASET_LIST=("aime25")
PROBER_PATH=prober/DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ENABLE_TIMING=false
echo "Starting WordSaladChopper generation..."

TIMING_ARGS=()
if [ "$ENABLE_TIMING" = true ]; then
    TIMING_ARGS+=(--enable-timing)
fi

for DATASET in "${DATASET_LIST[@]}"; do
    python src/generate.py \
        --prober-path $PROBER_PATH \
        --dataset $DATASET \
        --method $METHOD \
        --dtype $DTYPE \
        --temperature 0.6 \
        --top-p 0.95 \
        --model $MODEL_NAME \
        "${TIMING_ARGS[@]}"
done