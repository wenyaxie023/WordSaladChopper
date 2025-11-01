set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# Generation parameters
TEMPERATURE=0.6
TOP_P=0.95
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TASK="aime25"
BATCH_SIZE=-1
N_SAMPLES=8

# Paths
ROOT=reproduced
RESULT_DIR="$ROOT/results/full_traces/${TEMPERATURE}/${TOP_P}"
LOG_DIR="$ROOT/logs/full_gen/${TEMPERATURE}/${TOP_P}"

# Log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${MODEL_NAME}/${TASK}_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

timeout 43200 bash -c "
skythought evaluate \
    --model \"$MODEL_NAME\" \
    --task \"$TASK\" \
    --backend vllm \
    --backend-args tensor_parallel_size=2 \
    --sampling-params temperature=\"$TEMPERATURE\",top_p=\"$TOP_P\" \
    --batch-size \"$BATCH_SIZE\" \
    --n \"$N_SAMPLES\" \
    --result-dir \"$RESULT_DIR\" \
    >> \"$LOG_FILE\" 2>&1
"