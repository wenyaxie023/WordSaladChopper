set -e
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

ROOT=reproduced
cd "$ROOT"

python run.py -m \
  labeler=semantic \
  prober=logistic \
  extractor=layer27_full \
  model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  +pos_neg_ratio="1:1"