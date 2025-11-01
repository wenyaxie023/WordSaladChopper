set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false

ROOT=reproduced
cd "$ROOT"
python run.py --config-name gen -m \
  labeler=semantic \
  prober=logistic \
  extractor=layer27_full \
  trimmer=streak_filter_rollback_0p5_2_10_5 \
  eval_datasets=aime25_temp_0_6_top0p95_n8 \
  continuation_cues=version_2 \
  gen_params=temp_0_6_top0p95_4k \
  model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
  +pos_neg_ratio="1:1" \
  +override=true