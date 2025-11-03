# Reproduced Setup (Paper Version)

The paper experiments use a three-stage offline evaluation pipeline instead of on-the-fly inference. This design is more efficient for large-scale benchmarking because reasoning is generated once, and the trimming and regeneration steps operate directly on saved outputs without re-running full model decoding. 

---

## 1. Environment Setup

Create a conda environment and install the dependencies:

```bash
conda create -n wsc_reproduce python=3.10
conda activate wsc_reproduce

cd reproduced
pip install -r requirements.txt
```


## 2. Prepare Offline Traces & Config Paths
Below are the key points regarding how training and evaluation trace data are referenced.

Hydra looks up absolute JSON paths through two resolver files:
- `configs/resources/dataset_paths.yaml` maps each `(model_name, dataset_id)` to the training trace used during probe fitting.
- `configs/resources/eval_dataset_paths.yaml` does the same for evaluation traces consumed in trimming.

### 2.1 Training traces for the probe (`dataset_paths.yaml`)

1. Download or generate trace files that match the model you want to probe. We publish ready-to-use `s1` traces on Hugging Face:

```bash
mkdir -p training_data/DeepSeek-R1-Distill-Qwen-7B_s1
wget https://huggingface.co/datasets/xiewenya/WordSaladChopper_Classifier_Data/resolve/main/DeepSeek-R1-Distill-Qwen-7B_s1/results.json \
     -O training_data/DeepSeek-R1-Distill-Qwen-7B_s1/results.json
```

2. Add (or update) the corresponding entry in `configs/resources/dataset_paths.yaml`. Example:

```yaml
dataset_paths:
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:
    s1_temp_0_6_top0p95: training_data/DeepSeek-R1-Distill-Qwen-7B_s1/results.json
```

### 2.2 Evaluation traces for trimming and regeneration (`eval_dataset_paths.yaml`)

1. Register every evaluation dataset (per model) in `configs/resources/eval_dataset_paths.yaml`. Each key should follow the naming pattern used in `configs/eval_datasets/*.yaml`. Example entry:

```yaml
eval_dataset_paths:
  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B:
    gsm8k_temp_0_6_top0p95_n1: results/full_traces/0.6/0.95/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_gsm8k_xxxxx/results.json (replace your own file path here)
```

2. If you split a dataset across multiple shards (e.g. `..._part1`, `..._part2`), keep the shared prefix identical and optionally set `full_name` in the per-dataset YAML so the pipeline can merge them automatically.

3. Whenever `sh/run_full_gen.sh` produces a new result JSON, copy the absolute path into this mapping so that subsequent `eval` runs can locate the data.

---

## 3. Train the Repetition Classifier (Probe)

1. Select the desired model/extractor pair. The extractor must match the final layer index of the model (e.g. `layer31_full` for `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, `layer27_full` for DeepSeek-R1-Distill-Qwen variants).

2. Optionally adjust `configs/train.yaml`:
   - `train_mix`: choose the dataset mixture file (e.g. `s1_temp_0_6_top0p95`).
   - `labeler` / `prober`: use `semantic`, `logistic`.
   - `root_dir`: default output root (`artifacts/`).

3. Launch training:

```bash
bash sh/run_train.sh
```

Probe checkpoints are saved to `<root_dir>/<model>/<train_mix>/<hash_id>/prober_train/probe.pkl` for reuse in evaluation.

---

## 4. Offline Evaluation Pipeline (Three Stages)

Each stage is driven by a shell wrapper that passes Hydra overrides into `run.py`. You can edit the scripts or call `python run.py --config-name <mode> -m ...` manually.

### Step 1: Generate Full Reasoning Traces

```bash
bash sh/run_full_gen.sh
```

- Key parameters are defined at the top of the script (`TEMPERATURE`, `TOP_P`, `MODEL_NAME`, `TASK`, `N_SAMPLES`). Adjust them to match the experiment.
- Outputs will be stored in `results/full_traces/<temperature>/<top_p>/` with filenames produced by SkyThought (e.g. `deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_gsm8k_<hash>/results.json`).
- Update `configs/resources/eval_dataset_paths.yaml` with the absolute path of each new `results.json` immediately after generation finishes.

### Step 2: Detect and Trim Degenerated Segments

```bash
bash sh/run_eval.sh
```

- Ensure `model_name`, `extractor`, `labeler`, `prober`, `trimmer`, and `eval_datasets` align with the probe you trained.
- `eval_datasets` refers to YAML files under `configs/eval_datasets/`, which in turn point to the IDs registered in `eval_dataset_paths.yaml`.
- Results are written to `<root_dir>/<model>/<train_mix>/<hash_id>/eval_trim/<trim_tag>/`. For each dataset you will find `<dataset>_summary.json` (metrics) and `<dataset>_trimmed_results.json` (trimmed traces). 

### Step 3: Regenerate Trimmed Segments

```bash
bash sh/run_regen.sh
```

- Use the same probe/extractor configuration as Step 2, and keep `eval_datasets`, `trimmer`, etc. consistent.
- The script reads trimmed traces from Step 2 and only regenerates responses whose token count was reduced.
- Regeneration outputs will be stored in `<root_dir>/<model>/<train_mix>/<hash_id>/eval_trim/<trim_tag>/<gen_tag>/<cue>/<dataset>_regen_results.json`, along with aggregated metrics. Ensure `gen_params` matches the evaluation stage (`extractor`, `trimmer`, etc.) so the directory layout aligns.

---
## Support
If you run into issues, please open an issue on the repository so we can help.