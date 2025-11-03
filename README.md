# WordSaladChopper

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**WordSaladChopper (WSC)** is a lightweight plug-and-play module that detects and removes *“word salad”* repetitions in large reasoning models.

![WordSaladChopper Paper Preview](asset/wordsaladchopper.jpg)



---
## News
- [11/01/2025] v1 released — first public version now available.

## 🚀 1. Quick Start

Here we show the on-the-fly (end-to-end) version implemented with Hugging Face.
For large-scale paper reproduction, see [reproduced/README.md](reproduced/README.md).

### Installation

Prerequisites:
- Python 3.10+
- PyTorch with a compatible CUDA setup (optional, recommended for GPU inference)

```bash
git clone https://github.com/wenyaxie023/WordSaladChopper.git
cd WordSaladChopper
conda create -n wsc python=3.10
conda activate wsc
pip install -e .
```

### Example

This example uses DeepSeek-R1-Distill-Qwen-7B with a ready-to-use classifier hosted on Hugging Face.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from wscgen.chopper import Chopper
from wscgen.generate import wsc_generate
from wscgen.prober import build_prober
from wscgen.utils import find_newline_token_ids, set_seed

set_seed(41)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the trained prober
prober_file = hf_hub_download(
    repo_id="xiewenya/WordSaladChopper_Classifier",
    filename="DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl",
    repo_type="model",
)
prober = build_prober("logistic").load(prober_file)

# Initialize chopper
chopper = Chopper(
    tokenizer=tokenizer, detector=prober,
    thresh=0.5, streak_len=2, short_streak_len=5, len_threshold=10
)

messages = [
    {"role": "user", "content": "Return your final response within \\boxed{}. Compute: $1-2+3-4+5- \\dots +99-100$."}
]
prompt_txt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

newline_token_ids = find_newline_token_ids(tokenizer)
gen_cfg = {"temperature": 0.6, "top_p": 0.95}


result_wsc = wsc_generate(
    model, tokenizer, prompt_txt, chopper,
    newline_token_ids=newline_token_ids, gen_cfg=gen_cfg,
    rescue_prompt="I can find a clearer solution if I focus on the core problem.",
    token_budget=32768, rescue_budget=4096, max_rescues=1
)

print("Generated text:", result_wsc["response"])
print("Total tokens used:", result_wsc["total_used_tokens"])
```

### Ready-to-Use Classifiers
- **DeepSeek-R1-Distill-Qwen-7B** → `DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl`
- **DeepSeek-R1-Distill-Qwen-1.5B** → `DeepSeek-R1-Distill-Qwen-1.5B_s1/probe.pkl`
- **DeepSeek-R1-Distill-Llama-8B** → `DeepSeek-R1-Distill-Llama-8B_s1/probe.pkl`

Fetch any probe via:
```python
from huggingface_hub import hf_hub_download

probe_file = hf_hub_download(
    repo_id="xiewenya/WordSaladChopper_Classifier",
    filename="DeepSeek-R1-Distill-Qwen-1.5B_s1/probe.pkl",  # swap to the probe you need
    repo_type="model",
)
```
---

## 2. CLI Usage
### 2.1 Prepare the Prober

```bash
mkdir -p prober/DeepSeek-R1-Distill-Qwen-7B_s1
wget https://huggingface.co/xiewenya/WordSaladChopper_Classifier/resolve/main/DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl \
  -O prober/DeepSeek-R1-Distill-Qwen-7B_s1/probe.pkl
```

### 2.2 Generate

```bash
bash sh/generate.sh
```

To use your own classifier, set `PROBER_PATH` in `sh/generate.sh` before running.
---

## 3. Training a Custom Classifier (Probe)

The classifier detects **hidden-state signals of degeneracy**.
You can train your own classifier on:

* your own reasoning traces, or
* our released datasets on Hugging Face.

### Available Training Data

We provide s1 reasoning traces on Hugging Face for multiple models. You can download them and put them in your data path.

```bash
mkdir -p data/DeepSeek-R1-Distill-Qwen-7B_s1
wget https://huggingface.co/datasets/xiewenya/WordSaladChopper_Classifier_Data/resolve/main/DeepSeek-R1-Distill-Qwen-7B_s1/results.json \
     -O data/DeepSeek-R1-Distill-Qwen-7B_s1/results.json
```

### Edit Config

Modify data path in `configs/train.yaml` before training:

```yaml
model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

train_mix:
  s1_temp_0_6_top0p95:
    ratio: 1
    temp: 0.6
    top_p: 0.95
    dataset_id: s1_temp_0_6_top0p95
    path: data/DeepSeek-R1-Distill-Qwen-7B_s1/results.json
```

### Start Training

```bash
bash sh/training.sh
```

### Using Your Own Dataset

To train on your own samples:

Step 1: Format your dataset to match the structure of our released traces.
Step 2: Modify the data path in `configs/train.yaml`.
---

## 4. Repository Structure

```
WordSaladChopper/
├── asset/                  # Documentation assets (PDFs, figures)
├── configs/                # Training configs
├── reproduced/             # Full paper reproduction pipeline
├── sh/                     # Shell entrypoints for generation & training
├── src/                    # Thin wrappers for CLI usage & packaging
├── wscgen/                 # Core Python package
│   ├── chopper.py          # Word Salad detection & chopping logic
│   ├── generate.py         # WSC generator
│   ├── prober.py           # Probe loading / inference helpers
│   ├── pipeline/           # Probe training + evaluation utilities
│   ├── training/           # Training loops for probes
│   └── utils.py            # Shared helper functions
└── pyproject.toml          # Project metadata & dependencies
```

---

## 📚 Citation
If you find this work helpful, please cite:

```bibtex
@inproceedings{xie-etal-2025-word,
    title = "Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly",
    author = "Xie, Wenya  and Zhong, Shaochen  and Le, Hoang Anh Duy  and Xu, Zhaozhuo  and Xie, Jianwen  and Liu, Zirui",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing"
}
```

---

## 📄 License
Released under the [MIT License](LICENSE).

---

## Acknowledgment

This project partially builds upon the **SkyThought** large-scale reasoning framework and draws on the evaluation methodology from **Qwen2.5-Math**.
- SkyThought: [https://github.com/NovaSky-AI/SkyThought](https://github.com/NovaSky-AI/SkyThought)
- Qwen2.5-Math: [https://github.com/QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)
