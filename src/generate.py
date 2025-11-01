from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import hashlib
import yaml
import torch

from wscgen.chopper import Chopper
from wscgen.prober import build_prober
from wscgen.wsc_datasets import get_dataset_handler
from wscgen.generate import wsc_generate
from wscgen.utils import set_seed, setup_logger, prepare_resume, find_newline_token_ids


def vanilla_generate(model, tokenizer, prompt_txt, gen_cfg, max_new_tokens):
    """Plain generation to mirror wsc_generate's return schema (minimal)."""

    inputs = tokenizer(prompt_txt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generate_kwargs = dict(max_new_tokens=max_new_tokens, **gen_cfg)
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)

    gen_ids = output[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return {
        "response": response,
        "total_used_tokens": int(gen_ids.shape[-1] -1), ## -1 for the <eos> token
        "rescue_time": 0.0,
        "kept_texts": [],
        "kept_scores": [],
    }

class TextHandler:
    """For --text mode: build messages from raw text; correctness is undefined."""
    def build_messages(self, sample):
        return [{"role": "user", "content": sample["text"]}]
    def check_correctness(self, sample, response):
        return None

def get_config_str(args):
    temp_str = str(args.temperature).replace(".", "p")
    top_p_str = str(args.top_p).replace(".", "p")
    return f"temp_{temp_str}_top_p_{top_p_str}_len_{args.rescue_budget}"

def get_cfg_hash(args, exclude_keys=("dataset", "dataset_root")):
    config_dict = {k: v for k, v in vars(args).items() if k not in exclude_keys}
    cfg_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(cfg_json.encode()).hexdigest()[:8]

def run_one_pass(
    sample, *,
    handler, model, tokenizer, chopper,
    newline_token_ids, gen_cfg, rescue_prompt,
    token_budget, rescue_budget, max_rescues, method
):
    """
    Run one pass of generation and evaluation.
    """

    prompt = handler.build_messages(sample)
    prompt_txt = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True)

    if method == "wsc":
        result = wsc_generate(
            model, tokenizer, prompt_txt, chopper,
            newline_token_ids=newline_token_ids,
            gen_cfg=gen_cfg, rescue_prompt=rescue_prompt,
            token_budget=token_budget, rescue_budget=rescue_budget,
            max_rescues=max_rescues)
            
    else:
        result = vanilla_generate(
            model, tokenizer, prompt_txt, gen_cfg, max_new_tokens=token_budget
        )
    return {
        "prompt": prompt_txt,
        "response": result["response"],
        "correct": handler.check_correctness(sample, result["response"]),
        "used_tokens": result["total_used_tokens"],
        "rescue_time": result["rescue_time"],
        "kept_texts": result["kept_texts"],
        "full_scores": result["full_scores"],
        "full_texts": result["full_texts"],
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation for WSC Generate")

    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--use-flash-attention-2", action="store_true", default=False)

    # model params
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

    # method
    p.add_argument("--method", choices=["wsc", "vanilla"], default="wsc",
                   help="Choose between WSC method or plain vanilla generation baseline")

    # dataset params
    p.add_argument("--dataset", help="Dataset name supported by `get_dataset_handler`")
    p.add_argument("--dataset-root", type=str, default=None, help="Local dataset path")
    p.add_argument("--text", help="Path to a local text file for demo mode (directly used as prompt input)")

    # chopper params
    p.add_argument("--prober-kind", type=str, default="logistic", help="Prober kind")
    p.add_argument("--prober-path", type=str, help="Path to *.pkl prober file")
    p.add_argument("--thresh", type=float, default=0.5, help="Threshold for chopper")
    p.add_argument("--streak-len", type=int, default=2, help="Streak length for chopper")
    p.add_argument("--len-threshold", type=int, default=10, help="Length threshold for chopper")
    p.add_argument("--short-streak-len", type=int, default=5, help="Short streak length for chopper")
    
    # generation params
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--n-samples", type=int, default=1, help="Number of samples to generate")
    p.add_argument("--token-budget", type=int, default=32768, help="Token budget for generation (first generation)")
    p.add_argument("--rescue-budget", type=int, default=4096, help="Rescue budget for generation (rescues)")
    p.add_argument("--max-rescues", type=int, default=1, help="Maximum number of rescues")
    p.add_argument("--rescue-prompt", default="Let me reconsider this problem with a clear and confident mindset.", help="Rescue prompt")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)

    # save params
    p.add_argument("--out-dir", type=str, default="./outputs", help="Output directory")
    p.add_argument("--override", action="store_true", help="Override existing results")
    
    # selective run params
    p.add_argument("--indices-file", type=str, default=None,
                   help="Path to a JSON array of dataset indices to run, in desired order (e.g., output of filter.py)")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1. Init
    set_seed(args.seed)
    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Invalid dtype: {args.dtype}")

    torch_dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        use_flash_attention_2=args.use_flash_attention_2
    )
    if args.method == "wsc":
        prober = build_prober(args.prober_kind)
        prober_path = args.prober_path
        prober = prober.load(prober_path)
        chopper = Chopper(tokenizer=tokenizer, 
                      detector=prober,
                      thresh=args.thresh, 
                      streak_len=args.streak_len, 
                      len_threshold=args.len_threshold, 
                      short_streak_len=args.short_streak_len)
    else:
        prober = None
        chopper = None
    
    newline_token_ids = find_newline_token_ids(tokenizer)

    dataset = args.dataset
    dataset_root = args.dataset_root
    gen_cfg = {}
    if args.temperature == 0.0:
        args.n_samples = 1
        gen_cfg["do_sample"] = False
    else:
        gen_cfg["do_sample"] = True
        gen_cfg["temperature"] = args.temperature
        gen_cfg["top_p"] = args.top_p
    rescue_prompt = args.rescue_prompt
    token_budget = args.token_budget
    rescue_budget = args.rescue_budget
    max_rescues = args.max_rescues
    seed = args.seed
    n_samples = args.n_samples
    model_id = args.model.split("/")[-1]

    ## config_str: generation config
    config_str = get_config_str(args)
    cfg_hash = get_cfg_hash(args, exclude_keys=("dataset", "dataset_root"))

    dataset = args.dataset
    dataset_root = args.dataset_root
    method = args.method

    if args.text:
        ## get name of the text file
        text_name = Path(args.text).name.split(".")[0]
        run_dir = Path(args.out_dir) / model_id / method / text_name / config_str / cfg_hash
    else:
        run_dir = Path(args.out_dir) / model_id / method / dataset / config_str / cfg_hash

    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_path=run_dir / "log.txt")
    logger.info(f"run_dir: {run_dir}")

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(vars(args), f)

    # 2. Prepare samples and results
    if args.text is not None:
        with open(args.text, "r") as f:
            samples = [{"text": f.read()}]
        handler = TextHandler()
        args.override = True
    else:
        handler = get_dataset_handler(dataset)
        samples = list(handler.load(dataset_root))

        # If user provided indices file, restrict and order the samples accordingly
        if args.indices_file is not None:
            try:
                with open(args.indices_file, "r", encoding="utf-8") as f:
                    selected_indices = json.load(f)
                if not isinstance(selected_indices, list):
                    raise ValueError("indices-file must contain a JSON array")
                # validate and cast to int
                indices: list[int] = []
                for v in selected_indices:
                    iv = int(v)
                    if iv < 0 or iv >= len(samples):
                        raise IndexError(f"Index out of range: {iv}")
                    indices.append(iv)
                samples = [samples[i] for i in indices]
            except Exception as e:
                raise RuntimeError(f"Failed to apply indices-file '{args.indices_file}': {e}")

    all_results, samples_to_run = prepare_resume(run_dir, samples, args.override, logger)
    logger.info(f"all_results: {len(all_results)}")
    logger.info(f"samples_to_run: {len(samples_to_run)}")

    # 3. Run WSCGenerate
    for idx, sample in tqdm(enumerate(samples_to_run), total=len(samples_to_run),
                            desc=f"Generating {dataset}"):
        per_sample_runs = []
        # Run n_samples times
        for k in range(args.n_samples):
            result = run_one_pass(
                    sample,
                    handler=handler, model=model, tokenizer=tokenizer, chopper=chopper,
                    newline_token_ids=newline_token_ids, gen_cfg=gen_cfg,
                    rescue_prompt=rescue_prompt, token_budget=token_budget,
                    rescue_budget=rescue_budget, max_rescues=max_rescues, method=args.method
                )
            per_sample_runs.append(result)


        base = {
            "id": sample.get("id", idx),
            "responses": [r["response"] for r in per_sample_runs],
            "avg_used_tokens": sum(r["used_tokens"] for r in per_sample_runs) / args.n_samples,
            "correctnesses": [r["correct"] for r in per_sample_runs],
            "details": [{
                "kept_texts": r["kept_texts"],
                "full_scores": r["full_scores"],
                "full_texts": r["full_texts"],
                "used_tokens": r["used_tokens"],
                "rescue_time": r["rescue_time"],
            } for r in per_sample_runs]
        }

        if "question" in sample and "answer" in sample:
            base["question"] = sample["question"]
            base["answer"] = sample["answer"]
            corr_vals = [int(bool(c)) for c in base["correctnesses"]]
            base['correctness'] = [r["correct"] for r in per_sample_runs]
            base["avg_correctness"] = sum(corr_vals) / args.n_samples
        else:
            # text mode: no labels
            base["input_text"] = sample.get("text", "")
            base["avg_correctness"] = None
        all_results.append(base)

        if idx % 1 == 0:
            (run_dir / "temp.json").write_text(
                json.dumps(all_results, indent=2, ensure_ascii=False))

    # 4. Save results
    total_samples = len(all_results)
    # dataset accuracy only if labels exist
    labeled = [s for s in all_results if s.get("avg_correctness") is not None]
    dataset_accuracy = (sum(s["avg_correctness"] for s in labeled) / len(labeled)) if labeled else None
    dataset_avg_tokens = sum(s["avg_used_tokens"] for s in all_results) / total_samples


    if dataset_accuracy is not None:
        logger.info(f"Accuracy (Average n={args.n_samples} times): {dataset_accuracy:.4f}")
    else:
        logger.info("Accuracy: N/A (text mode)")
    logger.info(f"Avg used tokens : {dataset_avg_tokens:.1f}")

    (Path(run_dir) / "results.json").write_text(
        json.dumps(all_results, indent=4, ensure_ascii=False))

    summary = {
        "config": vars(args),
        "accuracy": dataset_accuracy,
        "avg_used_tokens": dataset_avg_tokens,
    }
    (Path(run_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    
    with open(run_dir / f"summary.json", "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)


