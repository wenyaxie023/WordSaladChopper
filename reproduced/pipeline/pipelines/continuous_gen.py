# pipeline/pipelines/continuation_gen.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any, Union

from transformers import AutoTokenizer
import re, hashlib

from pipeline.config    import PipelineConfig
from pipeline.inference import VllmBackend
from pipeline.utils     import save_json, load_json
from pipeline.datasets  import load_dataset_handler

logger = logging.getLogger(__name__)

class ContinuationGenerationPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self.out_dir = cfg.make_gen_dir()
        self.base_trim_dir = cfg.make_trim_dir()
        logger.info(f"out_dir:{self.out_dir}")
        cfg.to_json(self.out_dir / "config.json")

        self.eval_meta: Dict[str, Dict] = {}
        if self.cfg.eval_datasets:
            ds_dict = self.cfg.eval_datasets

            if len(ds_dict) > 1:
                full_names = [v.get("full_name") for v in ds_dict.values()]
                assert all(full_names), (
                    "When multiple eval_datasets are given, each must have a 'full_name'."
                )
                unique_full = set(full_names)
                assert len(unique_full) == 1, (
                    f"All 'full_name's must be identical when merging, got: {unique_full}"
                )

                common_name = full_names[0]
                merged_path = str(self.base_trim_dir / f"{common_name}_trimmed_results.json")

                sample_cfg = dict(next(iter(ds_dict.values())))
                sample_cfg["path"] = merged_path

                self.eval_meta = {common_name: sample_cfg}
                self.cfg.eval_datasets = {common_name: sample_cfg}

            else:
                key, params = next(iter(ds_dict.items()))
                params = dict(params)
                params["path"] = str(self.base_trim_dir / f"{key}_trimmed_results.json")

                self.eval_meta = {key: params}
                self.cfg.eval_datasets = {key: params}
               
        logger.info(f'*'*80)
        logger.info(f"eval_datasets:{self.eval_meta}")
        logger.info(f'*'*80)
        self.handlers: Dict[str, Any] = {}
        self.continuation_cues = cfg.continuation_cues
        self.continuation_cue = (
            self.continuation_cues[0] if self.continuation_cues else ""
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

        # backend
        self.backend = VllmBackend(cfg)
        logger.info(f"-----------continuation_cues-------------")
        logger.info(self.continuation_cues)
        self.cue_len = len(
            self.tokenizer.encode(self.continuation_cue, add_special_tokens=False)
        )
        def _maybe_int(d, k, default):
            logger.info(f"maybe_int:{d},{k},{default}")
            v = d.get(k, default)
            logger.info(f"maybe_int:{v}")
            return int(v) if v is not None else None

        def _maybe_float(d, k, default):
            v = d.get(k, default)
            return float(v) if v is not None else None

        raw = cfg.gen_params or {}
        logger.info(f'='*80)
        logger.info(f"raw:{raw}")
        if raw.get('max_tokens') is None: 
            logger.info(f"raw.get('max_tokens') is none")
            self.gen_kwargs = {
                "temperature": _maybe_float(raw, "temperature", 0.6),
                "top_p": _maybe_float(raw, "top_p", 0.95),
                **({"stop": raw["stop"]} if "stop" in raw else {})
            }
        else:
            logger.info(f"raw.get('max_tokens') is not none")
            self.gen_kwargs = {
                "max_tokens": _maybe_int(raw, "max_tokens", None),
                "temperature": _maybe_float(raw, "temperature", 0.6),
                "top_p": _maybe_float(raw, "top_p", 0.95),
                **({"stop": raw["stop"]} if "stop" in raw else {})
            }
    def _run_single(self, cue: str, cue_tag: str):
        self.continuation_cue = cue
        self.cue_len = len(self.tokenizer.encode(cue, add_special_tokens=False))

        out_dir = (self.out_dir / cue_tag)
        out_dir.mkdir(parents=True, exist_ok=True)

        for ds_name, trim_file in self.cfg.eval_datasets.items():
            handler = self._get_handler(ds_name)

            # check if results already exist
            if (out_dir / f"{ds_name}_regen_results.json").exists():
                logger.warning("[%s] results already exist - skip", ds_name)
                return
            trim_path = Path(trim_file["path"])
            logger.info(f"------------trim_path:{trim_path}---------------")
            if not trim_path.exists():
                logger.warning(f"{ds_name} trimmed_results.json not found - skip")
                continue
            
            data = load_json(trim_path)
            logger.info(f"{ds_name} loaded {len(data)} trimmed samples")

            items: List[Dict] = list(self._iter_items(data))
            to_gen, prompts = [], []
            for it in items:    
                for idx, (trim_resp, orig_resp) in enumerate(zip(it["trimmed_responses"], it["responses"])):
                    need = trim_resp["token_stats"]["trim"] < trim_resp["token_stats"]["orig"]
                    trim_resp["need_regen"] = need
                    if need:
                        to_gen.append((it, idx))
                        prompts.append(self._build_prompt(it, idx, cue))

            # regenerate with continuation cue
            gen_texts: List[str] = []
            gen_token_lens: List[int] = []

            if prompts:
                logger.info(f"{ds_name} generating {len(prompts)} continuations …")
                logger.info(f"*"*80)
                logger.info(f"gen_kwargs when generate:{self.gen_kwargs}")
                logger.info(f"*"*80)
                raw_outs = self.backend.generate_raw(prompts, **self.gen_kwargs)
                
                for out in raw_outs:
                    gen_texts.append(out.outputs[0].text)
                    gen_token_lens.append(len(out.outputs[0].token_ids))

            comp_iter = iter(zip(gen_texts, gen_token_lens))
            

            for it in items:
                regen_resps = []
                for idx, (trim_resp, orig_resp) in enumerate(zip(it["trimmed_responses"], it["responses"])):
                    need_gen = trim_resp["need_regen"]

                    # build prompt
                    prompt_used = self._build_prompt(it, idx, cue)

                    if need_gen:
                        regen_txt, regen_len = next(comp_iter)
                        full_txt = trim_resp["content"] + cue + regen_txt
                        full_len = trim_resp["token_stats"]["trim"] + self.cue_len + regen_len
                    else:
                        regen_txt, regen_len = "", 0
                        full_txt = orig_resp["content"]
                        full_len = trim_resp["token_stats"]["orig"]

                    correct = handler.check_correctness(it, full_txt)

                    regen_resps.append({
                        "need_regen": need_gen,
                        "correctness": bool(correct), 
                        "final_prompt": prompt_used,
                        "final_response": {
                            "regen_content": regen_txt,
                            "full_content": full_txt,
                            "correctness": bool(correct),
                        },
                        "token_stats": {
                            "orig": trim_resp["token_stats"]["orig"],
                            "trim": trim_resp["token_stats"]["trim"],
                            "cue": self.cue_len if need_gen else 0,
                            "gen": regen_len if need_gen else 0,
                            "regen": full_len,
                        },
                    })
                it["regen_responses"] = regen_resps

            # save results
            regen_path = out_dir / f"{ds_name}_regen_results.json"
            save_json(regen_path, data)
            logger.info(f"{ds_name} per-sample results → {regen_path.as_posix()}")

            # aggregate metrics
            metrics = self._aggregate_metrics(data)
            summary_path = out_dir / f"{ds_name}_summary.json"
            save_json(summary_path, metrics)
            logger.info(f"{ds_name} summary → {summary_path.as_posix()}")

    def run(self):
        for cue in self.continuation_cues:
            cue_tag = self._sanitize_cue(cue)
            logger.info(f"=== Regenerating with cue: {cue} (tag={cue_tag}) ===")
            self._run_single(cue, cue_tag)
        all_metrics = {}
        for cue in self.continuation_cues:
            cue_tag = self._sanitize_cue(cue)
            cue_summary_files = (self.out_dir / cue_tag).glob("*_summary.json")
            for f in cue_summary_files:
                ds_name = f.stem.replace("_summary", "")
                key = f"{ds_name}/{cue_tag}"
                all_metrics[key] = load_json(f)
        save_json(self.out_dir / "all_cues_summary.json", all_metrics)
        logger.info("Regeneration completed.")

    def _sanitize_cue(self, cue: str) -> str:
        text = re.sub(r"\s+", "_", cue) 
        text = re.sub(r"[<>/\\\\\"']", "", text)
        text = text[:32] if len(text) > 32 else text
        if not text or len(set(text)) < 3:
            text = hashlib.md5(cue.encode()).hexdigest()[:8]
        return text

    @staticmethod
    def _iter_items(d: Union[Dict, List]):
        return d.values() if isinstance(d, dict) else d

    def _build_prompt(self, item: Dict, idx: int, cue: str) -> str:
        msgs = item.get("input_conversation", [])
        trimmed = item["trimmed_responses"][idx]["content"].rstrip()
        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return prompt + f"{trimmed} {cue}"

    def _get_handler(self, name: str):
        if name not in self.handlers:
            params = self.eval_meta.get(name, {})
            self.handlers[name] = load_dataset_handler(name, self.cfg, params)
        return self.handlers[name]

    def _aggregate_metrics(self, data) -> Dict[str, float]:
        num_items = num_resp = 0
        acc_r_avg = 0
        tot_orig_tok = tot_trim_tok = tot_regen_tok = 0

        for it in self._iter_items(data):
            num_items += 1
            for r in it["regen_responses"]:
                num_resp += 1
                acc_r_avg += r["correctness"]
                tok_stats = r["token_stats"]
                tot_orig_tok  += tok_stats["orig"]
                tot_trim_tok  += tok_stats["trim"]
                tot_regen_tok += tok_stats["regen"]

        comp_avg_pct = (tot_orig_tok - tot_regen_tok) / tot_orig_tok * 100 if tot_orig_tok else 0

        to_pct = lambda x, d: x / d * 100 if d else 0
        return {
            "regen_accuracy_avg": to_pct(acc_r_avg, num_resp),
            "avg_compression_%_avg": comp_avg_pct,
            "avg_tokens_orig":  tot_orig_tok / num_resp,
            "avg_tokens_trim":  tot_trim_tok / num_resp,
            "avg_tokens_regen": tot_regen_tok / num_resp,
            "num_items": num_items,
            "num_responses": num_resp,
        }