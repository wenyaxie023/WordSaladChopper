# pipeline/pipelines/trim_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
from pipeline.labeler import build_labeler

import numpy as np

from pipeline.config import PipelineConfig
from pipeline.datasets import load_dataset_handler
from pipeline.extractor import HiddenStateExtractor
from pipeline.prober import build_prober
from pipeline.utils import save_json
from pipeline.trimmer import build_trimmer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

logger = logging.getLogger(__name__)
import re

class TrimEvalPipeline:
    def __init__(self, cfg: PipelineConfig,):
        self.cfg = cfg
        print(f"Load from {cfg.probe_path}")
        self.probe_path = Path(cfg.probe_path)

        self.out_dir = cfg.make_trim_dir()
        self.trimmer = build_trimmer(cfg.trimmer_type, cfg.trimmer_params)


        self.extractor = HiddenStateExtractor(cfg.model_name,
                                              cfg.layer_idx,
                                              cfg.module_path)
        self.labeler = build_labeler(cfg, tokenizer=self.extractor.tokenizer)
        
        self.eval_meta: Dict[str, dict] = {}
        if cfg.eval_datasets:
            for k, v in cfg.eval_datasets.items():
                self.eval_meta[k] = v
        self.handlers: Dict[str, any] = {}
        cfg.to_json(self.out_dir / "config.json")

        prober_params = cfg.prober_params.copy()
        self.prober = build_prober(cfg.prober_name, **prober_params)
        self.prober.load(self.probe_path)

    def _group_eval_datasets(self, eval_ds_dict: dict[str, dict]) -> dict[str, list[tuple[str, dict]]]:
        groups: dict[str, list[tuple[str, dict]]] = {}
        for ds_name, params in eval_ds_dict.items():
            # Use full_name if available, otherwise fall back to the original regex-based logic
            if "full_name" in params and params["full_name"]:
                base = params["full_name"]
            else:
                m = re.fullmatch(r"(.+?)_(\d+)$", ds_name)
                base = m.group(1) if m else ds_name
            groups.setdefault(base, []).append((ds_name, params))
        return groups


    def run(self):
        all_metrics = {}
        all_trimmed_data = {}
        dataset_groups = self._group_eval_datasets(self.cfg.eval_datasets)
        for base_name, parts in dataset_groups.items():
            summary_path  = self.out_dir / f"{base_name}_summary.json"
            trimmed_path  = self.out_dir / f"{base_name}_trimmed_results.json"
            if trimmed_path.exists() and not self.cfg.override:
                logger.info(f"[{base_name}] results already exist - skip.")
                continue

            combined_data: dict[str, Any] = {}
            y_true_ds, y_prob_ds = [], []

            # merge all datasets
            for ds_name, params in parts:
                handler = load_dataset_handler(ds_name, self.cfg, params)
                data = handler.ds
                logger.info(f"[{ds_name}] Loaded {len(data)} items")

                if isinstance(data, dict):
                    for k, v in data.items():
                        combined_data[f"{ds_name}#{k}"] = v
                else:  
                    for idx, v in enumerate(data):
                        combined_data[f"{ds_name}#{idx}"] = v

            # process each item
            for _, item in combined_data.items():
                labels, probs = self._process_item(item)
                y_true_ds.extend(labels)
                y_prob_ds.extend(probs.tolist())

            trim_metrics = self._aggregate_trim_metrics(data)
            cls_metrics  = self._classification_metrics(
                np.asarray(y_true_ds, dtype=int), np.asarray(y_prob_ds, dtype=float)
            )
            metrics = {**trim_metrics, **cls_metrics}

            all_metrics[ds_name] = metrics

            save_json(summary_path, metrics)
            save_json(trimmed_path, combined_data)
            all_trimmed_data.update(combined_data)

    def _process_item(self, item: Dict):
        """
        Trim all candidate responses for a single QA item.
        """
        labels_all: List[int] = []
        probs_all: List[float] = []
        trimmed_responses: List[dict] = []
        all_details: List[List[dict]] = []

        prompt = self.extractor.tokenizer.apply_chat_template(item["input_conversation"], tokenize=False, add_generation_prompt=True)
        for resp in item["responses"]:
            answer = resp["content"]

            # labels: use for metrics calculation, probs: use for trimming
            sentences, probs, labels, states = self._predict_repetition(prompt, answer)
            # trim the text
            trimmed_text = self.trimmer.trim(sentences, probs)

            trimmed_responses.append({
                "content": trimmed_text,
                "token_stats": self._token_stats(answer, trimmed_text),
            })

            details = [
                {"sentence": s, "pred": float(p), "label": int(l)}
                for s, p, l in zip(sentences, probs, labels)
            ]
            all_details.append(details)

            labels_all.extend(labels)
            probs_all.extend(probs.tolist())

        # attach results back to item
        item["trimmed_responses"] = trimmed_responses
        item["trimmed_responses_details"] = all_details 
        return labels_all, np.asarray(probs_all, dtype=float)


    def _predict_repetition(self, prompt: str, answer: str) -> Tuple[List[str], np.ndarray, List[int], np.ndarray]:
        """
        predict repetition probabilities for each sentence
        
        Returns:
            Tuple[List[str], np.ndarray, List[int], np.ndarray]: (sentences, probs, labels, states)
        """
        sent_lab_nested = self.labeler.label_texts([answer])
        sent_lab = sent_lab_nested[0]

        # sent_lab -> (chunk, label, similarity)
        sentences = [chunk for chunk, label, sim in sent_lab]
        labels    = [label for chunk, label, sim in sent_lab]

        prompt_len = len(self.extractor.tokenizer.encode(prompt, add_special_tokens=False))

        states = self.extractor.extract_sentence_states(prompt + answer, sentences, prefix_len=prompt_len)

        X = np.vstack([
            v if isinstance(v, np.ndarray) else v.numpy()
            for v in states
        ])
        probs = self.prober.predict_proba(X)
        return sentences, probs, labels, X

    def _token_stats(self, original: str, trimmed: str) -> Dict[str, float]:
        tok = self.extractor.tokenizer.encode
        ori_toks = len(tok(original, add_special_tokens=False))
        tri_toks = len(tok(trimmed,  add_special_tokens=False))
        ratio = 0.0 if ori_toks == 0 else max(0, (ori_toks - tri_toks) / ori_toks * 100)
        return {"orig": ori_toks, "trim": tri_toks, "compression_%": ratio}

    def _get_handler(self, name: str):
        if name not in self.handlers:
            params = self.eval_meta.get(name, {})
            self.handlers[name] = load_dataset_handler(name, self.cfg, params)
        return self.handlers[name]

    def _count_labels(self, text: str) -> tuple[int, int]:
        sent_lab = self.labeler.label_duplicates([text])[0]   # [(chunk, label, sim), ...]
        n_pos = sum(label for _, label, _ in sent_lab)
        n_neg = len(sent_lab) - n_pos
        return n_pos, n_neg

    def _find_boundary(self, labels: List[int], streak_len: int) -> int | None:
        for i in range(streak_len - 1, len(labels)):
            if all(labels[i - j] == 1 for j in range(streak_len)):
                return i - streak_len + 1
        return None

    def _count_tokens_by_label(self, text: str) -> tuple[int, int]:
        sent_lab = self.labeler.label_duplicates([text])[0]   # [(chunk, label, sim), ...]
        tok = self.extractor.tokenizer.encode                
        pos_tok = neg_tok = 0
        for chunk, label, _ in sent_lab:
            n_tok = len(tok(chunk, add_special_tokens=False))
            if label == 1:
                pos_tok += n_tok
            else:
                neg_tok += n_tok
        return pos_tok, neg_tok


    def _aggregate_trim_metrics(self, data) -> Dict[str, float]:
        tot_ori_tok_avg = tot_trim_tok_avg = 0  
        num_resp = 0 
        num_items = 0 

        trim_pos_total = 0 
        trim_neg_total = 0  
        ori_pos_total = 0  
        ori_neg_total = 0  
        
        trim_pos_tok_total = 0
        trim_neg_tok_total = 0
        ori_pos_tok_total  = 0
        ori_neg_tok_total  = 0
        
        streak_lens = (2, 10) 
        gold_counts = {
            k: dict(pre_pos=0, pre_neg=0, post_pos=0, post_neg=0) for k in streak_lens
        }

        for item in (data.values() if isinstance(data, dict) else data):
            num_items += 1
            assert len(item["responses"]) == len(item["trimmed_responses"])

            for orig_ans, trim_ans in zip(item["responses"], item["trimmed_responses"]):
                num_resp += 1

                stats = trim_ans["token_stats"]
                tot_ori_tok_avg += stats["orig"]
                tot_trim_tok_avg += stats["trim"]
                n_pos, n_neg = self._count_labels(trim_ans["content"])
                n_pos_ori, n_neg_ori = self._count_labels(orig_ans["content"])
                trim_pos_total += n_pos
                trim_neg_total += n_neg
                ori_pos_total += n_pos_ori
                ori_neg_total += n_neg_ori

                n_pos_tok_trim, n_neg_tok_trim = self._count_tokens_by_label(trim_ans["content"])
                n_pos_tok_ori,  n_neg_tok_ori  = self._count_tokens_by_label(orig_ans["content"])

                trim_pos_tok_total += n_pos_tok_trim
                trim_neg_tok_total += n_neg_tok_trim
                ori_pos_tok_total  += n_pos_tok_ori
                ori_neg_tok_total  += n_neg_tok_ori
                gt_labels = [
                    lbl for _, lbl, _ in
                    self.labeler.label_duplicates([orig_ans["content"]])[0]
                ]

                for k in streak_lens:
                    boundary = self._find_boundary(gt_labels, k)
                    if boundary is None:
                        pre_labels, post_labels = gt_labels, []
                    else:
                        pre_labels  = gt_labels[:boundary]
                        post_labels = gt_labels[boundary:]

                    gold_counts[k]["pre_pos"]  += sum(pre_labels)
                    gold_counts[k]["pre_neg"]  += len(pre_labels)  - sum(pre_labels)
                    gold_counts[k]["post_pos"] += sum(post_labels)
                    gold_counts[k]["post_neg"] += len(post_labels) - sum(post_labels)


        comp_avg = (
            (tot_ori_tok_avg - tot_trim_tok_avg) / tot_ori_tok_avg * 100 if tot_ori_tok_avg else 0.0
        )

        salad_shrink = trim_pos_total / ori_pos_total

        if (trim_pos_tok_total + trim_neg_tok_total) == 0:
            repetition_ratio_after_trim_tok = 0.0
        else:
            repetition_ratio_after_trim_tok = trim_pos_tok_total / (
                trim_pos_tok_total + trim_neg_tok_total
            )

        if (ori_pos_tok_total + ori_neg_tok_total) == 0:
            repetition_ratio_before_trim_tok = 0.0
        else:
            repetition_ratio_before_trim_tok = ori_pos_tok_total / (ori_pos_tok_total + ori_neg_tok_total)
        metrics = {
            "salad_shrink": salad_shrink,  
            "avg_compression_%_avg": comp_avg,  
            "num_items": num_items,  
            "num_responses": num_resp,  
            "cls_trim_n_pos": trim_pos_total,  
            "cls_trim_n_neg": trim_neg_total, 
            "repetition_ratio_before_trim_token": repetition_ratio_before_trim_tok,  
            "repetition_ratio_after_trim_token":  repetition_ratio_after_trim_tok,  
        }


        for k in streak_lens:
            pre_pos  = gold_counts[k]["pre_pos"]  
            pre_neg  = gold_counts[k]["pre_neg"]  
            post_pos = gold_counts[k]["post_pos"] 
            post_neg = gold_counts[k]["post_neg"] 

            pre_den  = pre_pos  + pre_neg  
            post_den = post_pos + post_neg 

            metrics[f"golden_pre_pos_ratio_{k}"]  = pre_pos  / pre_den  if pre_den  else 0.0
            metrics[f"golden_post_pos_ratio_{k}"] = post_pos / post_den if post_den else 0.0

        return metrics

    @staticmethod
    def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        if len(y_true) == 0:
            return {
                "cls_accuracy": 0.0,
                "cls_auroc": 0.0,
            }

        y_pred = (y_prob >= 0.5).astype(int)
        cls_n_pos = int(y_true.sum())
        cls_n_neg = int((1 - y_true).sum())
        metrics = {
            "cls_accuracy": float(accuracy_score(y_true, y_pred)),
            "cls_auroc": float(roc_auc_score(y_true, y_prob)),
            "cls_n_pos": cls_n_pos,
            "cls_n_neg": cls_n_neg,
        }
        
        cls_rep = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        for lbl in ("0", "1"):
            for k in ("precision", "recall", "f1-score"):
                metrics[f"cls_class_{lbl}_{k}"] = cls_rep[lbl][k]
        return metrics