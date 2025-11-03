import logging
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np

from wscgen.pipeline.extractor import HiddenStateExtractor
from wscgen.pipeline.collector import TrainingDataCollector
from wscgen.pipeline.builder import ProbeDatasetBuilder
from wscgen.pipeline.config import PipelineConfig
from wscgen.prober import build_prober
from wscgen.pipeline.labeler import build_labeler
from wscgen.pipeline.utils import save_json
import wandb
import random
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

def _parse_ratio(ratio: str) -> tuple[float, float]:
    ratio = ratio.strip()
    try:
        pos_str, neg_str = ratio.split(":")
        pos_w, neg_w = float(pos_str), float(neg_str)
    except ValueError:
        raise ValueError(
            f"Invalid ratio '{ratio}'. Expected format 'pos:neg', e.g. '1:4'."
        )

    if pos_w <= 0 or neg_w <= 0:
        raise ValueError("Both parts of the ratio must be positive numbers.")

    return pos_w, neg_w


def _balance_indices(y: np.ndarray, ratio: str, rng: np.random.RandomState) -> np.ndarray:
    """Return a boolean mask selecting samples such that the class balance
    approximates the requested *pos:neg* ratio.

    The algorithm keeps all of the minority class and downsamples the majority
    class to meet the target ratio.
    """
    pos_w, neg_w = _parse_ratio(ratio)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        logger.warning("[balance] Only one class present – skipping re‑balancing")
        full_mask = np.zeros_like(y, dtype=bool)
        full_mask[:] = True
        return full_mask

    # pick the minority class as anchor
    if len(pos_idx) <= len(neg_idx):
        keep_pos = pos_idx
        target_neg = int(len(keep_pos) * neg_w / pos_w)
        rng.shuffle(neg_idx)
        keep_neg = neg_idx[:min(target_neg, len(neg_idx))]
    else:
        keep_neg = neg_idx
        target_pos = int(len(keep_neg) * pos_w / neg_w)
        rng.shuffle(pos_idx)
        keep_pos = pos_idx[:min(target_pos, len(pos_idx))]

    keep = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(keep)
    mask = np.zeros_like(y, dtype=bool)
    mask[keep] = True
    logger.info("[balance] after re‑balancing: pos=%d, neg=%d", mask[y == 1].sum(), mask[y == 0].sum())
    return mask


class ProberTrainingPipeline:
    def __init__(self, cfg: PipelineConfig):
            self.cfg = cfg
            self.rng = np.random.RandomState(cfg.seed)

            ## extend to support multiple datasets
            if cfg.train_mix:
                self.collector = TrainingDataCollector(cfg, random.Random(cfg.seed))
            else:
                raise ValueError("`train_mix` must be provided in the config.")
            self.extract = HiddenStateExtractor(cfg.model_name, cfg.layer_idx, cfg.module_path)

            self.labeler = build_labeler(cfg, tokenizer=self.extract.tokenizer)
            self.builder  = ProbeDatasetBuilder(self.extract, self.labeler, cfg)
            # Include seed in prober_params if not already present
            prober_params = cfg.prober_params.copy()
            self.prober = build_prober(cfg.prober_name, **prober_params)
            self.out_dir = Path(cfg.make_save_dir()) / "prober_train"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            cfg.to_json(self.out_dir / "config.json")
            wandb.login(key=cfg.wandb_key, relogin=True)
            self.wb_run = wandb.init(
                project = cfg.wandb_project,
                name    = cfg.wandb_run_name,
                config  = cfg.__dict__,
                
            )

    @staticmethod
    def _metric_dict(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy":        float(accuracy_score(y_true, y_pred)),
            "auroc":           float(roc_auc_score(y_true, y_prob)),
            "n_pos":           int(y_true.sum()),
            "n_neg":           int((1 - y_true).sum()),
        }
        cls_rep = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        for lbl in ("0", "1"):
            for k in ("precision", "recall", "f1-score"):
                metrics[f"class_{lbl}_{k}"] = cls_rep[lbl][k]
        return metrics

    def run(self):
        ## check whether the probe.pkl is exist. If exist, skip.
        if (self.out_dir / "probe.pkl").exists():
            logger.info("Probe already exists. Skipping training.")
            return
        train_dict, val_dict = self.collector.collect()

        save_json(self.out_dir / "train_split.json", train_dict)
        if val_dict:
            save_json(self.out_dir / "val_split.json",   val_dict)

        logger.info("Building features …")
        X_tr, y_tr = self.builder.build(train_dict)
        if val_dict:
            X_val, y_val = self.builder.build(val_dict)

        if self.cfg.pos_neg_ratio:
            mask = _balance_indices(y_tr, self.cfg.pos_neg_ratio, self.rng)
            X_tr, y_tr = X_tr[mask], y_tr[mask]

        logger.info("Training probe on %d samples (pos=%d, neg=%d)", len(y_tr), y_tr.sum(), len(y_tr) - y_tr.sum())

        self.prober.fit(X_tr, y_tr, self.wb_run)
        probe_path = self.out_dir / "probe.pkl"
        self.prober.save(probe_path)
        logger.info(f"save prober to {probe_path} successfully")

        # ----- metrics -----
        train_prob = self.prober.predict_proba(X_tr)
        train_metrics = self._metric_dict(y_tr, train_prob)
        save_json(self.out_dir / "train_metrics.json", train_metrics)
        if val_dict:
            val_prob   = self.prober.predict_proba(X_val)
            val_metrics   = self._metric_dict(y_val,  val_prob)
            save_json(self.out_dir / "val_metrics.json",   val_metrics)
            self.wb_run.log({f"val/{k}":   v for k, v in val_metrics.items()})

        self.wb_run.log({f"train/{k}": v for k, v in train_metrics.items()})

        self.wb_run.finish()
        return self.prober