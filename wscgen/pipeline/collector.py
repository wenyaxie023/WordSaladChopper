import math, random
from typing import Dict, List
from wscgen.pipeline.config import PipelineConfig
from wscgen.pipeline.utils import load_json
import logging
from pathlib import Path
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class TrainingDataCollector:
    """Collects and subsamples training examples from a dataset based on specified ratios."""
    def __init__(self, cfg: PipelineConfig, rng: random.Random):
        if not cfg.train_mix:
            raise ValueError("`train_mix` must be provided in the config.")
        self.cfg, self.rng = cfg, rng

        self.handlers = {}

    def _sample_dict(self, ds: dict, k: int) -> dict:
        keys = sorted(ds.keys())
        subset_keys = self.rng.sample(keys, k)
        return {k: ds[k] for k in subset_keys}

    def _split(self, ds: Dict, val_ratio: float) -> Tuple[Dict, Dict]:
        """split and shuffle for single data source"""
        keys = list(ds.keys())
        self.rng.shuffle(keys)
        n_val = int(round(len(keys) * val_ratio))
        val_keys   = keys[:n_val]
        train_keys = keys[n_val:]
        tr = {k: ds[k] for k in train_keys}
        va = {k: ds[k] for k in val_keys}
        return tr, va

    def collect(self) -> Tuple[Dict, Dict]:
        tr_all, va_all = {}, {}
        split_needed   = (self.cfg.train_mode == "indomain" and self.cfg.val_ratio > 0)

        for ds_name, details in sorted(self.cfg.train_mix.items()):
            path  = Path(details["path"])
            ratio = details.get("ratio", 1.0)
            raw   = load_json(path)
            k     = int(math.ceil(len(raw) * ratio))
            sampled = self._sample_dict(raw, k)

            if split_needed:
                tr_part, va_part = self._split(sampled, self.cfg.val_ratio)
            else:
                tr_part, va_part = sampled, {}

            for tgt, part in (("train", tr_part), ("val", va_part)):
                for orig_id, ex in part.items():
                    ex["__dataset__"] = ds_name
                    gid = f"{ds_name}#{orig_id}"
                    (tr_all if tgt == "train" else va_all)[gid] = ex

            logger.info(f"  [{ds_name}] sample={k}  → train={len(tr_part)}  val={len(va_part)}")


        logger.info(f"TOTAL → train={len(tr_all)}  val={len(va_all)}")
        return tr_all, va_all

    def build_prompts(self, problems: List[dict]) -> List[str]:
        """Build prompts for each problem, respecting their source dataset"""
        prompts = []
        for problem in problems:
            dataset_name = problem.get("__dataset__")
            if dataset_name and dataset_name in self.handlers:
                handler = self.handlers[dataset_name]
                prompts.append(handler.build_prompt(problem))
            else:
                raise ValueError(f"Cannot find handler for problem: {problem}")
        return prompts
