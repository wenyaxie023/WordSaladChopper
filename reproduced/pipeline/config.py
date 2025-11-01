from __future__ import annotations
import logging
import json
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

from types import SimpleNamespace
import os
@dataclass
class PipelineConfig:
    # -------------- Run Mode ---------------------------------
    run_mode: str
    # -------------- Model & Data ---------------------------------
    model_name: str  # HF id or local path
    train_mix: Dict[str, Dict[str, Any]] | None = None  # dataset_name -> sampling ratio
    training_file: str | None = None  # for on‑the‑fly
    run_tag: str | None = None

    # -------------- Labeler ---------------------------------------
    labeler_type: str = "semantic"  # "rule" | "semantic"
    labeler_params: Dict[str, Any] = field(default_factory=dict)

    # -------------- Extractor -------------------------------------
    layer_idx: int = -1
    module_path: str | None = None

    # -------------- Prober ----------------------------------------
    prober_name: str = "logistic"
    prober_params: Dict[str, Any] = field(default_factory=dict)  #     
    pos_neg_ratio: str | None = None  # e.g. "1:1"

    # -------------- Builder ---------------------------------------
    min_token_len: int = 10

    # -------------- Training / split ------------------------------
    train_mode: str = "crossdomain"  # "indomain" | "crossdomain"
    val_ratio: float = 0.2
    ckpt_dir: str | None = None

    # -------------- Evaluation ------------------------------------
    eval_datasets: Dict[str, Dict[str, Any]] | None = None  # name -> meta
    probe_path: str | None = None

    # -------------- Trimmer ---------------------------------------
    trimmer_type: str = "streak_filter_rollback"
    trimmer_params: Dict[str, Any] = field(default_factory=dict) # repeat time / threshold

    # -------------- Generation (regen) -----------------------------
    backend: str = "vllm"
    tp_size: int = 4
    gen_params: Dict[str, Any] = field(default_factory=dict)  # temp, top_p, …
    continuation_cues: list[str] = field(
        default_factory=lambda: ["I can find a clearer solution if I focus on the core problem."]
    )
    batch_size:  int   = -1

    # -------------- Misc ------------------------------------------
    seed: int = 41
    root_dir: str = "./artifacts"
    debug_dump: bool = True
    override: bool = False

    # -------------- Logging ----------------------------------------
    wandb_enabled: bool = True
    wandb_project: str = "probe_training"
    wandb_run_name: str = "default_run"
    wandb_key: str = os.getenv("WANDB_API_KEY", "")

    # ------------------------------------------------------------------
    # utilities & helpers
    # ------------------------------------------------------------------
    def default_probe_path(self) -> Path:
        print(f"default_probe_path:{self.make_save_dir()}")
        return self.make_save_dir() / "prober_train" / "probe.pkl"

    @staticmethod
    def _slug(s: str, keep: int = None) -> str:
        return s.split("/")[-1].replace(".", "-")[:keep]

    def _hash(self, obj: Any, n: int = 6) -> str:
        return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:n]
    
    def _make_training_data_tag(self) -> str:
        parts = []
        for name, meta in sorted(self.train_mix.items()):
            ratio = meta.get("ratio", "")
            temp  = meta.get("temp", None)
            tag = f"{self._slug(name)}"
            if temp is not None:
                tag += f"t{str(temp).replace('.', 'p')}"
            tag += f"-{str(ratio).replace('.', 'p')}"
            parts.append(tag)
        return "-".join(parts) if parts else "unknown_data"

    # ------------------------- labeler tag -------------------------------
    def _make_labeler_tag(self) -> str:
        if self.labeler_type == "rule":
            return "lbl_rule"
        if self.labeler_type == "semantic":
            embed = self._slug(self.labeler_params.get("embed_model", "mpnet"))
            sim = str(self.labeler_params.get("sim_threshold", 0.9)).replace(".", "p")
            return f"lbl_sem-{embed}-{sim}"

        return f"lbl_{self.labeler_type}"
    
    def _make_extractor_tag(self) -> str:
        if not self.module_path:
            module_path_str = "None"
        else:
            module_path_str = self._slug(self.module_path)
        return f"ext-{self.layer_idx}-{module_path_str}"

    # ------------------------- prober tag -------------------------------
    def _make_prober_tag(self) -> str:
        if not self.prober_params:
            return self.prober_name.lower()
        return f"{self.prober_name.lower()}-{self._hash(self.prober_params, 4)}"

    # ------------------------- prober tag -------------------------------

    def _make_trimmer_tag(self) -> str:
        base = self.trimmer_type.lower()
        if not self.trimmer_params:
            return base

        # 1) sort keys
        items = sorted(self.trimmer_params.items())  
        # 2) join with underscores
        param_str = "_".join(f"{k}-{v}" for k, v in items)  
        # 3) append a short hash so we don't end up with super-long names
        short_hash = self._hash(
                            self.trimmer_params, 
                            4)
        return f"{base}-{param_str}-{short_hash}"
    def _get_max_tokens(self) -> str:
        """Return max_tokens from self.gen_params (or 4096 if absent)."""
        max_token = self.gen_params.get("max_tokens", None)
        if max_token:
            return str(max_token)
        else:
            return "free"

    def _make_eval_data_tag(self) -> str:
        parts = []
        for name, meta in sorted(self.eval_datasets.items()):
            tag_parts = []
            for key, val in sorted(meta.items()):
                if key in ("dataset_id", "path"):
                    continue
                s = str(val).replace(".", "p")
                tag_parts.append(f"_{key}{s}")
            parts.append("".join(tag_parts))
        return "-".join(parts)

    # ------------------------- generation tag ---------------------------
    def _make_gen_tag(self) -> str | None:
        if not self.gen_params:
            return None
        base     = f"gen-{self._hash(self.gen_params)}"
        max_tok  = self._get_max_tokens()
        return f"{base}_{max_tok}"  

    def __post_init__(self):
        if self.eval_datasets and not isinstance(self.eval_datasets, dict):
            converted: Dict[str, Dict[str, Any]] = {}
            for d in self.eval_datasets:  # type: ignore[arg-type]
                if isinstance(d, dict):
                    for k, v in d.items():
                        converted[k] = v
                else:
                    raise ValueError(f"Invalid eval_datasets entry: {d!r}")
            self.eval_datasets = converted
        
        mdl = self._slug(self.model_name)
        label = self._make_labeler_tag()
        prober = self._make_prober_tag()
        data_tag = self._make_training_data_tag()

        if self.probe_path is None:
            self.probe_path = self.default_probe_path()

        extractor_tag = self._make_extractor_tag()
        # experiment hash
        sig = {
            "seed": self.seed,
            "train_mode": self.train_mode,
            "labeler": self.labeler_type,
            "lparams": self.labeler_params,
            "prober": self.prober_name,
            "pparams": self.prober_params,
        }
        exp = self._hash(sig)
        # assemble run name
        if self.run_mode == 'eval' or self.run_mode == 'gen':
            self.wandb_enabled = False
        self.wandb_run_name = f"{mdl}-{data_tag}-{extractor_tag}-{label}-{prober}-{exp}"

    # ------------------------- save‑dir ---------------------------------
    def make_save_dir(self, *, include_eval: bool = True, include_gen: bool = True) -> Path:
        """Hierarchical path encoding the *essentials* of the experiment."""
        mdl_tag = self._slug(self.model_name)

        dataset_tag = self._make_training_data_tag()

        extractor_tag = self._make_extractor_tag()

        # ---- labeler / prober ----
        labeler_tag = self._make_labeler_tag()
        prober_tag = self._make_prober_tag()
        builder_tag = f"fl_{str(self.min_token_len)}"

        # ---- ratio tag (optional) ----
        ratio_tag = None
        if self.pos_neg_ratio:
            pos, neg = self.pos_neg_ratio.split(":")
            ratio_tag = f"pos{pos}_neg{neg}"

        # ---- run‑tag (hash) ----
        exp_tag = None
        if self.run_tag:
            exp_tag = self.run_tag  
        else:
            sig = {
                "seed": self.seed,
                "train_mode": self.train_mode,
                "labeler": self.labeler_type,
                "lparams": self.labeler_params,
                "prober": self.prober_name,
                "pparams": self.prober_params,
                "extractor": extractor_tag,
                "training_data": dataset_tag,
                "builder_tag": builder_tag
            }
            exp_tag = self._hash(sig)

        # ---- assemble path ----
        parts = [self.root_dir, mdl_tag, dataset_tag, labeler_tag, extractor_tag, prober_tag, builder_tag]
        if ratio_tag:
            parts.append(ratio_tag)
        parts.append(exp_tag)

        base_path = Path(*parts)

        base_path.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = str(base_path / "prober_train" / "ckpt")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        return base_path

    def make_trim_dir(self) -> Path:
        base_dir = self.make_save_dir()
        eval_tag = self._make_eval_data_tag() if self.eval_datasets else "noeval"
        trim_tag = f"{self._make_trimmer_tag()}-{eval_tag}"
        trim_dir = base_dir / "eval_trim" / trim_tag
        trim_dir.mkdir(parents=True, exist_ok=True)
        return trim_dir

    # ------------------------- gen-dir -----------------------------------
    def make_gen_dir(self) -> Path:

        #  ---- tags -------------------------------------------------------
        eval_tag   = self._make_eval_data_tag() if self.eval_datasets else "noeval"
        base_dir = self.make_trim_dir()
        gen_tag    = f"{self._make_gen_tag()}-{eval_tag}"

        gen_dir    = base_dir / gen_tag
        gen_dir.mkdir(parents=True, exist_ok=True)

        # dump config for reproducibility
        try:
            self.to_json(gen_dir / "config.json")
        except Exception as e:
            logger.warning(f"Failed to dump config.json to {gen_dir}: {e}")

        return gen_dir


    def to_json(self, fp: str | Path):
        cfg_dict = asdict(self)
        cfg_dict["timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = Path(fp)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(cfg_dict, indent=2, default=str))
        logger.info(f"Config saved to {fp}")