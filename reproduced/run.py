from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

TRAINING_PATH_MAP = OmegaConf.load("configs/resources/dataset_paths.yaml")["dataset_paths"]
EVALUATION_DATA_PATH_MAP = OmegaConf.load("configs/resources/eval_dataset_paths.yaml")["eval_dataset_paths"]

def _ds_path(model_name: str, ds_id: str) -> str:
    """
    config in hydra:
        path: ${ds_path:${model.model_name},${.dataset_id}}
    """
    try:
        return TRAINING_PATH_MAP[model_name][ds_id]
    except KeyError as e:
        raise KeyError(
            f"[dataset_paths.yaml] cannot find <model={model_name}, dataset_id={ds_id}>"
        ) from e

OmegaConf.register_new_resolver("ds_path", _ds_path)

def _eds_path(model_name: str, ds_id: str) -> str:
    """
    config in hydra:
        path: ${ds_path:${model.model_name},${.dataset_id}}
    """
    try:
        return EVALUATION_DATA_PATH_MAP[model_name][ds_id]
    except KeyError as e:
        raise KeyError(
            f"[dataset_paths.yaml] cannot find <model={model_name}, dataset_id={ds_id}>"
        ) from e

OmegaConf.register_new_resolver("eds_path", _eds_path)

from pipeline.config import PipelineConfig
from pipeline.pipelines import (
    ProberTrainingPipeline,
    TrimEvalPipeline,
    ContinuationGenerationPipeline
)

def setup_logger(exp_dir: Path, phase: str) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = exp_dir / f"log_{phase}_{ts}.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="w")],
    )

def flatten_config(cfg: DictConfig) -> dict:
    flat_cfg = {}
    for k, v in OmegaConf.to_container(cfg, resolve=True, structured_config_mode=False).items():
        if isinstance(v, dict) and k not in ["labeler_params", "prober_params", "train_mix", "eval_datasets", "gen_params"]:
            flat_cfg.update(v)
        else:
            flat_cfg[k] = v
    return flat_cfg

def set_seed(seed: int = 41) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    default config: conf/config.yaml
    can be overridden by CLI
    """
    print(OmegaConf.to_yaml(cfg))
    resolved_cfg_dict = flatten_config(cfg) 
    pcfg = PipelineConfig(**resolved_cfg_dict)

    set_seed(pcfg.seed)

    exp_dir = pcfg.make_save_dir()
    setup_logger(exp_dir, cfg.run_mode)

    logging.info(f"==========  Resolved Config  ==========\n{OmegaConf.to_yaml(cfg)}")

    if cfg.run_mode == "train":
        ProberTrainingPipeline(pcfg).run()

    elif cfg.run_mode == "eval":
        TrimEvalPipeline(pcfg).run()

    elif cfg.run_mode == "gen":
        ContinuationGenerationPipeline(pcfg).run()

    else:
        raise ValueError(f"Unknown run_mode: {cfg.run_mode}")

if __name__ == "__main__":
    main()