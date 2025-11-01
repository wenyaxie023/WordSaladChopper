from __future__ import annotations

from typing import Dict, Type

from pipeline.config import PipelineConfig
from pipeline.datasets.base import TaskHandler
from pipeline.datasets.math500 import MathTaskHandler
from pipeline.datasets.gpqa_diamond import GPQADiamondTaskHandler
from pipeline.datasets.aime25 import AIMETaskHandler
from pipeline.datasets.gsm8k import GSM8KTaskHandler
from pipeline.datasets.livecodebench import LiveCodeBenchTaskHandler

_DATASET_REGISTRY: Dict[str, Type[TaskHandler]] = {
    "math500": MathTaskHandler,
    "gpqa_diamond": GPQADiamondTaskHandler,
    "aime25_1": AIMETaskHandler,
    "aime25_2": AIMETaskHandler,
    "aime25": AIMETaskHandler,
    "gsm8k": GSM8KTaskHandler,
}

def load_dataset_handler(name: str,
                         cfg,
                         params: dict | None = None) -> TaskHandler:
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name}")
    return _DATASET_REGISTRY[name](cfg, params)