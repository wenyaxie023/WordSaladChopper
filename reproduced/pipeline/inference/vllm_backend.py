from __future__ import annotations

import logging
from typing import Union, List, Any
# deepcopy
from copy import deepcopy


from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

from vllm import LLM, SamplingParams

import torch, gc, ray, logging


class VllmBackend:
    """Thin wrapper around vLLM's generate() API."""

    # --------------------- construction --------------------- #
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.seed = getattr(cfg, "seed", 41)

        if LLM is None:
            logger.warning("vLLM not installed – using mock completions.")
            self.llm = None
        else:
            logger.info("-" * 80)
            logger.info("initialising vLLM engine @ %s...", cfg.model_name)
            logger.info("-" * 80)

            self.llm = LLM(
                model=cfg.model_name,
                tensor_parallel_size=getattr(cfg, "tp_size", 4),
            )
            logger.info("vLLM engine initialised @ %s", cfg.model_name)

        if SamplingParams is None:
            self._default_sampling = None
        else:
            self._default_sampling = SamplingParams(
                temperature=getattr(cfg, "temperature", 0.0),
                top_p=getattr(cfg, "top_p", 1.0),
                max_tokens=getattr(cfg, "max_tokens", 32768),
                n=1,
                seed=self.seed,
            )
        logger.info(f"default sampling params: {self._default_sampling}")
        self.batch_size: int | None = getattr(cfg, "batch_size", None)

    def generate(
        self,
        prompts: Union[str, List[str]],
        **override_sampling: Any,
    ) -> Union[str, List[str]]:
        # logger.info(f"prompt sample 1:{prompts[0]}")
        for sample in prompts[:10]:
            logger.debug(f"prompt sample:{sample}")
        outs = self.generate_raw(prompts, **override_sampling)
        texts = [o.outputs[0].text for o in outs]
        return texts[0] if isinstance(prompts, str) else texts

    # ------------------------------------------------------- #
    def generate_raw(
        self,
        prompts: Union[str, List[str]],
        **override_sampling: Any,
    ):
        prompt_list: List[str] = _to_list(prompts)

        if self.llm is None:
            raise RuntimeError("vLLM is not installed")

        sparams = self._build_sampling_params(**override_sampling)

        bs = len(prompt_list) if self.batch_size in (None, -1) else self.batch_size
        logger.info(f"batch size: {bs}")

        results = []
        for i in range(0, len(prompt_list), bs):
            batch = prompt_list[i : i + bs]
            result = self.llm.generate(batch, sampling_params=sparams)
            
            results.extend(result)
        return results

    # -------------------- internals ------------------------ #
    def _build_sampling_params(self, **override) -> SamplingParams | None:
        if SamplingParams is None:
            return None

        if not override:
            return self._default_sampling

        params = deepcopy(self._default_sampling)
        for k, v in override.items():
            if hasattr(params, k):
                setattr(params, k, v)
            else:
                raise ValueError(f"SamplingParams has no attribute '{k}'")
        return params


def _to_list(x: Union[str, List[str]]) -> List[str]:
    return [x] if isinstance(x, str) else x
