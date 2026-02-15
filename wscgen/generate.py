from dataclasses import dataclass, field
from typing import Any, List, Tuple
import time

import torch
from transformers import (
    StoppingCriteria,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import logging

logger = logging.getLogger(__name__)

class DoubleNewlineCriteria(StoppingCriteria):
    """Stop when the most-recent token ∈ `newline_token_ids`."""

    def __init__(self, newline_token_ids: List[int]) -> None:
        self._ids = set(map(int, newline_token_ids))

    def __call__(self, input_ids: torch.Tensor, scores, **kwargs) -> bool:
        return int(input_ids[0, -1]) in self._ids


@dataclass
class GenState:
    context_ids: torch.Tensor
    remaining_token_budget: int
    total_used_tokens: int = -1
    rescue_time: int = 0
    full_texts: list[str] = field(default_factory=list)
    kept_texts: list[str] = field(default_factory=list)
    full_scores: list[float] = field(default_factory=list)

    def rebuild(
        self,
        tokenizer,
        *pieces: str,
        device: torch.device | None = None,
    ) -> None:
        ids = [
            tokenizer.encode(p, add_special_tokens=False, return_tensors="pt")
            for p in pieces
        ]
        
        self.context_ids = torch.cat(ids, dim=-1).to(device or self.context_ids.device)


@dataclass
class DecodeCache:
    past_key_values: Any
    next_logits: torch.Tensor


def _prefill_cache(
    model: AutoModelForCausalLM,
    context_ids: torch.Tensor,
) -> DecodeCache:
    out = model(
        input_ids=context_ids,
        use_cache=True,
        return_dict=True,
    )
    return DecodeCache(
        past_key_values=out.past_key_values,
        next_logits=out.logits[:, -1, :],
    )


def _apply_sampling_filters(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False

        remove_mask = torch.zeros_like(logits, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        logits = logits.masked_fill(remove_mask, float("-inf"))

    return logits


def _sample_next_token(
    logits: torch.Tensor,
    gen_cfg: dict[str, Any] | None = None,
) -> torch.Tensor:
    cfg = gen_cfg or {}
    do_sample = bool(cfg.get("do_sample", False))
    temperature = float(cfg.get("temperature", 1.0))
    top_p = max(0.0, min(1.0, float(cfg.get("top_p", 1.0))))
    top_k = int(cfg.get("top_k", 0))

    if (not do_sample) or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = logits / temperature
    filtered = _apply_sampling_filters(filtered, top_k=top_k, top_p=top_p)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _is_eos_token(token_id: int, eos_token_id: int | list[int] | tuple[int, ...] | None) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, (list, tuple)):
        return token_id in eos_token_id
    return token_id == int(eos_token_id)


def _generate_step_cached(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state: GenState,
    cache: DecodeCache,
    *,
    stop_on_newline: bool,
    newline_token_ids: set[int],
    gen_cfg: dict[str, Any] | None = None,
) -> Tuple[str, int, torch.Tensor | None, bool]:
    new_ids: list[int] = []
    last_hid: torch.Tensor | None = None

    while len(new_ids) < state.remaining_token_budget:
        next_token = _sample_next_token(cache.next_logits, gen_cfg)
        token_id = int(next_token.item())
        new_ids.append(token_id)

        step_out = model(
            input_ids=next_token,
            past_key_values=cache.past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        cache.past_key_values = step_out.past_key_values
        cache.next_logits = step_out.logits[:, -1, :]
        last_hid = step_out.hidden_states[-2][0, -1]

        if _is_eos_token(token_id, tokenizer.eos_token_id):
            break
        if stop_on_newline and token_id in newline_token_ids:
            break

    if not new_ids:
        return "", 0, None, True

    used_tokens = len(new_ids)
    should_stop = (
        used_tokens == state.remaining_token_budget
        or _is_eos_token(new_ids[-1], tokenizer.eos_token_id)
    )

    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return new_text, used_tokens, last_hid, should_stop

def wsc_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    chopper: "Chopper",
    *,
    newline_token_ids: List[int] | None = None,
    gen_cfg: dict[str, Any] | None = None,
    rescue_prompt: str = "Let's",
    token_budget: int = 128,
    rescue_budget: int = 128,
    max_rescues: int = 1,
) -> str:
    """
    On-the-fly generation with hidden-state classified + rescue
    """

    logger.debug(f"------------prompt in wsc_generate------------")
    logger.debug(prompt)
    logger.debug(f"------------prompt in wsc_generate------------")

    gen_cfg = gen_cfg or {}
    supported_cfg_keys = {"do_sample", "temperature", "top_p", "top_k"}
    unsupported_cfg_keys = sorted(set(gen_cfg.keys()) - supported_cfg_keys)
    if unsupported_cfg_keys:
        logger.warning(
            "wsc_generate fast path ignores unsupported gen_cfg keys: %s",
            unsupported_cfg_keys,
        )
    newline_token_ids_set = set(map(int, newline_token_ids or []))

    state = GenState(
        context_ids=tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        .to(model.device)
        .input_ids,
        remaining_token_budget=token_budget,
    )
    t_start = time.perf_counter()
    prefill_seconds = 0.0
    decode_seconds = 0.0
    chop_seconds = 0.0
    rebuild_seconds = 0.0
    prefill_calls = 0

    t_prefill = time.perf_counter()
    cache = _prefill_cache(model, state.context_ids)
    prefill_seconds += time.perf_counter() - t_prefill
    prefill_calls += 1

    with torch.inference_mode():
        while state.rescue_time <= max_rescues and state.remaining_token_budget > 0:
            # 1. Generate one step
            in_rescue = state.rescue_time > 0
            t_decode = time.perf_counter()
            txt, used_tokens, hid, should_stop = _generate_step_cached(
                model,
                tokenizer,
                state,
                cache,
                stop_on_newline=(not in_rescue),
                newline_token_ids=newline_token_ids_set,
                gen_cfg=gen_cfg,
            )
            decode_seconds += time.perf_counter() - t_decode
            if used_tokens == 0:
                break
            
            logger.debug(f"txt: {txt}")
            logger.debug(f"used_tokens: {used_tokens}")
            logger.debug(f"should_stop: {should_stop}")
            logger.debug(f"state.kept_texts: {state.kept_texts}")
            logger.debug(f"state.full_texts: {state.full_texts}")
            logger.debug(f"state.full_scores: {state.full_scores}")
            logger.debug(f"state.rescue_time: {state.rescue_time}")
            logger.debug(f"state.remaining_token_budget: {state.remaining_token_budget}")
            logger.debug(f"state.total_used_tokens: {state.total_used_tokens}")
            ## If the generated text is complete or the token budget is exhausted, then stop
            state.full_texts.append(txt)
            state.total_used_tokens += used_tokens
            state.remaining_token_budget -= used_tokens
            if in_rescue:
                state.kept_texts.append(txt)
            if should_stop:
                break
            # 2. Chop the generated text
            t_chop = time.perf_counter()
            before = len(state.full_texts)
            state.kept_texts, state.full_scores = chopper.chop(
                state.full_texts, state.full_scores, hid, used_tokens
            )
            chop_seconds += time.perf_counter() - t_chop

            # 3. If chopped, then rescue
            if len(state.kept_texts) < before:
                logger.info(f"Rescuing...")
                state.rescue_time += 1

                ## Add the rescue prompt length to the total used tokens
                state.total_used_tokens += tokenizer.encode(rescue_prompt, add_special_tokens=False, return_tensors="pt").shape[1]
                state.remaining_token_budget = rescue_budget
                t_rebuild = time.perf_counter()
                if len(state.kept_texts) > 0:
                    state.rebuild(
                        tokenizer,
                        prompt,
                        "".join(state.kept_texts),
                        rescue_prompt,
                        device=model.device,
                    )
                else:
                    state.rebuild(
                        tokenizer,
                        prompt,
                        rescue_prompt,
                        device=model.device,
                    )
                rebuild_seconds += time.perf_counter() - t_rebuild
                t_prefill = time.perf_counter()
                cache = _prefill_cache(model, state.context_ids)
                prefill_seconds += time.perf_counter() - t_prefill
                prefill_calls += 1
            else:
                # No rollback happened, so we can continue decoding from current KV cache.
                pass
    total_seconds = time.perf_counter() - t_start
    gen_tokens = max(0, state.total_used_tokens)
    speed_toks_per_sec = gen_tokens / max(total_seconds, 1e-9)

    return {
        "response": "".join(state.kept_texts),
        "rescue_time": state.rescue_time,
        "total_used_tokens": state.total_used_tokens,
        "kept_texts": state.kept_texts,
        "full_scores": state.full_scores,
        "full_texts": state.full_texts,
        "timing": {
            "total_seconds": total_seconds,
            "decode_seconds": decode_seconds,
            "chop_seconds": chop_seconds,
            "rebuild_seconds": rebuild_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_calls": prefill_calls,
            "tokens_per_second": speed_toks_per_sec,
        },
    }
