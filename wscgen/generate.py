from dataclasses import dataclass, field
from typing import Any, List, Tuple

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

def _generate_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    state: GenState,
    criterion: StoppingCriteria | None,
    gen_cfg: dict[str, Any] = None,
) -> Tuple[str, int, torch.Tensor, bool]:
    """
    Generate one step of text until the stopping criterion is met.

    Parameters
    ----------
    model : AutoModelForCausalLM
        The model to generate text from.
    tokenizer : AutoTokenizer
        The tokenizer to use for decoding the generated text.
    state : GenState
        The current generation state.
    criterion : StoppingCriteria
        The stopping criterion to use.
    gen_cfg : dict[str, Any]
        The generation configuration.

    Returns
    -------
    Tuple[str, torch.Tensor, int]
        The generated text, used tokens, the hidden state of the last token, and whether to stop.
    """
    out = model.generate(
        state.context_ids,
        stopping_criteria=[criterion] if criterion is not None else None,
        use_cache=True,
        return_dict_in_generate=False,
        max_new_tokens=state.remaining_token_budget,
        **gen_cfg,
    )
    prompt_len = state.context_ids.size(1)
    sequences = out                         # already a tensor
    new_ids = sequences[:, prompt_len:]
    new_ids = new_ids[0]

    with torch.inference_mode():
        full = model(
            input_ids=sequences,
            attention_mask=torch.ones_like(sequences),
            output_hidden_states=True,
            use_cache=True,
        )

    last_hid = full.hidden_states[-2][0, -1]
    new_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    last_id = new_ids[-1].item() if new_ids.numel() else None
    should_stop = (
        new_ids.numel() == state.remaining_token_budget or
        last_id == tokenizer.eos_token_id
    )
    del out
    torch.cuda.empty_cache()
    del full
    torch.cuda.empty_cache()
    return new_text, new_ids.numel(), last_hid, should_stop

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

    crit = DoubleNewlineCriteria(newline_token_ids)

    state = GenState(
        context_ids=tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        .to(model.device)
        .input_ids,
        remaining_token_budget=token_budget,
    )

    with torch.inference_mode():
        while state.rescue_time <= max_rescues and state.remaining_token_budget > 0:
            # 1. Generate one step
            in_rescue = state.rescue_time > 0

            ## If in rescue, then no need to stop
            if in_rescue:
                crit = None
            
            txt, used_tokens, hid, should_stop = _generate_step(model, tokenizer, state, crit, gen_cfg)
            
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
            before = len(state.full_texts)
            state.kept_texts, state.full_scores = chopper.chop(
                state.full_texts, state.full_scores, hid, used_tokens
            )

            # 3. If chopped, then rescue
            if len(state.kept_texts) < before:
                logger.info(f"Rescuing...")
                state.rescue_time += 1

                ## Add the rescue prompt length to the total used tokens
                state.total_used_tokens += tokenizer.encode(rescue_prompt, add_special_tokens=False, return_tensors="pt").shape[1]
                state.remaining_token_budget = rescue_budget
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
            else:
                state.rebuild(
                    tokenizer,
                    prompt,
                    "".join(state.kept_texts),
                    device=model.device,
                )

    return {
        "response": "".join(state.kept_texts),
        "rescue_time": state.rescue_time,
        "total_used_tokens": state.total_used_tokens,
        "kept_texts": state.kept_texts,
        "full_scores": state.full_scores,
        "full_texts": state.full_texts,
    }
