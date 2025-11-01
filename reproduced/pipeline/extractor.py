from __future__ import annotations

from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class HiddenStateExtractor:
    def __init__(self, model_name: str,
                 layer_idx: int = -1,
                 module_path: str | None = None,
                 device: str = "cuda"):
        import pdb; pdb.set_trace()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = (AutoModelForCausalLM
                      .from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
                      .eval())
        
        self.device = device
        self.layer_idx   = layer_idx
        self.module_path = module_path
        
        self._hook_handle = None
        
        if module_path:
            self._last_hidden = None
            blk = self._get_block(layer_idx)
            sub = self._get_submodule(blk, module_path)
            self._hook_handle = sub.register_forward_hook(self._hook_fn)

    # ---------- submodule ----------
    def _get_block(self, idx):
        n_blocks = len(self.model.model.layers)
        if idx < 0: idx += n_blocks
        return self.model.model.layers[idx]

    def _get_submodule(self, block, path: str):
        mod = block
        for attr in path.split('.'):
            mod = getattr(mod, attr)
        return mod

    def _hook_fn(self, module, inp, out):
        # out shape: [bs, seq_len, hidden_dim]
        self._last_hidden = out.detach()

    def extract_sentence_states(self,
                                full_text: str,
                                sentences: List[str],
                                prefix_len: int = 0,
                                return_indices: bool=False) -> List[torch.Tensor]:
                            
        inputs = self.tokenizer(full_text, return_tensors="pt",
                                truncation=False, add_special_tokens=False).to(self.device)

        if self.module_path is None:
            with torch.no_grad():
                outs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            h = outs.hidden_states[self.layer_idx][0].detach().to("cpu")        # [seq, dim]
        else:
            self._last_hidden = None
            with torch.no_grad():
                _ = self.model(**inputs)
            h = self._last_hidden[0]                         # [seq, dim]
            if h is None:
                raise RuntimeError("hook disconnected; check module_path")
        offset = prefix_len

        indices  = []         
        for s in sentences:
            toks = self.tokenizer(s, add_special_tokens=False).input_ids
            if not toks:
                indices.append(None)
                continue
            idx = offset + len(toks) - 1
            indices.append(min(idx, h.size(0) - 1))            
            offset += len(toks)

        states = [
            h[i].cpu() if i is not None else None
            for i in indices
        ]
        input_ids = inputs["input_ids"][0]  # [seq_len]
        for i, sent in zip(indices, sentences):
            if i is None:
                continue
            tok_id = input_ids[i].item()
            decoded_tok = self.tokenizer.decode([tok_id], skip_special_tokens=False)
            logger.debug(f"[{sent}] ends at token {i}: '{decoded_tok}'")
        logger.debug(f"len toks: {len(input_ids)}")
        logger.debug(f"[Debugging for hidden state of repeat sentence] len states: {len(states)}")
        return (states, indices) if return_indices else states

    def __del__(self):
        if self._hook_handle: self._hook_handle.remove()