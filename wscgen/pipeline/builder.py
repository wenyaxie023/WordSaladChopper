from typing import Dict, Tuple
import numpy as np
import logging
from tqdm import tqdm
import torch
from .utils import save_json
from .extractor import HiddenStateExtractor
from .labeler import SentenceLabeler

logger = logging.getLogger(__name__)

class ProbeDatasetBuilder:
    """
    Builds a dataset of probe features and labels.

    Parameters
    ----------
    extractor : HiddenStateExtractor
        The extractor to use for extracting hidden states.
    labeler : SentenceLabeler
        The labeler to use for labeling sentences.
    cfg : dict
        The configuration for the builder.
    """
    def __init__(self, extractor: HiddenStateExtractor, labeler: SentenceLabeler, cfg):
        self.ext = extractor
        self.lab = labeler
        self.tok = extractor.tokenizer  
        self.min_token_len = cfg.min_token_len
        self.cfg = cfg
    def apply_template(self, conv: list) -> str:
        return self.tok.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def _token_len(self, chunk: str) -> int:
        if self.tok is not None:
            return len(self.tok.encode(chunk, add_special_tokens=False))
        return len(chunk.split())
    
    def build(
        self,
        data_dict: Dict,
        return_inputs: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, list | None]:
        feats, labels, sent_inputs = [], [], []
        items = list(data_dict.values())
        logger.info(f"[INFO] building features for {len(items)} items")

        answers = [it["responses"][0]["content"] for it in items]


        all_lab_sents = self.lab.label_texts(answers)  # List[List[(sent, lbl)]]

        debug_samples = [] 
        for idx, (item, lab_sents) in tqdm(enumerate(zip(items, all_lab_sents))):
            prompt   = self.apply_template(item["input_conversation"])

            answer   = answers[idx]
            full_txt = prompt + answer
            prefix_len = len(self.tok(prompt, add_special_tokens=False).input_ids)
            
            sents, y, sims = zip(*lab_sents)

            states, tok_idx = self.ext.extract_sentence_states(
                full_txt, list(sents), prefix_len=prefix_len, return_indices=True
            )

            sample_dbg = {"sample_id": idx, "chunks": []}
            sample_dbg['prompt'] = prompt
            sample_dbg['prefix_len'] = prefix_len
            for (s, lbl, sim), vec, ti in zip(lab_sents, states, tok_idx):
                if self._token_len(s) <= self.min_token_len:
                    continue
                if vec is None:
                    continue
                vec_fp32 = vec.detach().to(torch.float32).cpu()
                feats.append(vec_fp32)
                labels.append(lbl)
                sample_dbg["chunks"].append(
                    {
                        "text": s,
                        "label": lbl,
                        "similarity": round(sim, 4),
                        "token_idx": int(ti) if ti is not None else None,
                        "hidden_state100": vec[:100].cpu().tolist(),
                    }
                )
                if return_inputs:
                    sent_inputs.append(s)
            debug_samples.append(sample_dbg)
        feats_arr  = np.asarray(feats, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=int)
        return (feats_arr, labels_arr, sent_inputs) if return_inputs else (feats_arr, labels_arr)