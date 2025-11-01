from typing import Dict, Tuple
import numpy as np
import logging
from tqdm import tqdm
from pipeline.utils import save_json

from pipeline.extractor import HiddenStateExtractor
from pipeline.labeler import SentenceLabeler

logger = logging.getLogger(__name__)

class ProbeDatasetBuilder:
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

        """
        Convert dataset items into sentence-level features and labels.

        It labels each sentence in the answers, extracts embeddings for them,
        filters out short or invalid ones, and returns feature and label arrays.

        Args:
            data_dict (Dict): Input data containing conversations and responses.
            return_inputs (bool): If True, also return the sentence texts used.

        Returns:
            Tuple[np.ndarray, np.ndarray] or (np.ndarray, np.ndarray, list):
                Sentence feature vectors, labels, and optionally the sentence texts.
        """
        feats, labels, sent_inputs = [], [], []
        items = list(data_dict.values())
        logger.info(f"[INFO] building features for {len(items)} items")

        answers = [it["responses"][0]["content"] for it in items]

        # label the chunks in the answers
        all_lab_sents = self.lab.label_texts(answers)  # List[List[(sent, lbl)]]
        debug_samples = [] 

        # extract the hidden states of the chunks
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
                feats.append(vec.cpu().numpy())
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

        feats_arr  = np.asarray(feats)
        labels_arr = np.asarray(labels, dtype=int)
        if self.cfg.debug_dump:
            dump_path = self.cfg.make_save_dir() / "prober_train" / "train_debug.json"
            save_json(dump_path, debug_samples[:20])
        return (feats_arr, labels_arr, sent_inputs) if return_inputs else (feats_arr, labels_arr)