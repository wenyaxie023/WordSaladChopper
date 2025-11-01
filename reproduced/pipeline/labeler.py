import json
import logging
import re
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

class SentenceLabeler:
    def __init__(
        self,
        labeler_type: str = "semantic",  # "rule" | "semantic"
        sim_threshold: float = 0.99,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        tokenizer: AutoTokenizer | None = None,
        slide_window: int = 2000,
        label_chunk: int = 10,
    ) -> None:
        self.labeler_type = labeler_type.lower()
        self.sim_threshold = sim_threshold
        self.tokenizer = tokenizer
        self.slide_window = slide_window
        self.label_chunk = label_chunk

        if self.labeler_type == "semantic":
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            logger.info(
                f"[SentenceLabeler] semantic mode – loading '{embed_model}' on {self.device}"
            )
            self.embedder = SentenceTransformer(embed_model, device=self.device)

    @staticmethod
    def split_text(text: str) -> List[str]:
        return [s for s in re.split(r"(?<=\n\n)", text) if s]

    def label_duplicates(
        self,
        texts: Sequence[str],
        batch_size_embed: int | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> List[List[Tuple[str, int, float]]]:
        
        if self.labeler_type == "rule":
            results = []
            for txt in texts:
                chunks = self.split_text(txt)
                seen = set()
                chunk_results = []
                for chunk in chunks:
                    is_duplicate = chunk in seen
                    chunk_results.append((chunk, int(is_duplicate), 1.0 if is_duplicate else 0.0))
                    seen.add(chunk)
                results.append(chunk_results)
            return results

        # ------------ semantic -----------------
        bs_embed = batch_size_embed or 256
        th = self.sim_threshold

        all_chunks, chunk_offsets = [], []      # offsets record (sample_idx, local_idx)
        for s_idx, txt in enumerate(texts):
            chs = [c for c in self.split_text(txt)]
            all_chunks.extend(chs)
            chunk_offsets.extend([(s_idx, i) for i in range(len(chs))])

        if not all_chunks:
            return [[] for _ in texts]

        embs = (
            self.embedder.encode(
                all_chunks,
                batch_size=bs_embed,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            .to(dtype)
            .cuda(non_blocking=True)
        )

        results: List[List[Tuple[str, int, float]]] = [[] for _ in texts]
        ptr = 0
        for s_idx, txt in enumerate(texts):
            n_chunks = sum(1 for off in chunk_offsets if off[0] == s_idx)
            if n_chunks == 0:
                continue
            sample_embs = embs[ptr : ptr + n_chunks]
            ptr += n_chunks

            max_prev_sim = torch.zeros(n_chunks, dtype=torch.float32, device=sample_embs.device)
            sims = torch.matmul(sample_embs, sample_embs.T).tril(diagonal=-1)
            max_prev_sim = sims.max(dim=1).values
            max_prev_sim[0] = 0.0

            labels = (max_prev_sim >= th).int().tolist()
            sims_cpu = max_prev_sim.cpu().tolist()
            chunks = self.split_text(txt)
            for c, lbl, sim_v in zip(chunks, labels, sims_cpu):
                results[s_idx].append((c, int(lbl), float(sim_v)))

        return results

    def _token_len(self, chunk: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(chunk, add_special_tokens=False))
        return len(chunk.split())
   
    def label_texts(
        self,
        texts: Sequence[str],
        batch_size_embed: int | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> List[List[Tuple[str, int, float]]]:
        """Label chunks inside each text using the configured strategy.

        Returns a list (per sample) of tuples ``(chunk, label, similarity)``.
        Label is **0** until the first occurrence of ``label_chunk`` consecutive
        chunks whose similarity scores (computed in a sliding window of size
        ``slide_window``) exceed ``sim_threshold``. From that boundary onward
        every chunk is labelled **1**. If no such boundary exists, all chunks
        are labelled **0**.
        """
        if self.labeler_type == "rule":
            return self._label_rule(texts)

        bs_embed = batch_size_embed or 256
        th = self.sim_threshold
        W = self.slide_window
        L = self.label_chunk

        # 1. collect chunks
        all_chunks, chunk_to_sample, sample_chunks = [], [], []
        for s_idx, txt in enumerate(texts):
            chs = [c for c in self.split_text(txt)]
            sample_chunks.append(chs)
            all_chunks.extend(chs)
            chunk_to_sample.extend([s_idx] * len(chs))

        N = len(all_chunks)
        if N == 0:
            return [[] for _ in texts]

        # 2. embed all chunks
        embs = self.embedder.encode(
            all_chunks,
            batch_size=bs_embed,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embs = embs.to(device=self.device, dtype=dtype, non_blocking=(self.device == "cuda"))

        # 3. iterate per sample to compute similarity + labels
        results: List[List[Tuple[str, int, float]]] = [[] for _ in texts]
        start_idx = 0
        for s_idx, chunks in enumerate(sample_chunks):
            n_chunks = len(chunks)
            if n_chunks == 0:
                continue

            sample_embs = embs[start_idx : start_idx + n_chunks]  # [n, d]
            sim_scores = torch.zeros(n_chunks, dtype=torch.float32, device=sample_embs.device)

            # compute similarity with sliding window
            for i in range(n_chunks):
                if i == 0:
                    sim_scores[i] = 0.0
                    continue
                j0 = max(0, i - W)
                prev_embs = sample_embs[j0:i]  # at most W previous chunks
                curr_emb = sample_embs[i : i + 1]  # keep 2‑D
                sim = torch.matmul(curr_emb.float(), prev_embs.float().T)[0]
                sim_scores[i] = sim.max()

            # find boundary
            boundary = None
            if n_chunks >= L:
                for i in range(L - 1, n_chunks): 
                    window = sim_scores[i - L + 1 : i + 1]
                    if torch.all(window >= th):
                        boundary = i - L + 1  # first idx of streak
                        break

            # assign labels 
            if boundary is None:
                labels = torch.zeros(n_chunks, dtype=torch.int8)
            else:
                labels = torch.zeros(n_chunks, dtype=torch.int8)
                labels[boundary:] = 1

            for chunk, lbl, sim_v in zip(chunks, labels.tolist(), sim_scores.cpu().tolist()):
                results[s_idx].append((chunk, int(lbl), float(sim_v)))

            start_idx += n_chunks

        return results

    def _label_rule(self, texts: Sequence[str]) -> List[List[Tuple[str, int, float]]]:
        results = []
        for txt in texts:
            seen = set()
            sample_res = []
            for chunk in self.split_text(txt):
                label = 1 if chunk in seen else 0
                seen.add(chunk)
                sample_res.append((chunk, label, 1.0 if label else 0.0))
            results.append(sample_res)
        return results

def build_labeler(cfg: PipelineConfig, tokenizer: AutoTokenizer | None = None):
    lp = cfg.labeler_params or {}
    return SentenceLabeler(
        labeler_type=cfg.labeler_type,
        sim_threshold=lp.get("sim_threshold", 0.99),
        embed_model=lp.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=lp.get("device", "auto"),
        tokenizer=tokenizer,
        slide_window=lp.get("slide_window", 2000),
        label_chunk=lp.get("label_chunk", 10)
    )