from __future__ import annotations

from typing import List, Protocol
import numpy as np

class StreakTrimmer:
    """Default heuristic: stop once we hit `streak_len` consecutive > threshold."""

    def __init__(self, thresh: float = 0.5, streak_len: int = 2):
        self.thresh = thresh
        self.streak_len = streak_len

    def trim(self, sentences: List[str], scores: np.ndarray) -> str:
        keep, streak = [], 0
        for s, p in zip(sentences, scores):
            streak = streak + 1 if p > self.thresh else 0
            if streak >= self.streak_len:
                break
            keep.append(s)
        return "".join(keep)

class ThresholdTrimmer:
    """Drop everything *after* the first sentence whose score > threshold."""

    def __init__(self, thresh: float = 0.5):
        self.thresh = thresh

    def trim(self, sentences: List[str], scores: np.ndarray) -> str:
        keep = []
        for s, p in zip(sentences, scores):
            if p > self.thresh:
                break
            keep.append(s)
        return "".join(keep)

class StreakFilterTrimmer:
    """
    Variant of StreakTrimmer that ignores *single* short sentences in the
    high-score streak, but still trims if a run of short sentences themselves
    exceeds `short_streak_len`.

    """
    def __init__(
        self,
        tokenizer=None,
        thresh: float = 0.5,
        streak_len: int = 2,
        len_threshold: int = 3,
        short_streak_len: int = 2,
    ):
        self.tokenizer = tokenizer
        self.thresh = thresh
        self.streak_len = streak_len
        self.len_threshold = len_threshold
        self.short_streak_len = short_streak_len

    def _num_tokens(self, s: str) -> int:
        if self.tokenizer is None:
            return len(s.split())
        return len(self.tokenizer.encode(s, add_special_tokens=False))

    def _is_short(self, s: str) -> bool:
        return self._num_tokens(s) < self.len_threshold

    def trim(self, sentences: List[str], scores: np.ndarray) -> str:
        keep: List[str] = []
        long_streak = 0  
        short_streak = 0    

        for s, p in zip(sentences, scores):
            short = self._is_short(s)

            if short:
                long_streak = 0                      
                if p > self.thresh:
                    short_streak += 1
                    if short_streak >= self.short_streak_len:
                        break                           
                else:
                    short_streak = 0
                keep.append(s)
                continue

            short_streak = 0                           
            if p > self.thresh:
                long_streak += 1
                if long_streak >= self.streak_len:
                    break
            else:
                long_streak = 0
            keep.append(s)

        return "".join(keep)

class StreakFilterRollbackTrimmer(StreakFilterTrimmer):
    """
    Same as StreakFilterTrimmer, but if a high-score streak finally
    triggers trimming, we roll back (drop) all sentences that belong to
    that streak instead of keeping the early ones.
    """

    def trim(self, sentences: List[str], scores: np.ndarray) -> str:
        keep: List[str] = []
        buffer: List[str] = []      
        long_streak = short_streak = 0  

        for s, p in zip(sentences, scores):
            short = self._is_short(s)

            if p > self.thresh:
                # potentially repetitive: keep in buffer for now
                buffer.append(s)

                if short:
                    long_streak = 0
                    short_streak += 1
                    if short_streak >= self.short_streak_len:
                        break          # drop buffer & stop
                else:
                    short_streak = 0
                    long_streak += 1
                    if long_streak >= self.streak_len:
                        break          # drop buffer & stop
                continue               # still inside a high-score streak

            # non-repetitive sentence
            # flush whatever is in buffer, then keep current sentence
            keep.extend(buffer)
            buffer.clear()

            long_streak = short_streak = 0  
            keep.append(s)

        else:
            # loop finished without early break → flush remaining buffer
            keep.extend(buffer)

        return "".join(keep)
    

def build_trimmer(t_type: str, params: dict):
    t_type = t_type.lower()
    if t_type == "streak":
        return StreakTrimmer(**params)
    if t_type == "threshold":
        return ThresholdTrimmer(**params)
    if t_type == "streak_filter":
        return StreakFilterTrimmer(**params)
    if t_type == "streak_filter_rollback":
        return StreakFilterRollbackTrimmer(**params)
    
    raise ValueError(f"Unknown trimmer_type: {t_type}")
