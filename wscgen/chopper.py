from typing import List, Tuple
import numpy as np


class Chopper:
    """
    Chopper the generated sentences when detect the repetition.

    Parameters
    ----------
    detector : Callable[[np.ndarray], float]
        Use classifier to predict the repetition score of the current sentence.
    thresh : float
        The threshold to determine the repetition sentence.
    streak_len : int
        When `streak_len` consecutive repetition sentences appear, trigger the chop.
    len_threshold : int
        If the sentence length is ≤ len_threshold, it is regarded as a "short sentence" and will be kept.
    short_streak_len : int
        When `short_streak_len` consecutive repetition short sentences appear, trigger the chop.
    """

    def __init__(
        self,
        tokenizer,
        detector,
        thresh: float = 0.5,
        streak_len: int = 2,
        short_streak_len: int = 5,
        len_threshold: int = 10,
    ):
        self.tokenizer = tokenizer
        self.detector = detector
        self.thresh = thresh
        self.streak_len = streak_len
        self.short_streak_len = short_streak_len
        self.len_threshold = len_threshold

    def _num_tokens(self, s: str) -> int:
        if self.tokenizer is None:
            return len(s.split())
        return len(self.tokenizer.encode(s, add_special_tokens=False))

    def _is_short(self, token_length: int) -> bool:
        return token_length < self.len_threshold

    def chop(
        self,
        full_sentences: List[str],
        full_scores: List[float],
        hidden_state: np.ndarray,
        token_length: int,
    ) -> Tuple[List[str], List[float]]:
        """
        Predicts the repetition score for the new sentence and decides whether to truncate
        the output based on repetition streaks.

        Logic:
        - Keeps a buffer of consecutive repetitive sentences.
        - If the buffer reaches `streak_len` (for long sentences) or `short_streak_len` (for short ones),
          early stopping is triggered, and only the non-repetitive part is returned.

        Parameters
        ----------
        full_sentences : List[str]
            The sentences that have been "confirmed to be kept" so far.
        full_scores : List[float]
            The history scores corresponding to full_sentences.
        hidden_state : np.ndarray
            The hidden state of the current sentence (passed to the detector for scoring).

        Returns
        -------
        chopped_results : List[str]
            The sentences that have been chopped and should continue to be output.
        chopped_scores : List[float]
            The scores corresponding to the chopped results.
        """
        # 1. Predict the repetition score
        cur_score = float(self.detector.predict_proba(hidden_state))

        sentences = full_sentences
        scores = np.asarray(full_scores + [cur_score], dtype=float)

        keep: List[str] = []
        buffer: List[str] = []

        long_streak = 0
        short_streak = 0

        for s, p in zip(sentences, scores):
            short = self._is_short(token_length)

            if p > self.thresh:
                buffer.append(s)

                if short:
                    long_streak = 0
                    short_streak += 1   
                    if short_streak >= self.short_streak_len:
                        break # Trigger chop on short repetition streak
                else:
                    short_streak = 0
                    long_streak += 1
                    if long_streak >= self.streak_len:
                        break # Trigger chop on long repetition streak
                continue

            keep.extend(buffer)

            buffer.clear()
            long_streak = short_streak = 0

            keep.append(s)
        else:
            keep.extend(buffer)

        return keep, scores.tolist()