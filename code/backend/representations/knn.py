"""k-Nearest Neighbors with temperature — B.1 in the taxonomy.

Soft majority vote over the top-k cosine neighbors:
:math:`\\hat y = \\arg\\max_c \\sum_{i \\in \\mathcal{N}_k(u)}
\\mathbb{1}[y_i = c] \\exp(S_C(u, x_i) / \\tau)`.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import pair_scores_return


class KNN(Representation):
    # _per_class_probs already returns a proper simplex.
    _raw_is_posterior = True

    def __init__(self, k: int = 7, temperature: float = 0.1, eps: float = 1e-12):
        self.k = k
        self.temperature = temperature
        self.eps = eps
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "KNN":
        _ = hints, aggregator                               # k-NN: no centroids to seed
        self._X = self.transform(X)
        self._y = np.asarray(y)
        self._labels = np.unique(self._y)
        self._mark_fitted(X)
        return self

    def _per_class_votes(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        sims = self.score_xy(X, self._X)                            # (N, N_L) via score_xy
        k = min(self.k, sims.shape[1])
        top_idx = self.topk_closest_fn(sims, k=k, axis=1)              # (N, k) — convention
        top_sims = np.take_along_axis(sims, top_idx, axis=1)        # (N, k)
        top_labels = self._y[top_idx]                               # (N, k)
        weights = np.exp(top_sims / max(self.temperature, 1e-12))
        votes = np.zeros((sims.shape[0], len(self._labels)), dtype=weights.dtype)
        for ki, cls in enumerate(self._labels):
            mask = top_labels == cls
            votes[:, ki] = np.where(mask, weights, 0.0).sum(axis=1)
        return votes

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        # POLARITY: higher = closer — soft vote ∈ [0, 1] with mass on
        # classes whose neighbors are high-cosine (= most similar).
        votes = self._per_class_votes(X)
        total = votes.sum(axis=1, keepdims=True)
        return votes / np.maximum(total, 1e-12)

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        # Domain detection (NOT a score reduction): score_per_class
        # gives probabilities ∈ [0, 1]; score_pairs gives the embedded
        # primitive ∈ [-1, 1]. Negative present → primitive branch.
        s = np.asarray(scores, dtype=np.float64)
        if (s < 0.0).any():
            return np.clip((s + 1.0) * 0.5, 0.0, 1.0)
        return np.clip(s, 0.0, 1.0)


__all__ = ["KNN"]
