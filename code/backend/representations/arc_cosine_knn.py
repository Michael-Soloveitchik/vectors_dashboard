"""Kernelized k-NN (Arc-Cosine) — D.2 in the taxonomy.

Replaces cosine in k-NN with the RKHS similarity
:math:`K_n(x, y) = J_n(\\arccos(x^\\top y)) / \\pi`. Votes are weighted
by the kernel value rather than exponentiated cosine — the kernel
itself already does the nonlinear warping.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import arc_cosine_kernel, pair_scores_return


class ArcCosineKNN(Representation):
    _raw_is_posterior = True                                # normalized votes = simplex

    def __init__(self, k: int = 7, order: int = 1, eps: float = 1e-12):
        if order not in (0, 1, 2):
            raise ValueError(f"order must be in {{0, 1, 2}}, got {order}")
        self.k = k
        self.order = order
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
    ) -> "ArcCosineKNN":
        _ = hints, aggregator                               # k-NN: no centroids to seed
        self._X = self.transform(X)
        self._y = np.asarray(y)
        self._labels = np.unique(self._y)
        self._mark_fitted(X)
        return self

    def _per_class_votes(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        K = self.score_xy(X, self._X)                                # (N, N_L) via score_xy
        k = min(self.k, K.shape[1])
        top_idx = self.topk_closest_fn(K, k=k, axis=1)                  # convention-routed
        top_K = np.take_along_axis(K, top_idx, axis=1)
        top_labels = self._y[top_idx]
        votes = np.zeros((K.shape[0], len(self._labels)), dtype=K.dtype)
        for ki, cls in enumerate(self._labels):
            mask = top_labels == cls
            votes[:, ki] = np.where(mask, top_K, 0.0).sum(axis=1)
        return votes

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — arc-cosine-weighted vote ∈ [0, 1]."""
        votes = self._per_class_votes(X)
        total = votes.sum(axis=1, keepdims=True)
        return votes / np.maximum(total, 1e-12)

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        # Both score_pairs (kernel) and score_per_class (prob) live in [0, 1].
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


    def _pair_kernel(self, M_pair, Xn):
        return arc_cosine_kernel(M_pair, n=self.order)

__all__ = ["ArcCosineKNN"]
