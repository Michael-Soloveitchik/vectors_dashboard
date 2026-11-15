"""Arc-cosine kernel as a pure pair-scorer (baseline for D.1 / D.2).

Lifts cosine similarities through :math:`J_n(\\arccos(\\cdot))/\\pi`
so pair scores live in the RKHS of an infinitely wide ReLU net (order
``n = 1`` is the NTK). No learning — the kernel is the representation.
For the classifier variants see ``arc_cosine_svm`` (D.1) and
``arc_cosine_knn`` (D.2).
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import arc_cosine_kernel, pair_scores_return


class ArcCosineKernel(Representation):
    def __init__(self, order: int = 1, eps: float = 1e-12):
        if order not in (0, 1, 2):
            raise ValueError(f"order must be in {{0, 1, 2}}, got {order}")
        self.order = order
        self.eps = eps
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "ArcCosineKernel":
        _ = hints, aggregator                               # kernel: no centroids
        self._X = self.transform(X)
        self._y = None if y is None else np.asarray(y)
        self._labels = None if self._y is None else np.unique(self._y)
        self._mark_fitted(X)
        return self

    def _K(self, X: np.ndarray) -> np.ndarray:
        # Consistent with score_xy: both route through arc-cosine of
        # cosine-similarity. Kept for backward-compat; new code should
        # call score_xy directly.
        self._require_fit()
        return self.score_xy(X, self._X)

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — arc-cosine kernel J_n(θ)/π is
        monotone non-increasing in θ, peaks at θ = 0. Closest-member
        selection routed through ``closest_score_fn``.
        """
        self._require_fit()
        K = self.score_xy(X, self._X)                        # (N, N_L)
        out = np.empty((K.shape[0], len(self._labels)), dtype=K.dtype)
        for ki, lbl in enumerate(self._labels):
            out[:, ki] = self.closest_score_fn(K[:, self._y == lbl], axis=1)
        return out

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        # Arc-cosine kernel values already live in [0, 1] — just clip.
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


    def _pair_kernel(self, M_pair, Xn):
        return arc_cosine_kernel(M_pair, n=self.order)

__all__ = ["ArcCosineKernel"]
