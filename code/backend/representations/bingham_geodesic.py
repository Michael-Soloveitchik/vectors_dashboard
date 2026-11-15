"""Bingham–Geodesic kernel — III.3 (proposed in the taxonomy).

Antipodally-invariant kernel on the sphere:

    K_BG(x, y; γ, p) = exp(−γ · (arccos(|x · y|) / π)^p)

``fit`` stores the labeled set; inference is a soft-max-over-class
kernel score. Prefer this over arc-cosine when polysemy is suspected
to be antipodal (PCA features, contrastive features without sign-fixing
augmentation). For sign-meaningful embeddings (speaker / face),
arc-cosine is the safer default.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import bingham_geodesic_kernel, pair_scores_return


class BinghamGeodesic(Representation):
    def __init__(
        self,
        gamma: float = 4.0,
        p: float = 2.0,
        eps: float = 1e-12,
    ):
        self.gamma = gamma
        self.p = p
        self.eps = eps
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def _K(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return bingham_geodesic_kernel(X @ Y.T, gamma=self.gamma, p=self.p)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "BinghamGeodesic":
        _ = hints, aggregator                               # kernel trick: no centroids
        self._X = self.transform(X)
        self._y = np.asarray(y)
        self._labels = np.unique(self._y)
        self._mark_fitted(X)
        return self

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — exp(−γ·(arccos|·|/π)^p) peaks at 1
        for collinear pairs. Closest-member via ``closest_score_fn``.
        """
        self._require_fit()
        K = self.score_xy(X, self._X)                        # (N, N_L) via score_xy
        out = np.empty((K.shape[0], len(self._labels)), dtype=K.dtype)
        for ki, cls in enumerate(self._labels):
            mask = self._y == cls
            out[:, ki] = self.closest_score_fn(K[:, mask], axis=1)
        return out

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        # Bingham-Geodesic kernel is exp(...) — always in (0, 1].
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


    def _pair_kernel(self, M_pair, Xn):
        return bingham_geodesic_kernel(M_pair, gamma=self.gamma, p=self.p)

__all__ = ["BinghamGeodesic"]
