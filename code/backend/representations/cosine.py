"""Plain cosine similarity on the unit sphere (baseline)."""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import pair_scores_return


class Cosine(Representation):
    """L2-normalize + 1-NN cosine against stored training set."""

    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X, y=None, hints=None, aggregator=None):
        _ = hints, aggregator                                # cosine: no centroids
        self._X = self.transform(X)
        self._y = None if y is None else np.asarray(y)
        self._labels = None if self._y is None else np.unique(self._y)
        self._mark_fitted(X)
        return self

    def _score_per_class_raw(self, X):
        """POLARITY: higher = closer. closest-member score per class."""
        self._require_fit()
        sims = self.score_xy(X, self._X)                    # (N, N_train) via score_xy
        out = np.empty((sims.shape[0], len(self._labels)), dtype=sims.dtype)
        for ki, lbl in enumerate(self._labels):
            out[:, ki] = self.closest_score_fn(sims[:, self._y == lbl], axis=1)
        return out

__all__ = ["Cosine"]
