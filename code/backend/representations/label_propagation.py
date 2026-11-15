"""Label Propagation (graph Laplacian) — B.2 in the taxonomy.

Builds an RBF-on-cosine-distance affinity over the training set and
diffuses labels until convergence. Wraps scikit-learn's transductive
``LabelPropagation`` — inference on new points is done by the same
kernel-weighted vote scikit-learn uses internally.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import pair_scores_return


class LabelPropagation(Representation):
    # sklearn.predict_proba already returns a proper simplex.
    _raw_is_posterior = True

    def __init__(
        self,
        gamma: float = 20.0,
        max_iter: int = 1000,
        tol: float = 1e-3,
        eps: float = 1e-12,
    ):
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self._model = None
        self._labels: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "LabelPropagation":
        _ = hints, aggregator                               # graph-Laplacian: no centroids
        from sklearn.semi_supervised import LabelPropagation

        Xn = self.transform(X)
        y = np.asarray(y)
        self._model = LabelPropagation(
            kernel="rbf", gamma=self.gamma, max_iter=self.max_iter, tol=self.tol
        ).fit(Xn, y)
        self._labels = self._model.classes_
        self._Xtr = Xn                                      # kept for score_xy pair primitive
        self._mark_fitted(X)
        return self

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — propagated posterior ∈ [0, 1]."""
        self._require_fit()
        return self._model.predict_proba(self.transform(X))

    def score_xy(self, x, y):
        """Pair score = RBF affinity used by the fitted graph —
        exp(−γ · ||x_n − y_n||²) for transformed unit-sphere vectors.
        Matches the kernel sklearn uses under the hood for propagation.
        POLARITY: higher = closer — affinity ∈ (0, 1], peaks at d = 0.
        """
        self._require_fit()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x1, y1 = (x.ndim == 1), (y.ndim == 1)
        if x1: x = x[None, :]
        if y1: y = y[None, :]
        xn, yn = self.transform(x), self.transform(y)
        # ||xn - yn||^2 = 2 - 2 cos(xn, yn) for unit vectors
        M = 2.0 - 2.0 * (xn @ yn.T)
        M = np.exp(-self.gamma * M)
        if x1 and y1: return float(M[0, 0])
        if x1: return np.asarray(M[0], dtype=np.float64)
        if y1: return np.asarray(M[:, 0], dtype=np.float64)
        return np.asarray(M, dtype=np.float64)

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        # Domain detection (NOT a score reduction): propagated posterior
        # ∈ [0, 1] vs embedded-pair primitive ∈ [-1, 1]. Negative
        # present → primitive branch.
        s = np.asarray(scores, dtype=np.float64)
        if (s < 0.0).any():
            return np.clip((s + 1.0) * 0.5, 0.0, 1.0)
        return np.clip(s, 0.0, 1.0)


__all__ = ["LabelPropagation"]
