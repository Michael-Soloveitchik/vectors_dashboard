"""Sparse Representation Classifier — E.1 in the taxonomy.

Reconstruct each query as a sparse combination of all labeled vectors,
then assign to the class whose subset reconstructs best:

    α̂ = argmin ||u − D α||_2^2 + λ ||α||_1
    ŷ = argmin_c ||u − D · δ_c(α̂)||_2     (restrict α to class c)

Taxonomy caveat (V.1): vanilla SRC underperforms on L2-normalized dense
features — its strength is on raw, corrupted, sparse-friendly data.
Included for completeness and documented honestly.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import pair_scores_return


class SRC(Representation):
    def __init__(
        self,
        alpha: float = 1e-5,
        max_iter: int = 2000,
        eps: float = 1e-12,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
        self._D: np.ndarray | None = None                   # (N_L, D) dictionary atoms
        self._y: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "SRC":
        _ = hints, aggregator                               # dictionary atoms, not centroids
        self._D = self.transform(X)
        self._y = np.asarray(y)
        self._labels = np.unique(self._y)
        self._mark_fitted(X)
        return self

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """``(N, K)`` per-class reconstruction scores.

        Native formula is a DISTANCE (residual ``ℓ2``-norm — smaller =
        better). Converted to a convention-compliant score via
        ``distance_to_score_fn`` so no hand-negation leaks into callers:
        higher = closer, same as every other rep in this module.
        """
        from sklearn.linear_model import Lasso
        self._require_fit()
        Xn = self.transform(X)
        Dt = self._D.T                                       # (D, N_L)
        resid_norms = np.empty((Xn.shape[0], len(self._labels)), dtype=np.float64)
        lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter, positive=False)
        for i, u in enumerate(Xn):
            lasso.fit(Dt, u)
            coef = lasso.coef_
            for ki, cls in enumerate(self._labels):
                mask = self._y == cls
                resid = u - Dt @ np.where(mask, coef, 0.0)
                resid_norms[i, ki] = np.linalg.norm(resid)
        # DISTANCE → convention-SCORE via the single helper.
        return self.distance_to_score_fn(resid_norms)

__all__ = ["SRC"]
