"""Arc-Cosine Kernel SVM — D.1 in the taxonomy.

Max-margin hyperplane in the RKHS of an infinitely wide ReLU network.
Uses sklearn's SVC with a precomputed arc-cosine Gram matrix (order
``n ∈ {0, 1, 2}``). One-vs-rest multi-class handled by sklearn.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import arc_cosine_kernel, pair_scores_return


class ArcCosineSVM(Representation):
    def __init__(
        self,
        order: int = 1,
        C: float = 1.0,
        decision_function_shape: str = "ovr",
        eps: float = 1e-12,
    ):
        if order not in (0, 1, 2):
            raise ValueError(f"order must be in {{0, 1, 2}}, got {order}")
        self.order = order
        self.C = C
        self.decision_function_shape = decision_function_shape
        self.eps = eps
        self._Xtr: np.ndarray | None = None
        self._model = None
        self._labels: np.ndarray | None = None

    def _K(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return arc_cosine_kernel(X @ Y.T, n=self.order)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "ArcCosineSVM":
        _ = hints, aggregator                               # kernel trick: no centroids
        from sklearn.svm import SVC

        self._Xtr = self.transform(X)
        y = np.asarray(y)
        K_train = self._K(self._Xtr, self._Xtr)
        self._model = SVC(
            kernel="precomputed",
            C=self.C,
            decision_function_shape=self.decision_function_shape,
        ).fit(K_train, y)
        self._labels = self._model.classes_
        self._mark_fitted(X)
        return self

    def _decision(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        # score_xy already wraps transform + arc-cosine kernel so the SVM
        # Gram row consistency with training is automatic.
        K_test = self.score_xy(X, self._Xtr)
        df = self._model.decision_function(K_test)
        if df.ndim == 1:
            df = np.stack([-df, df], axis=1)                    # binary → (N, 2)
        return df

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — SVM decision margin toward class
        ``c`` grows with affinity to ``c``'s support vectors.
        """
        return self._decision(X)                             # (N, K)

    def _pair_kernel(self, M_pair, Xn):
        return arc_cosine_kernel(M_pair, n=self.order)

__all__ = ["ArcCosineSVM"]
