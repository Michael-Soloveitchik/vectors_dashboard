"""CosFace / ArcFace — C.2 in the taxonomy.

Angular-margin softmax. Full CosFace requires SGD on the margin loss,
but per the [Neural Collapse] equivalence (Part III.1, row 2) the
converged prototypes coincide with NCM class means up to rotation —
so we fit prototypes by training L2-normalized multinomial logistic
regression, then apply the CosFace / ArcFace score at inference:

    CosFace:   s · (cos θ_c − m)       for the predicted class
    ArcFace:   s · cos(θ_c + m)

``score`` returns the scaled post-margin similarity for the winning
class (NOT the softmax probability) — higher = more confident.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import l2_normalize, pair_scores_return


class CosFace(Representation):
    def __init__(
        self,
        margin: float = 0.2,
        scale: float = 32.0,
        variant: str = "cosface",                        # "cosface" | "arcface"
        C: float = 10.0,
        eps: float = 1e-12,
    ):
        if variant not in ("cosface", "arcface"):
            raise ValueError(f"variant must be 'cosface' or 'arcface', got {variant}")
        self.margin = margin
        self.scale = scale
        self.variant = variant
        self.C = C
        self.eps = eps
        self._labels: np.ndarray | None = None
        self._W: np.ndarray | None = None                # (K, D) unit-norm prototypes

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "CosFace":
        # CosFace has exactly one prototype per class by construction —
        # `hints` and `aggregator` are accepted for API uniformity but ignored.
        _ = hints, aggregator
        from sklearn.linear_model import LogisticRegression

        Xn = self.transform(X)
        y = np.asarray(y)
        clf = LogisticRegression(solver="lbfgs", C=self.C, max_iter=1000).fit(Xn, y)
        W = clf.coef_
        self._labels = np.asarray(clf.classes_)
        if W.shape[0] == 1:
            # Binary case: reconstruct both class prototypes.
            W = np.vstack([-W, W])
        self._W = l2_normalize(W, eps=self.eps)
        self._mark_fitted(X)
        return self

    def get_centroids(self) -> dict[object, np.ndarray]:
        self._require_fit()
        return {self._labels[i]: self._W[i : i + 1] for i in range(len(self._labels))}

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        # Raw cosine to each class prototype, routed through score_xy so
        # the single pair primitive is the only scoring path.
        # POLARITY: higher = closer — cosine ∈ [−1, 1] already matches.
        self._require_fit()
        return self.score_xy(X, self._W)

__all__ = ["CosFace"]
