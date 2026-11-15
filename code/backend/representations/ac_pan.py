"""AC-PAN (Arc-Cosine Proxy-Anchored Neighborhoods) — III.4 in the taxonomy.

Merger of Multi-Proxy Anchor (C.3), Arc-Cosine kernel (D.1), and k-NN
(B.1): learnable proxies, RKHS-warped similarity, k-NN vote over the
proxy bank at inference — cost Θ(K · P · d), independent of training
set size.

Proxy placement uses spherical k-means per class (same hard-EM limit
used by ``MultiProxyAnchor``); the arc-cosine kernel
(order ``n ∈ {0, 1, 2}``) is applied at inference. Caveat from the
taxonomy: this does *not* solve the antipodal-symmetry obstruction on
its own — use the Bingham–Geodesic kernel for that (§III.3).
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import (
    arc_cosine_kernel,
    pair_scores_return,
    spherical_kmeans,
)


class ACPAN(Representation):
    _raw_is_posterior = True                                # normalized votes = simplex
    _uses_aggregator = True

    def __init__(
        self,
        n_proxies: int = 4,
        order: int = 1,
        k: int = 3,
        temperature: float = 0.1,
        random_state: int = 42,
        max_iter: int = 50,
        eps: float = 1e-12,
    ):
        if order not in (0, 1, 2):
            raise ValueError(f"order must be in {{0, 1, 2}}, got {order}")
        self.n_proxies = n_proxies
        self.order = order
        self.k = k
        self.temperature = temperature
        self.random_state = random_state
        self.max_iter = max_iter
        self.eps = eps
        self._labels: np.ndarray | None = None
        self._proxies: np.ndarray | None = None
        self._proxy_labels: np.ndarray | None = None            # class index per proxy

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "ACPAN":
        """Place per-class proxies via spherical k-means. ``hints[label]``
        overrides ``n_proxies`` for that class when present."""
        if aggregator is None:
            raise ValueError(
                "ACPAN.fit: aggregator is required "
                "(e.g. 'mean', 'median', 'percentile:75', or a callable)."
            )
        self._aggregator_reducer_fn(aggregator)
        self._aggregator = aggregator
        Xn = self.transform(X)
        y = np.asarray(y)
        hints = hints or {}
        labels = np.unique(y)
        proxies: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        for ki, lbl in enumerate(labels):
            X_lbl = Xn[y == lbl]
            requested = int(hints.get(lbl, self.n_proxies))
            p = max(1, min(requested, X_lbl.shape[0]))
            if p == 1:
                centers = self._centroid_under_score(
                    X_lbl, aggregator=aggregator
                )[None]
            else:
                km = spherical_kmeans(
                    X_lbl, k=p, max_iter=self.max_iter,
                    random_state=self.random_state + ki,
                )
                assign = self.arg_closest_score_fn(X_lbl @ km.T, axis=1)
                centers = np.empty_like(km)
                for j in range(p):
                    mask = assign == j
                    if mask.any():
                        centers[j] = self._centroid_under_score(
                            X_lbl[mask], aggregator=aggregator
                        )
                    else:
                        centers[j] = km[j]
            proxies.append(centers)
            owners.append(np.full(centers.shape[0], ki))
        self._labels = labels
        self._proxies = np.concatenate(proxies, axis=0)
        self._proxy_labels = np.concatenate(owners, axis=0)
        self._mark_fitted(X)
        return self

    def get_centroids(self) -> dict[object, np.ndarray]:
        self._require_fit()
        return {
            self._labels[ki]: self._proxies[self._proxy_labels == ki]
            for ki in range(len(self._labels))
        }

    def _per_class_votes(self, X: np.ndarray) -> np.ndarray:
        self._require_fit()
        K = self.score_xy(X, self._proxies)                       # (N, total_proxies) via score_xy
        kk = min(self.k, K.shape[1])
        top_idx = self.topk_closest_fn(K, k=kk, axis=1)              # convention-routed
        top_K = np.take_along_axis(K, top_idx, axis=1)
        top_owner = self._proxy_labels[top_idx]
        weights = np.exp(top_K / max(self.temperature, 1e-12))
        votes = np.zeros((K.shape[0], len(self._labels)), dtype=weights.dtype)
        for ki in range(len(self._labels)):
            mask = top_owner == ki
            votes[:, ki] = np.where(mask, weights, 0.0).sum(axis=1)
        return votes

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — soft vote over proxies whose
        arc-cosine kernel to x is large (small angle θ).
        """
        votes = self._per_class_votes(X)
        total = votes.sum(axis=1, keepdims=True)
        return votes / np.maximum(total, 1e-12)

    def normalize_score(self, scores: np.ndarray) -> np.ndarray:
        # Both score_pairs (kernel) and score_per_class (prob) live in [0, 1].
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


    def _pair_kernel(self, M_pair, Xn):
        return arc_cosine_kernel(M_pair, n=self.order)

__all__ = ["ACPAN"]
