"""Multi-Proxy Anchor — C.3 in the taxonomy.

P prototypes per class with "smooth max" scoring. Full Proxy-Anchor
needs SGD on the LSE loss, but its per-class prototypes at convergence
approximate spherical k-means centers per class (hard-EM limit,
α → ∞). We use that hard-EM limit directly — it's training-free,
deterministic, and faithful to the P>1 polysemy story.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import (
    l2_normalize,
    pair_scores_return,
    spherical_kmeans,
)


class MultiProxyAnchor(Representation):
    _uses_aggregator = True

    def __init__(
        self,
        n_proxies: int = 4,
        random_state: int = 42,
        max_iter: int = 50,
        eps: float = 1e-12,
    ):
        self.n_proxies = n_proxies
        self.random_state = random_state
        self.max_iter = max_iter
        self.eps = eps
        self._labels: np.ndarray | None = None
        self._proxies: np.ndarray | None = None          # (K * P_eff, D) normalized
        self._owner: np.ndarray | None = None            # (K * P_eff,) class index

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "MultiProxyAnchor":
        """Fit per-class proxies by spherical k-means. ``hints[label]``
        overrides ``n_proxies`` for that class when present."""
        if aggregator is None:
            raise ValueError(
                "MultiProxyAnchor.fit: aggregator is required "
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
                # Spherical k-means for cluster membership; re-land each
                # center at the aggregator-argmax of its members so the
                # chosen aggregator actually drives the final prototype.
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
        self._owner = np.concatenate(owners, axis=0)
        self._mark_fitted(X)
        return self

    def get_centroids(self) -> dict[object, np.ndarray]:
        self._require_fit()
        return {
            self._labels[ki]: self._proxies[self._owner == ki]
            for ki in range(len(self._labels))
        }

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — closest-proxy score per class."""
        self._require_fit()
        sims = self.score_xy(X, self._proxies)                   # (N, total_proxies)
        K = len(self._labels)
        out = np.full((sims.shape[0], K), -np.inf, dtype=sims.dtype)
        for ki in range(K):
            mask = self._owner == ki
            out[:, ki] = self.closest_score_fn(sims[:, mask], axis=1)
        return out

__all__ = ["MultiProxyAnchor"]
