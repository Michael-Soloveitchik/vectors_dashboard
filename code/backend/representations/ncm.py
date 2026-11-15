"""Nearest Class Mean (NCM) — A.1 in the taxonomy.

Default: one centroid per class = the argmax of the **score** aggregator
over the class (NOT the L2 mean projected to the sphere, though for
``aggregator="mean"`` with the default cosine score these coincide —
that's the definition of the Fréchet mean on the sphere). With
``n_clusters > 1`` it runs K-means within each class (mean aggregator)
or computes ``k`` spherical-median-initialized components (median
aggregator). Per-class ``hints[label]`` overrides ``n_clusters`` for
that class.

All per-sample scoring routes through ``score_xy`` (inherited from
``Representation``) — the centroid-vs-sample comparison is the same
primitive as any pair evaluation, no shortcuts.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import l2_normalize, pair_scores_return


class NCM(Representation):
    _uses_aggregator = True

    def __init__(
        self,
        n_clusters: int = 1,
        random_state: int = 42,
        eps: float = 1e-12,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.eps = eps
        self._centroids: np.ndarray | None = None        # (total_centroids, D)
        self._owner: np.ndarray | None = None        # (total_centroids,) -> class index

    def fit(self, X, y, hints=None, aggregator=None):
        if aggregator is None:
            raise ValueError(
                "NCM.fit: aggregator is required "
                "(e.g. 'mean', 'median', 'percentile:75', or a callable)."
            )
        # Validate spec via the base-class resolver. Raises on unknown.
        self._aggregator_reducer_fn(aggregator)
        self._aggregator = aggregator
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        hints = hints or {}
        labels = np.unique(y)
        means_list: list[np.ndarray] = []
        owner_list: list[np.ndarray] = []
        for idx, lbl in enumerate(labels):
            X_lbl = X[y == lbl]
            requested = int(hints.get(lbl, self.n_clusters))
            k = max(1, min(requested, X_lbl.shape[0]))
            if k == 1:
                # Single centroid per class — Fréchet mean (aggregator=mean)
                # or spherical geometric median (aggregator=median) under
                # the score. NOT mean-in-L2-then-normalize.
                c = self._centroid_under_score(X_lbl, aggregator=aggregator)
                centers = c[None, :]
            else:
                centers = _k_centroids_under_score(
                    self, X_lbl, k=k, random_state=self.random_state,
                    aggregator=aggregator,
                )
            means_list.append(centers)
            owner_list.append(np.full(centers.shape[0], idx))
        self._labels = labels
        # Centroids stored AS RETURNED by _centroid_under_score:
        #   • aggregator='mean' / 'median' under cosine kernel → un-normalized
        #     L2 mean / Weiszfeld iterate (length ≤ 1, sits inside cluster)
        #   • medoid path → unit-norm class member (length = 1)
        # Scoring re-normalizes via score_xy → transform, so kernels see
        # the same Fréchet direction either way. Display paths get the
        # un-normalized vector, which matches the L2 cluster cloud on PCA.
        self._centroids = np.concatenate(means_list, axis=0)
        self._owner = np.concatenate(owner_list, axis=0)
        self._mark_fitted(X)
        return self

    def get_centroids(self):
        self._require_fit()
        return {self._labels[ki]: self._centroids[self._owner == ki]
                for ki in range(len(self._labels))}

    def _score_per_class_raw(self, X):
        """POLARITY: higher = closer — closest-centroid score per class.
        All reductions via ``closest_score_fn`` (the convention helper).
        """
        self._require_fit()
        sims = self.score_xy(X, self._centroids)                 # (N, total_centroids)
        K = len(self._labels)
        out = np.full((sims.shape[0], K), -np.inf, dtype=sims.dtype)
        for ki in range(K):
            mask = self._owner == ki
            if mask.any():
                out[:, ki] = self.closest_score_fn(sims[:, mask], axis=1)
        return out


def _k_centroids_under_score(
    rep: "NCM",
    X_lbl: np.ndarray,
    k: int,
    random_state: int,
    aggregator: str,
) -> np.ndarray:
    """Multi-prototype per class, consistent with the score and the
    chosen aggregator. Spherical k-means for ``k > 1`` is the standard
    proxy here; each final cluster center is recomputed via
    ``_centroid_under_score`` so the median aggregator actually lands
    at the spherical median of its cluster.
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X_lbl)
    centers = np.empty((k, X_lbl.shape[1]), dtype=np.float64)
    for j in range(k):
        mask = km.labels_ == j
        if mask.any():
            centers[j] = rep._centroid_under_score(
                X_lbl[mask], aggregator=aggregator
            )
        else:
            centers[j] = l2_normalize(km.cluster_centers_[j:j+1])[0]
    return centers


__all__ = ["NCM"]
