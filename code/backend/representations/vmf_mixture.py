"""vMF Mixture — A.2 in the taxonomy.

One GMM per class fit on raw vectors, means renormalized to the sphere
to play the role of vMF component directions. Polysemy-aware when
``n_components > 1``. For the dashboard's DataFrame pipeline (train/eval
holdout, time budget), use ``backend.centroids.fit_centroids`` — this
class is the generic X/y counterpart.
"""

from __future__ import annotations

import numpy as np

from .base import Representation
from .math_utils import l2_normalize, pair_scores_return


class VMFMixture(Representation):
    _uses_aggregator = True

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        random_state: int = 42,
        eps: float = 1e-12,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.eps = eps
        self._labels: np.ndarray | None = None
        self._centroids: np.ndarray | None = None        # (total_components, D) normalized
        self._owner: np.ndarray | None = None        # (total_components,) class index

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "VMFMixture":
        """Fit one GMM per class. ``hints[label]`` overrides ``n_components``
        for that class when present."""
        if aggregator is None:
            raise ValueError(
                "VMFMixture.fit: aggregator is required "
                "(e.g. 'mean', 'median', 'percentile:75', or a callable)."
            )
        self._aggregator_reducer_fn(aggregator)
        self._aggregator = aggregator
        from sklearn.mixture import GaussianMixture

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        hints = hints or {}
        labels = np.unique(y)
        means_list: list[np.ndarray] = []
        owner_list: list[np.ndarray] = []
        for idx, lbl in enumerate(labels):
            X_lbl = X[y == lbl]
            if X_lbl.shape[0] < 2:
                means_list.append(
                    self._centroid_under_score(X_lbl,
                                               aggregator=aggregator)[None]
                )
                owner_list.append(np.array([idx]))
                continue
            requested = int(hints.get(lbl, self.n_components))
            k = max(1, min(requested, X_lbl.shape[0]))
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                reg_covar=1e-4,
            ).fit(X_lbl)
            # Re-project each component's mean to the argmax of the
            # chosen aggregator over the samples it owns. GMM is used
            # to discover component membership; centroids land on the
            # Fréchet mean / spherical median, not on the GMM mean.
            resp = gmm.predict(X_lbl)
            comp = np.empty((k, X_lbl.shape[1]), dtype=np.float64)
            for j in range(k):
                mask = resp == j
                if mask.any():
                    comp[j] = self._centroid_under_score(
                        X_lbl[mask], aggregator=aggregator
                    )
                else:
                    comp[j] = l2_normalize(gmm.means_[j:j+1])[0]
            means_list.append(comp)
            owner_list.append(np.full(k, idx))
        self._labels = labels
        # Stored AS RETURNED — see NCM.fit comment. score_xy re-normalizes.
        self._centroids = np.concatenate(means_list, axis=0)
        self._owner = np.concatenate(owner_list, axis=0)
        self._mark_fitted(X)
        return self

    def get_centroids(self) -> dict[object, np.ndarray]:
        self._require_fit()
        return {
            self._labels[ki]: self._centroids[self._owner == ki]
            for ki in range(len(self._labels))
        }

    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """POLARITY: higher = closer — closest-component score per class."""
        self._require_fit()
        sims = self.score_xy(X, self._centroids)                     # (N, total_components)
        K = len(self._labels)
        out = np.full((sims.shape[0], K), -np.inf, dtype=sims.dtype)
        for ki in range(K):
            mask = self._owner == ki
            if mask.any():
                out[:, ki] = self.closest_score_fn(sims[:, mask], axis=1)
        return out

__all__ = ["VMFMixture"]
