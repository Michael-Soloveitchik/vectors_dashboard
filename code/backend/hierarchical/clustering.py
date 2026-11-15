"""Run scipy linkage on a confusion matrix and return a leaf-ordered result.

The clustering operates on a *distance* matrix derived from the
similarity confusion matrix:
    distance = 1 - similarity   (clipped to [0, 2] for safety)

The leaf order is the canonical "sorted by hierarchy" ordering used to
re-arrange the confusion matrix when the user enables the hierarchical
mode (spec point 40 — sort options become invisible, dendrogram drives
the order).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from ..confusion_matrix.matrix import ConfusionMatrix


@dataclass
class HierarchyResult:
    """Output of agglomerative clustering on a confusion matrix.

    `Z`               : scipy linkage matrix, shape (n-1, 4).
    `leaf_order`      : permutation of [0..n) by leaf position.
    `ordered_session_ids`: cm.session_ids reordered by leaf_order.
    `method`, `metric`: parameters echoed back for the front-end.
    `n`               : number of leaves.
    """

    Z: np.ndarray
    leaf_order: np.ndarray
    ordered_session_ids: list[str]
    method: str
    metric: str
    n: int


def _similarity_to_condensed_distance(matrix: np.ndarray) -> np.ndarray:
    """Convert a square similarity matrix into a condensed distance vector."""
    sim = np.clip(matrix, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    # Symmetrize to neutralize float drift before squareform.
    dist = 0.5 * (dist + dist.T)
    return squareform(dist, checks=False)


def cluster(
    cm: ConfusionMatrix,
    method: str = "average",
    metric: str = "cosine",
    optimal_ordering: bool = False,
) -> HierarchyResult:
    """Linkage on cm.matrix; returns a leaf-ordered result.

    When `optimal_ordering=True`, scipy reorders the leaves to minimize
    the sum of distances between adjacent leaves — visually smoother
    dendrogram at the cost of one extra TSP-style sort.
    """
    if cm.n < 2:
        return HierarchyResult(
            Z=np.zeros((0, 4), dtype=np.float64),
            leaf_order=np.arange(cm.n),
            ordered_session_ids=list(cm.session_ids),
            method=method,
            metric=metric,
            n=cm.n,
        )

    condensed = _similarity_to_condensed_distance(cm.matrix)
    Z = linkage(condensed, method=method, optimal_ordering=optimal_ordering)
    order = np.asarray(leaves_list(Z), dtype=np.int64)
    ordered_sids = [cm.session_ids[i] for i in order]
    return HierarchyResult(
        Z=Z,
        leaf_order=order,
        ordered_session_ids=ordered_sids,
        method=method,
        metric=metric,
        n=cm.n,
    )
