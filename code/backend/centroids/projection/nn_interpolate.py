"""Inverse-distance-weighted k-NN interpolation helper.

Used by ``project_with_centroids`` to place centroids / unit-ball samples
into a pre-existing 2D layout (t-SNE / UMAP fast path) without rerunning
the full non-linear reduction.
"""
from __future__ import annotations

import numpy as np


__all__: list[str] = []  # private — consumed by project.py only


def _nn_interpolate(
    points_hd: np.ndarray,
    points_2d: np.ndarray,
    queries_hd: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """Project query points into 2D via inverse-distance-weighted k-NN."""
    k = min(k, points_hd.shape[0])
    if k == 0:
        return np.zeros((len(queries_hd), 2), dtype=np.float32)
    dists = np.linalg.norm(
        points_hd[np.newaxis, :, :] - queries_hd[:, np.newaxis, :], axis=2
    )
    knn_idx = np.argpartition(dists, k, axis=1)[:, :k]
    rows = np.arange(len(queries_hd))[:, None]
    knn_dists = dists[rows, knn_idx]
    weights = 1.0 / (knn_dists + 1e-10)
    weights /= weights.sum(axis=1, keepdims=True)
    knn_2d = points_2d[knn_idx]
    return (knn_2d * weights[:, :, None]).sum(axis=1)
