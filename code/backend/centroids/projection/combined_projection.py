"""Result dataclass returned by ``project_with_centroids``."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...reduction_2d import ReductionResult


__all__ = ["CombinedProjection"]


@dataclass
class CombinedProjection:
    """Output of project_with_centroids.

    `points_2d`     : (N_points, 2) projected coordinates of the data.
    `centroids_2d`  : dict label → (k, 2) projected centroids.
    `centroid_labels`: list of label strings, len = sum_k.
    `result`        : the underlying ReductionResult on the joined matrix.
    `base_radii_2d` : dict label → (k,) per-centroid base radius in 2D
                     coordinates (median projected length of a unit raw
                     vector at that centroid). Multiply by the user's
                     threshold to get the visible circle radius.
    """

    points_2d: np.ndarray
    centroids_2d: dict[str, np.ndarray]
    centroid_labels: list[str]
    result: ReductionResult
    base_radii_2d: dict[str, np.ndarray] = field(default_factory=dict)
    ellipses_2d: dict[str, list[dict]] = field(default_factory=dict)
    pca_info: dict | None = None
    centroids_hd: dict[str, np.ndarray] = field(default_factory=dict)
