"""Per-label GMM centroids and projection helpers.

Public surface:
    fit_centroids(df, label_col, n_per_label, ...) -> CentroidSet
    project_centroids(centroids, vectors, method, params, cache) -> proj
"""

from .gmm import CentroidSet, fit_centroids
from .methodology_fit import fit_centroids_via_methodology
from .projection import project_with_centroids

__all__ = [
    "CentroidSet",
    "fit_centroids",
    "fit_centroids_via_methodology",
    "project_with_centroids",
]
