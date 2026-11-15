"""2D dimensionality reduction with method-aware caching.

Public surface:
    reduce_2d(method, vectors, params, cache=None) -> (proj, info)

`method` is one of {"PCA", "t-SNE", "UMAP"}. `vectors` is shape (N, D).
`proj` is shape (N, 2), already re-centered to (0, 0). `cache` is the
ReductionCache instance the GUI threads through every callback so
toggles are instant.
"""

from .base import reduce_2d, recenter, ReductionResult
from .cache import ReductionCache

__all__ = ["reduce_2d", "recenter", "ReductionResult", "ReductionCache"]
