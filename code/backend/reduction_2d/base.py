"""Top-level reduce_2d() entry point and shared helpers.

`reduce_2d` is intentionally a thin dispatcher: PCA / t-SNE / UMAP each
live in their own < 80-line file so we can swap an implementation
without touching the rest of the system.

The recentering pass moves the projected cloud to (0, 0) so the
front-end never has to translate the camera — required by spec
"re-centre by each of the methods". We re-center via the median, not
the mean, so an outlier point doesn't drag the camera off the cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache import ReductionCache

ReductionParams = dict[str, Any]


@dataclass
class ReductionResult:
    """Outcome of one reduction call.

    `proj`         : (N, 2) float32 projected coordinates, recentered.
    `method`       : the human label that was actually used.
    `params_used`  : parameters echoed back so the front-end can show what ran.
    `cache_hit`    : True iff the result came from the cache.
    """

    proj: np.ndarray
    method: str
    params_used: ReductionParams
    cache_hit: bool


def recenter(proj: np.ndarray) -> np.ndarray:
    """Subtract the per-axis median so the cloud is centered at (0, 0)."""
    if proj.size == 0:
        return proj
    return proj - np.median(proj, axis=0, keepdims=True)


def reduce_2d(
    method: str,
    vectors: np.ndarray,
    params: ReductionParams | None = None,
    cache: ReductionCache | None = None,
    cache_key_extra: tuple = (),
) -> ReductionResult:
    """Run one of {PCA, t-SNE, UMAP} with caching + recenter."""
    params = params or {}
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got {vectors.shape}")
    if vectors.shape[0] < 2:
        # degenerate inputs short-circuit to a single (0, 0) point
        proj = np.zeros((vectors.shape[0], 2), dtype=np.float32)
        return ReductionResult(proj, method, dict(params), False)

    method_norm = method.strip().lower().replace("-", "")

    if cache is not None:
        cached = cache.get(method_norm, vectors, params, cache_key_extra)
        if cached is not None:
            return ReductionResult(cached, method, dict(params), True)

    if method_norm == "pca":
        from .pca import run_pca

        proj = run_pca(vectors, params)
    elif method_norm == "tsne":
        from .tsne import run_tsne

        proj = run_tsne(vectors, params)
    elif method_norm == "umap":
        from .umap_impl import run_umap

        proj = run_umap(vectors, params)
    else:
        raise ValueError(
            f"unknown method {method!r}; expected one of PCA, t-SNE, UMAP"
        )

    proj = recenter(proj.astype(np.float32, copy=False))

    if cache is not None:
        cache.put(method_norm, vectors, params, cache_key_extra, proj)

    return ReductionResult(proj, method, dict(params), False)
