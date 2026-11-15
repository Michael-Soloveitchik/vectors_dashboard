# __init__.py

## Purpose
Package initializer for `backend.reduction_2d`. Re-exports the public API so consumers can import directly from the package without knowing the internal file structure.

## Re-exports

**From `base`:**
- `reduce_2d` -- the main dispatcher function
- `recenter` -- median-based recentering helper
- `ReductionResult` -- dataclass for reduction outcomes

**From `cache`:**
- `ReductionCache` -- in-memory cache for computed projections

All symbols are listed in `__all__`. The docstring documents the public surface: `reduce_2d(method, vectors, params, cache=None) -> ReductionResult`, where `method` is one of `{"PCA", "t-SNE", "UMAP"}`, `vectors` is shape `(N, D)`, and `proj` in the result is shape `(N, 2)` recentered to `(0, 0)`.
