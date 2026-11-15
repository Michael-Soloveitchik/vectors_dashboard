# base.py

## Purpose
Top-level entry point for 2D dimensionality reduction. Provides the `reduce_2d` dispatcher that routes to PCA, t-SNE, or UMAP implementations, applies median-based recentering, and integrates with the in-memory cache. Also defines the `ReductionResult` dataclass that all callers receive.

## Motivation
PCA, t-SNE, and UMAP each live in their own small file so implementations can be swapped without touching the rest of the system. This module is the thin dispatcher that unifies them behind a single call signature. The recentering pass (median, not mean) ensures the projected point cloud is always at `(0, 0)` so the front-end never needs to translate the camera -- a requirement from the dashboard spec.

## Key Classes / Functions

### `ReductionResult`
- **What it does:** Dataclass holding the outcome of one reduction call.
- **Fields:**
  - `proj` (np.ndarray): shape `(N, 2)` float32 projected coordinates, already recentered
  - `method` (str): the human-readable method label that was used
  - `params_used` (dict[str, Any]): parameters echoed back so the front-end can display what ran
  - `cache_hit` (bool): `True` if the result came from the cache
- **Example:**
  ```python
  result = reduce_2d("PCA", vectors)
  print(result.proj.shape)   # (N, 2)
  print(result.cache_hit)    # False
  ```

### `recenter`
- **What it does:** Subtracts the per-axis median from a 2D projection array so the cloud is centered at (0, 0).
- **Args:**
  - `proj` (np.ndarray): shape `(N, 2)` projection
- **Returns:** `np.ndarray` -- same shape, recentered. Returns input unchanged if empty.
- **Example call:**
  ```python
  centered = recenter(proj)
  # np.median(centered, axis=0) ≈ [0, 0]
  ```

### `reduce_2d`
- **What it does:** Runs one of PCA, t-SNE, or UMAP on a vector matrix, with optional caching and automatic recentering.
- **Args:**
  - `method` (str): one of `"PCA"`, `"t-SNE"`, `"UMAP"` (case-insensitive, hyphens ignored)
  - `vectors` (np.ndarray): shape `(N, D)` input embeddings
  - `params` (dict[str, Any] | None, default `None`): method-specific parameters (e.g., `perplexity`, `n_neighbors`)
  - `cache` (ReductionCache | None, default `None`): optional in-memory cache instance
  - `cache_key_extra` (tuple, default `()`): additional values folded into the cache key
- **Returns:** `ReductionResult` with `proj` shape `(N, 2)`.
- **Raises:** `ValueError` if `vectors` is not 2D, or if `method` is not one of the three supported methods.
- **Example call:**
  ```python
  from backend.reduction_2d import reduce_2d, ReductionCache
  cache = ReductionCache()
  result = reduce_2d("UMAP", vectors, {"n_neighbors": 15, "metric": "cosine"}, cache=cache)
  ```

### `ReductionParams`
- **What it is:** Type alias -- `dict[str, Any]`. Used for type hints on parameter dictionaries.

## Inputs & Outputs
- **In:** A 2D numpy array of shape `(N, D)` where N is the number of data points and D is the embedding dimension. Method name as a string. Optional parameter dict and cache.
- **Out:** `ReductionResult` containing a `(N, 2)` float32 array of recentered 2D coordinates, plus metadata.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| `vectors.shape[0] < 2` | Returns `(N, 2)` zeros, `cache_hit=False` | Degenerate input short-circuit |
| `vectors.ndim != 2` | `ValueError` raised | Must be exactly 2D |
| Unknown method string | `ValueError` with helpful message | Lists valid methods |
| Cache hit | Returns cached `proj`, `cache_hit=True` | No recomputation |
| Cache miss | Computes, recenters, stores in cache, returns | `cache_hit=False` |
| `cache=None` | Computes without caching | Always recomputes |
| Method string `"t-SNE"` / `"tsne"` / `"T-SNE"` | All normalize to `"tsne"` | Case-insensitive, hyphens stripped |

## Where This Is Used
- `backend/reduction_2d/__init__.py` -- re-exports `reduce_2d`, `recenter`, `ReductionResult`
- `backend/spots/reduction_2D.py:9` -- imports the entire public surface for the spots module
- `backend/centroids/projection.py:33` -- imports `reduce_2d`, `ReductionCache`, `ReductionResult` for centroid projection
- `TESTS/tests/test_density_contour.py:12` -- uses `reduce_2d` and `ReductionCache` in tests

## Known Gotchas & Non-Obvious Behavior
- Method imports are **lazy** (inside the if-branches) so that importing `base.py` does not pull in sklearn or umap-learn. This matters because PCA is the fast default path on first paint.
- Recentering uses the **median**, not the mean, so outlier points do not drag the camera off the cluster center.
- The `cache_key_extra` tuple is passed through to `ReductionCache.make_key` -- it allows callers (e.g., `centroids/projection.py`) to distinguish reductions of different logical datasets that happen to use the same method and params.
- The method string is normalized (lowercased, hyphens removed) before cache lookup and dispatch, but the original casing is preserved in `ReductionResult.method`.

## Dependencies
- `numpy`
- `dataclasses` (standard library)
- Internal: `backend.reduction_2d.cache.ReductionCache`
- Conditional: `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`, `umap.UMAP` (imported lazily at call time)
