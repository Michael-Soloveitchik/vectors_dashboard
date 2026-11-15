# umap_impl.py

## Purpose
Thin wrapper around `umap-learn`'s UMAP for projecting high-dimensional vectors to 2D. Handles `n_neighbors` clamping (UMAP requires `n_neighbors < n_samples`) and defaults to cosine metric, which is the natural choice for L2-normalized speaker embeddings.

## Motivation
UMAP is one of the three reduction methods offered by the dashboard. It lives in its own file (named `umap_impl.py` to avoid shadowing the `umap` package) so the import is lazy and the implementation is isolated. The cosine metric default and n_neighbors clamping justify the wrapper.

## Key Classes / Functions

### `run_umap`
- **What it does:** Projects an `(N, D)` vector matrix to 2D using UMAP with automatic n_neighbors clamping and cosine metric default.
- **Args:**
  - `vectors` (np.ndarray): shape `(N, D)` input embeddings
  - `params` (dict[str, Any]): optional parameters:
    - `n_neighbors` (int, default `15`): clamped to `[2, N-1]`
    - `min_dist` (float, default `0.1`): minimum distance between points in embedding
    - `metric` (str, default `"cosine"`): distance metric
    - `random_state` (int, default `42`): random seed
- **Returns:** `np.ndarray` of shape `(N, 2)`.
- **Example call:**
  ```python
  proj = run_umap(vectors, {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"})
  ```

## Inputs & Outputs
- **In:** `(N, D)` numpy array of float vectors, parameter dict.
- **Out:** `(N, 2)` numpy array of 2D projected coordinates.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| N > 15, default params | Normal UMAP result | n_neighbors within range |
| N = 5, n_neighbors = 15 | n_neighbors clamped to 4 (`N-1`) | Prevents umap-learn error |
| N = 3 | n_neighbors clamped to 2 (minimum) | Minimum floor |
| metric = "euclidean" | Uses euclidean instead of cosine | Overridable via params |

## Where This Is Used
- `backend/reduction_2d/base.py:81` -- imported lazily inside the `"umap"` branch of `reduce_2d`

## Known Gotchas & Non-Obvious Behavior
- The file is named `umap_impl.py` (not `umap.py`) to avoid shadowing the `umap` package import.
- The `umap` import is **lazy** (inside the function body) to avoid pulling `umap-learn` at module import time.
- The default metric is `"cosine"` rather than UMAP's default `"euclidean"` because the dashboard's primary use case is speaker/audio embeddings that are L2-normalized. Euclidean on the unit sphere is monotone with cosine but UMAP fits cosine faster on sparse high-dimensional clouds.
- `verbose=False` is hardcoded to suppress UMAP's progress output in the dashboard context.

## Dependencies
- `numpy`
- `umap-learn` (`umap.UMAP`, imported lazily)
