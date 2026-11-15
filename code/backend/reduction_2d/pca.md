# pca.py

## Purpose
Thin wrapper around sklearn's PCA for projecting high-dimensional vectors to 2D. This is the fast default path -- the front-end uses PCA on first paint because it is near-instant compared to t-SNE or UMAP.

## Motivation
PCA gets its own file so the import is cheap: the dashboard module graph does not pull all of sklearn just to draw configs. The sklearn import is lazy (inside the function body). Separating each reduction method into its own file also means implementations can be swapped without touching the dispatcher or each other.

## Key Classes / Functions

### `run_pca`
- **What it does:** Projects an `(N, D)` vector matrix to 2 components using sklearn PCA.
- **Args:**
  - `vectors` (np.ndarray): shape `(N, D)` input embeddings, float
  - `params` (dict[str, Any]): optional parameters:
    - `random_state` (int, default `42`): random seed for reproducibility
- **Returns:** `np.ndarray` of shape `(N, 2)`, dtype matches sklearn output (typically float64, cast to float32 by caller in `reduce_2d`).
- **Example call:**
  ```python
  proj = run_pca(vectors, {"random_state": 42})
  # proj.shape == (N, 2)
  ```

## Inputs & Outputs
- **In:** `(N, D)` numpy array of float vectors, parameter dict.
- **Out:** `(N, 2)` numpy array of 2D projected coordinates.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| N >= 2, D >= 2 | `(N, 2)` projection | Normal case |
| N < 2 | `(N, 2)` zeros | Short-circuit; also guarded by `reduce_2d` upstream |
| D == 1 | `(N, 2)` with second column zero-padded | `n_components` clamped to `min(2, N-1, D)` |
| N == 2 | `n_components = min(2, 1, D)` = 1 | Pads to 2 columns with zeros for plottability |

## Where This Is Used
- `backend/reduction_2d/base.py:73` -- imported lazily inside the `"pca"` branch of `reduce_2d`

## Known Gotchas & Non-Obvious Behavior
- The sklearn import happens **inside the function body**, not at module level. This is intentional to keep the initial import of the reduction_2d package lightweight.
- When `n_components` is clamped to 1 (because N == 2 or D == 1), the function zero-pads to 2 columns so the front-end always receives a plottable `(N, 2)` array.
- `n_components` is computed as `min(2, n - 1, vectors.shape[1])` to satisfy sklearn's requirement that `n_components < min(n_samples, n_features)`.

## Dependencies
- `numpy`
- `sklearn.decomposition.PCA` (imported lazily)
