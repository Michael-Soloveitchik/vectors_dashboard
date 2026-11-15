# tsne.py

## Purpose
Thin wrapper around sklearn's t-SNE for projecting high-dimensional vectors to 2D. Handles perplexity clamping (sklearn raises if `perplexity >= n_samples`) and the sklearn 1.5 API rename from `n_iter` to `max_iter`.

## Motivation
t-SNE is one of the three reduction methods offered by the dashboard controls panel. It lives in its own file so the sklearn import is lazy and the implementation can be modified independently of PCA or UMAP. The perplexity clamping and API compatibility shim justify having a wrapper rather than calling sklearn directly.

## Key Classes / Functions

### `run_tsne`
- **What it does:** Projects an `(N, D)` vector matrix to 2D using sklearn t-SNE with automatic perplexity clamping and API version compatibility.
- **Args:**
  - `vectors` (np.ndarray): shape `(N, D)` input embeddings
  - `params` (dict[str, Any]): optional parameters:
    - `perplexity` (float, default `30`): clamped to `[2, (N-1)/3]`
    - `init` (str, default `"pca"`): initialization method
    - `random_state` (int, default `42`): random seed
    - `n_iter` (int, default `750`): number of optimization iterations
- **Returns:** `np.ndarray` of shape `(N, 2)`.
- **Example call:**
  ```python
  proj = run_tsne(vectors, {"perplexity": 30, "n_iter": 1000})
  ```

## Inputs & Outputs
- **In:** `(N, D)` numpy array of float vectors, parameter dict.
- **Out:** `(N, 2)` numpy array of 2D projected coordinates.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| N > 90, default perplexity 30 | Normal t-SNE result | Perplexity within valid range |
| N = 10, perplexity = 30 | Perplexity clamped to 3.0 `((10-1)/3)` | Prevents sklearn error |
| N = 3 | Perplexity clamped to 2.0 (minimum) | Minimum perplexity floor |
| sklearn >= 1.5 | Uses `max_iter` parameter | New API |
| sklearn < 1.5 | Falls back to `n_iter` parameter | `TypeError` caught, retried |

## Where This Is Used
- `backend/reduction_2d/base.py:77` -- imported lazily inside the `"tsne"` branch of `reduce_2d`

## Known Gotchas & Non-Obvious Behavior
- The sklearn import is **lazy** (inside the function body) to avoid pulling sklearn at module import time.
- Perplexity is clamped to `max(2.0, min(perplexity, max(2.0, (N-1)/3.0)))`. The `(N-1)/3` upper bound is stricter than sklearn's own `perplexity < N` requirement, providing extra safety margin.
- The `n_iter` vs `max_iter` compatibility shim uses a try/except on `TypeError`: it first tries `max_iter` (sklearn 1.5+), and falls back to `n_iter` if that keyword is not recognized.
- `learning_rate` is always set to `"auto"` (sklearn's recommended default since 1.2).

## Dependencies
- `numpy`
- `sklearn.manifold.TSNE` (imported lazily)
