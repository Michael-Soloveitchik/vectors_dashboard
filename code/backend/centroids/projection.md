# projection.py

## Purpose
Projects GMM centroids into the same 2D coordinate space as the data points by running the chosen dimensionality reduction (PCA, t-SNE, UMAP) on the concatenated matrix `[points; centroids; unit_ball_samples]` and splitting the result back. Also computes per-centroid base radii and covariance ellipses in the 2D projected space, which the front-end uses to draw threshold circles and ellipses around each centroid.

## Motivation
For non-linear projection methods (t-SNE, UMAP), you cannot project centroids independently and expect them to land in the correct position relative to the data points. The only reliable approach is to include centroids in the input to the reduction so that the algorithm places them consistently with the surrounding data. Additionally, a "radius" in raw embedding space has no direct pixel meaning in 2D after non-linear projection, so this module calibrates the projected radius by embedding synthetic unit-sphere samples around each centroid and measuring their projected spread.

## Key Classes / Functions

### `CombinedProjection`
- **What it does:** Dataclass holding the split results of a joint reduction over points + centroids + unit-ball samples.
- **Fields:**
  - `points_2d` (np.ndarray): shape (N_points, 2), projected coordinates of the original data.
  - `centroids_2d` (dict[str, np.ndarray]): label -> (k, 2), projected centroid positions.
  - `centroid_labels` (list[str]): flat list of label strings, one per centroid in concatenation order.
  - `result` (ReductionResult): the full reduction result on the joined matrix (from `reduction_2d`).
  - `base_radii_2d` (dict[str, np.ndarray]): label -> (k,) per-centroid median projected distance of a unit-norm vector. Multiply by the user's threshold to get the visible circle radius.
  - `ellipses_2d` (dict[str, list[dict]]): label -> list of k ellipse descriptors, each with keys `cx`, `cy`, `rx`, `ry`, `angle`. Derived from the 2x2 covariance of projected unit-ball samples.

### `project_with_centroids`
- **What it does:** Concatenates data points, centroids, and unit-sphere samples into one matrix, runs 2D reduction, then splits and annotates the result.
- **Args:**
  - `points` (np.ndarray): shape (N, D), the original high-dimensional data.
  - `centroid_set` (CentroidSet): output of `fit_centroids` containing per-label centroids.
  - `method` (str): reduction method name (e.g. "pca", "tsne", "umap").
  - `params` (dict): method-specific parameters passed to `reduce_2d`.
  - `cache` (ReductionCache | None, default=None): optional reduction cache for reuse.
  - `n_unit_ball_samples` (int, default=16): number of random directions sampled per centroid on the unit sphere, used to estimate the projected radius and ellipse shape.
  - `rng_seed` (int, default=42): seed for generating unit-ball sample directions.
- **Returns:** `CombinedProjection` containing all split results, radii, and ellipses.
- **Raises:** No exceptions directly; degenerate cases (empty centroids, too few ball samples) produce zero-radius or zero-size ellipses.
- **Example call:**
  ```python
  proj = project_with_centroids(
      points=vecs, centroid_set=cset,
      method="umap", params={"n_neighbors": 15},
      n_unit_ball_samples=32,
  )
  ```

## Inputs & Outputs

**IN:**
- `points`: (N, D) float array of data vectors.
- `centroid_set`: `CentroidSet` from `gmm.fit_centroids`.
- Reduction method name and params.

**OUT:**
- `CombinedProjection` containing:
  - `points_2d`: (N, 2) float array.
  - `centroids_2d`: dict of (k, 2) float arrays per label.
  - `base_radii_2d`: dict of (k,) float arrays per label.
  - `ellipses_2d`: dict of lists of `{cx, cy, rx, ry, angle}` dicts per label.
  - `result`: full `ReductionResult` on the joined (N + sum_k + sum_k*S, 2) matrix.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal case | Full projection with radii and ellipses | Standard path |
| Empty centroid_set (no labels) | `centroids_2d`, `base_radii_2d`, `ellipses_2d` all empty dicts | `joined = points` only |
| `n_unit_ball_samples` < 2 per centroid | Ellipse has `rx=0, ry=0, angle=0` | Fallback when `ball_j.size < 4` |
| Single centroid with many samples | Works normally | k=1 for that label |
| Zero-norm direction vectors | Handled by `+ 1e-12` epsilon in normalization | Prevents division by zero |

## Where This Is Used

- `backend/centroids/__init__.py` -- re-exported as `project_with_centroids`.
- `frontend/callbacks/render.py:24` -- called during scatter plot rendering to place centroids in 2D.
- Test files `test_centroids.py`, `test_ellipses.py` exercise projection and ellipse computation.

## Known Gotchas & Non-Obvious Behavior

- The unit-ball samples are generated in the raw D-dimensional space (not 2D). Their purpose is purely to calibrate how a unit-length displacement around each centroid maps into the projected 2D plane. They are not visualized directly.
- The radius approximation is linear-around-the-centroid: exact for PCA, reasonable for small thresholds under t-SNE/UMAP, but increasingly inaccurate for very large thresholds where non-linear distortion accumulates.
- The cache key includes `n_per`, `labels`, `S`, and `rng_seed` via the `extra` tuple, so changing any of these forces a fresh reduction even if the method and params are unchanged.
- Ellipse `angle` is in degrees (not radians), computed via `np.degrees(np.arctan2(...))`.
- The covariance ellipse uses `np.linalg.eigh` (symmetric eigendecomposition). Eigenvalues are clamped to non-negative before taking sqrt to avoid NaN from floating-point noise on near-zero eigenvalues.
- Two separate cursor-tracking loops iterate over labels/centroids (one for radii, one for ellipses). This duplication exists because the ellipse computation was added later. Both loops produce identical cursor advancement.

## Dependencies

- `numpy` -- all array and linear algebra operations.
- `backend.reduction_2d` -- `reduce_2d`, `ReductionCache`, `ReductionResult`.
- `backend.centroids.gmm` -- `CentroidSet`.
