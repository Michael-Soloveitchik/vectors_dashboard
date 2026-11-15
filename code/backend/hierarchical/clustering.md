# clustering.py

## Purpose
Runs scipy agglomerative linkage on a confusion matrix (similarity-based) and returns a leaf-ordered result. The confusion matrix is first converted from similarity to distance space (`distance = 1 - similarity`, clipped and symmetrized), then condensed into the upper-triangle vector format scipy requires. The output `HierarchyResult` carries the linkage matrix, the leaf permutation, and the reordered session IDs that the front-end uses to rearrange the confusion matrix when hierarchical mode is active.

## Motivation
The dashboard needs a canonical ordering of sessions driven by hierarchical clustering so the confusion matrix rows/columns can be sorted "by hierarchy" rather than alphabetically or by label. scipy's linkage provides this, but the input is a similarity matrix (cosine-based) rather than a distance matrix, so a conversion layer is necessary. Wrapping the result in `HierarchyResult` keeps the linkage matrix, leaf order, and metadata in one serializable object that downstream modules (dendrogram drawing, label assignment) can consume without re-running the clustering.

## Key Classes / Functions

### `HierarchyResult`
- **What it does:** Dataclass holding the full output of agglomerative clustering.
- **Fields:**
  - `Z` (np.ndarray): scipy linkage matrix, shape `(n-1, 4)`. Empty `(0, 4)` when `n < 2`.
  - `leaf_order` (np.ndarray): permutation of `[0..n)` giving leaves sorted by hierarchy.
  - `ordered_session_ids` (list[str]): `cm.session_ids` reordered by `leaf_order`.
  - `method` (str): linkage method echoed back (e.g. `"average"`).
  - `metric` (str): distance metric echoed back (e.g. `"cosine"`).
  - `n` (int): number of leaves (sessions).

### `_similarity_to_condensed_distance(matrix)`
- **What it does:** Converts a square similarity matrix in [-1, 1] to a condensed distance vector suitable for `scipy.cluster.hierarchy.linkage`.
- **Args:**
  - `matrix` (np.ndarray): square similarity matrix, shape `(n, n)`.
- **Returns:** np.ndarray, condensed distance vector of length `n*(n-1)/2`.

### `cluster(cm, method, metric, optimal_ordering)`
- **What it does:** Runs linkage on a `ConfusionMatrix` and returns a `HierarchyResult` with the leaf-order permutation.
- **Args:**
  - `cm` (ConfusionMatrix): confusion matrix with `.matrix` and `.session_ids`.
  - `method` (str): scipy linkage method. Default `"average"`.
  - `metric` (str): echoed into the result for the front-end. Default `"cosine"`.
  - `optimal_ordering` (bool): if True, scipy reorders leaves to minimize adjacent-leaf distance (visually smoother, more expensive). Default `False`.
- **Returns:** `HierarchyResult`.
- **Example call:**
  ```python
  from backend.hierarchical.clustering import cluster
  result = cluster(cm, method="average", optimal_ordering=True)
  ```

## Inputs & Outputs
- **In:** `ConfusionMatrix` object from `backend.confusion_matrix.matrix` with `.matrix` shape `(n, n)` (float similarity values in [-1, 1]) and `.session_ids` list of length `n`.
- **Out:** `HierarchyResult` with `Z` shape `(n-1, 4)`, `leaf_order` shape `(n,)`, and `ordered_session_ids` length `n`.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| `n >= 2` | Full linkage with valid Z, leaf_order | Normal path |
| `n == 1` | `Z` is `(0, 4)` empty, `leaf_order` is `[0]` | Degenerate case, no merges possible |
| `n == 0` | `Z` is `(0, 4)` empty, `leaf_order` is `[]` | Degenerate case |
| Asymmetric similarity matrix | Symmetrized via `0.5 * (dist + dist.T)` | Handles float drift |
| Similarity values outside [-1, 1] | Clipped to [-1, 1] before conversion | Prevents negative distances |

## Where This Is Used
- `backend/hierarchical/dendrogram.py` -- consumes `HierarchyResult` to compute Plotly traces.
- `backend/hierarchical/labels.py` -- consumes `HierarchyResult.Z` for label assignment.
- `frontend/callbacks/render_helpers.py:20` -- calls `cluster()` directly.
- `TESTS/tests/test_hierarchical.py`, `test_hier_labels.py`, `test_optimal_ordering.py`, `test_figures.py`.

## Known Gotchas & Non-Obvious Behavior
- The `metric` parameter is **not used** for actual distance computation -- distance is always `1 - similarity`. The parameter is echoed into the result purely for front-end display/metadata purposes.
- `squareform(dist, checks=False)` disables scipy's symmetry check since the matrix is already manually symmetrized; this avoids false failures from floating-point noise.
- `optimal_ordering=True` is significantly more expensive (TSP-style sort) and should only be enabled when visual smoothness matters more than latency.

## Dependencies
- `numpy`
- `scipy.cluster.hierarchy` (linkage, leaves_list)
- `scipy.spatial.distance` (squareform)
- `backend.confusion_matrix.matrix.ConfusionMatrix` (internal)
