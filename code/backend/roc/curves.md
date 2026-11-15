# curves.py

## Purpose
Computes per-label ROC and DET curves for evaluating how well GMM centroids separate "member" sessions from the rest of the dataset. Each session is scored by its maximum cosine similarity to the label's centroids. The ROC curve sweeps a threshold over those scores with ground truth drawn from the sessions that were originally used to fit the centroids (not the current UI label selection). A DET curve variant applies probit (normal deviate) warping to both axes for the biometrics-standard view.

## Motivation
The dashboard needs a verification-style evaluation: given a label's centroids, how reliably can cosine similarity distinguish members from non-members? Standard ROC curves answer this, but speaker/biometric verification systems focus on the very-low-FPR region, so this module also provides a DET curve with probit-scaled axes that linearize Gaussian-distributed scores and spread the low-error region. Separating score computation from curve computation allows the front-end to cache scores and only recompute curves when thresholds change.

## Key Classes / Functions

### `LabelScores`
- **What it does:** Dataclass holding per-label score vectors aligned to the full session list.
- **Fields:**
  - `label` (str): label name.
  - `session_ids` (list[str]): all session IDs in the dataset.
  - `scores` (np.ndarray): cosine similarity scores, shape `(N,)`, higher = more similar.
  - `is_member` (np.ndarray): boolean array, shape `(N,)`, True for sessions that belong to this label.

### `ROCCurve`
- **What it does:** Dataclass holding a log-FPR-ready ROC curve for one label.
- **Fields:**
  - `label` (str): label name.
  - `fpr` (np.ndarray): false positive rates.
  - `tpr` (np.ndarray): true positive rates.
  - `thresholds` (np.ndarray): decision thresholds corresponding to each (fpr, tpr) point.
  - `auc` (float): area under the ROC curve. `NaN` for degenerate cases.

### `DETCurve`
- **What it does:** Dataclass holding a DET curve with probit-warped axes.
- **Fields:**
  - `label` (str): label name.
  - `x_ppf` (np.ndarray): `norm.ppf(fpr)` -- probit-scaled x axis.
  - `y_ppf` (np.ndarray): `norm.ppf(fnr)` -- probit-scaled y axis.
  - `fpr`, `fnr` (np.ndarray): raw rates before warping.
  - `thresholds` (np.ndarray): decision thresholds.
  - `eer` (float): equal-error rate (FPR == FNR crossing point).

### `_cosine_score_against_centroids(vectors, means)`
- **What it does:** Computes the maximum cosine similarity from each vector to any centroid.
- **Args:**
  - `vectors` (np.ndarray): shape `(N, D)`, dataset embedding vectors.
  - `means` (np.ndarray): shape `(K, D)`, centroid vectors for one label.
- **Returns:** np.ndarray, shape `(N,)`, scores in `[-1, 1]`.

### `label_cosine_scores(centroid_set, session_ids, vectors)`
- **What it does:** Computes `LabelScores` for every label in a `CentroidSet`.
- **Args:**
  - `centroid_set` (CentroidSet): from `backend.centroids.gmm`, holds per-label centroids and member session IDs.
  - `session_ids` (list[str]): all session IDs in the dataset.
  - `vectors` (np.ndarray): shape `(N, D)`, embedding vectors aligned to `session_ids`.
- **Returns:** `dict[str, LabelScores]` keyed by label name.
- **Example call:**
  ```python
  scores_dict = label_cosine_scores(centroid_set, session_ids, vectors)
  ```

### `filter_to_window(sc, low, high)`
- **What it does:** Returns a new `LabelScores` containing only the samples whose score ∈ `[low, high]`. Used by the front end to compute "AUC | score ∈ window" per label. At the full score range this is a no-op.
- **Args:**
  - `sc` (LabelScores): full unfiltered scores.
  - `low`, `high` (float): window bounds.
- **Returns:** new `LabelScores` — `session_ids`, `scores`, and `is_member` are all masked consistently.

### `roc_curve_cosine(scores)`
- **What it does:** Computes an ROC curve from a `LabelScores` object. Uses `sklearn.metrics.roc_curve` with `drop_intermediate=False` — every scored sample produces exactly one step, so the curve has `len(scores.scores) + 1` points. Narrowing a window by one threshold step adds/removes exactly one point from the curve.
- **Args:**
  - `scores` (LabelScores): per-label scores with ground truth membership (usually after `filter_to_window`).
- **Returns:** `ROCCurve` with fpr, tpr, thresholds, and AUC.
- **AUC edge cases:** returns `NaN` when all samples are of one class (no positives OR no negatives in the window) — front end drops such labels before rendering.

### `det_curve_from_roc(roc)`
- **What it does:** Converts a `ROCCurve` to DET axes with probit scaling and computes the EER.
- **Args:**
  - `roc` (ROCCurve): input ROC curve.
- **Returns:** `DETCurve` with probit-warped axes, raw rates, and EER.
- **Example call:**
  ```python
  det = det_curve_from_roc(roc)
  print(f"EER: {det.eer:.4f}")
  ```

## Inputs & Outputs
- **In:** `CentroidSet` (from `backend.centroids.gmm`) with `.centroids` dict of `{label: (K, D) ndarray}` and `.member_session_ids` dict of `{label: list[str]}`. Dataset `vectors` shape `(N, D)` and `session_ids` list of length `N`.
- **Out:** Per-label `LabelScores`, `ROCCurve`, and `DETCurve` objects with arrays aligned to the dataset.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal label with members and non-members | Valid ROC with AUC in [0, 1] | Standard path |
| All sessions are members (no negatives) | AUC = `NaN`, diagonal ROC | Degenerate: no negatives to score against |
| No sessions are members (no positives) | AUC = `NaN`, diagonal ROC | Degenerate: no positives |
| Single centroid per label | Works, max-over-centroids is identity | K=1 case |
| Zero-norm vectors | Safe: divided by `max(norm, 1e-12)` | Avoids division by zero |
| DET with fpr/fnr at 0.0 or 1.0 | Clipped to `[eps, 1-eps]` before `norm.ppf` | Prevents infinite probit values |

## Where This Is Used
- `backend/roc/__init__.py` -- re-exported.
- `frontend/figures/roc_curves.py` -- imports `ROCCurve` for Plotly rendering.
- `frontend/figures/det_curves.py` -- imports `DETCurve`, `ROCCurve`, `det_curve_from_roc`.
- `frontend/callbacks/render_roc.py` -- calls `label_cosine_scores` and `roc_curve_cosine`.
- `TESTS/tests/test_roc.py`, `test_figures_more.py`, `test_det_curves.py`.

## Known Gotchas & Non-Obvious Behavior
- Ground truth comes from `centroid_set.member_session_ids`, not from the current UI label selection. This is intentional: the ROC evaluates how well the fitted centroids identify the sessions they were trained on, independent of any subsequent label changes in the UI.
- `sklearn.metrics.roc_curve` is imported lazily inside `roc_curve_cosine()` to avoid a top-level sklearn dependency for modules that only need the dataclasses.
- `scipy.stats.norm` is imported lazily inside `det_curve_from_roc()` for the same reason.
- EER is estimated by finding the index where `|fpr - fnr|` is minimized, then averaging `fpr` and `fnr` at that index. This is an approximation; the true EER lies between two adjacent operating points.
- Cosine similarity is computed via L2-normalized dot product, not `scipy.spatial.distance.cosine`, for vectorized performance.

## Dependencies
- `numpy`
- `sklearn.metrics` (roc_curve, auc) -- lazy import
- `scipy.stats` (norm) -- lazy import
- `backend.centroids.gmm.CentroidSet` (internal)
