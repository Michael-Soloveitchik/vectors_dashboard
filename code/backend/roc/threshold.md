# threshold.py

## Purpose
Maps a user-selected threshold value (from a control panel slider) onto a per-label ROC curve. Finds the closest operating point on the curve and returns its `(FPR, TPR, index)` so the front-end can drop a marker at that exact position on the ROC or DET plot.

## Motivation
The dashboard exposes a cosine-similarity threshold slider. When the user drags it, the ROC/DET plot needs to show where that threshold falls on each label's curve. This is a simple nearest-threshold lookup, but isolating it in its own module keeps the curve computation (which is expensive) separate from the threshold resolution (which is cheap and called on every slider move).

## Key Classes / Functions

### `threshold_marker_xy(curve, threshold)`
- **What it does:** Returns the `(fpr, tpr, index)` of the ROC curve point whose threshold is closest to the given value.
- **Args:**
  - `curve` (ROCCurve): a computed ROC curve from `curves.py`.
  - `threshold` (float): the user-selected cosine similarity threshold.
- **Returns:** `tuple[float, float, int]` -- `(fpr, tpr, idx)`. If the curve has no thresholds, returns `(NaN, NaN, -1)`.
- **Example call:**
  ```python
  from backend.roc.threshold import threshold_marker_xy
  fpr, tpr, idx = threshold_marker_xy(roc_curve, threshold=0.75)
  ```

## Inputs & Outputs
- **In:** `ROCCurve` object (with `.thresholds`, `.fpr`, `.tpr` arrays) and a scalar `threshold` float.
- **Out:** Tuple of `(fpr, tpr, idx)` where `idx` is the index into the curve arrays.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal curve with many points | Closest threshold point | Nearest by absolute difference |
| Empty thresholds array | `(NaN, NaN, -1)` | Degenerate ROC from all-positive or all-negative labels |
| Threshold outside curve range | Snaps to nearest endpoint | `argmin(abs)` handles this naturally |
| Exact threshold match | Returns that exact point | No interpolation, exact index |

## Where This Is Used
- `backend/roc/__init__.py` -- re-exported.
- `frontend/figures/roc_curves.py:16` -- places threshold marker on ROC plot.
- `frontend/figures/det_curves.py:17` -- places threshold marker on DET plot.
- `TESTS/tests/test_roc.py` -- tested alongside curve computation.

## Known Gotchas & Non-Obvious Behavior
- No interpolation is performed between curve points. The marker snaps to the nearest discrete operating point, which may be visually off if the curve is coarse (few threshold values).
- The threshold is cast to `float()` before comparison to handle numpy scalar inputs.

## Dependencies
- `numpy`
- `.curves.ROCCurve` (internal)
