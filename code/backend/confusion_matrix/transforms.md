# transforms.py

## Purpose
Applies a chain of color transforms (absolute value, power scaling, clipping) to a confusion matrix before it is rendered as a heatmap. This is the post-processing step described in spec point 26 that lets users adjust contrast and range of the cosine-similarity visualization without recomputing the underlying matrix.

## Motivation
Raw cosine similarity values often cluster near zero or one, making the heatmap hard to read. The power transform lets users expand or compress the dynamic range (e.g., `power=0.5` stretches small values apart), while the clip range `[lo, hi]` focuses the colormap on the region of interest. The fixed order (abs -> power -> clip) is enforced so the user always gets predictable behavior regardless of parameter combinations.

## Key Classes / Functions

### `apply_color_transform`
- **What it does:** Returns a new `ConfusionMatrix` with the matrix values transformed by abs, power, and clip operations in that order.
- **Args:**
  - `cm` (ConfusionMatrix): the input matrix to transform.
  - `abs_` (bool): whether to take absolute values first.
  - `power` (float): exponent to apply. Sign is preserved on negative values (so `(-0.5)**0.5` does not produce NaN). When `abs_` is True, sign is always positive.
  - `lo` (float): lower clip bound.
  - `hi` (float): upper clip bound.
- **Returns:** A new `ConfusionMatrix` with transformed float32 matrix values and updated `meta` dict recording the transform parameters.
- **Raises:** No exceptions. If `lo > hi`, they are silently swapped.
- **Example call:**
  ```python
  vis_cm = apply_color_transform(cm, abs_=True, power=0.5, lo=0.0, hi=1.0)
  ```

## Inputs & Outputs

**IN:**
- `cm`: `ConfusionMatrix` with an (N, N) float matrix (typically cosine similarity in [-1, 1]).
- Transform parameters: `abs_`, `power`, `lo`, `hi`.

**OUT:**
- New `ConfusionMatrix` with:
  - `matrix`: (N, N) float32 with transforms applied.
  - `session_ids`: unchanged (copy of input list).
  - `is_full`: unchanged.
  - `meta`: merged with `{"abs": bool, "power": float, "color_lo": float, "color_hi": float}`.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| `abs_=False, power=1.0, lo=-1, hi=1` | Matrix unchanged (identity transform) | No-op path |
| `abs_=True` | All negative values become positive | Applied before power |
| `power=0.5` on negative values (abs off) | Sign preserved: `sign(m) * abs(m)**0.5` | Prevents NaN from fractional powers of negatives |
| `power=0.0` | All non-zero values become 1.0 (or -1.0) | `abs(m)**0 = 1` times sign |
| `lo > hi` | Silently swapped to `lo, hi = hi, lo` | Defensive guard |
| `cm.n == 0` | Returns cm unchanged | Early return |

## Where This Is Used

- `backend/confusion_matrix/__init__.py` -- re-exported as `apply_color_transform`.
- Front-end callbacks that handle the color transform sliders and toggles.

## Known Gotchas & Non-Obvious Behavior

- The power transform uses `sign(m) * abs(m)**p` rather than a naive `m**p`. This is necessary because fractional powers of negative floats produce NaN in numpy. When `abs_` is True, sign is always +1, so this collapses to the simple form.
- The transform chain order is **abs -> power -> clip**, hardcoded and not configurable. Spec point 26 mandates this order.
- The output is always float32 via `astype(np.float32, copy=False)`.
- The returned `ConfusionMatrix` preserves all pre-existing `meta` keys and adds/overwrites the four transform-related keys.
- `lo` and `hi` are converted to Python floats before comparison and clipping, so passing numpy scalars or ints works fine.

## Dependencies

- `numpy` -- all numeric operations.
- `.matrix` -- `ConfusionMatrix`.
