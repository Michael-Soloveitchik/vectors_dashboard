# Threshold Synchronization Fix Plan

## Current State (2026-05-07)

Fan-out from `ctrl_pair_cos_thr` → `store_thresholds` mirrors single value to all per-class keys. Per-class sliders update value display ✓. PCA dashed rings recompute via `_override_pca_rings` data-driven r-max ✓. But:

| Indicator | Status | Reason |
|---|---|---|
| Per-class slider input values | ✓ sync | fan-out callback writes all keys |
| PCA dashed rings | partial | rings update but radii vary wildly per class (0.3→1.09 at same thr) — rings can exceed PCA viewport (>1.0) when no points filter out |
| Dist vertical thr lines (toy) | ✗ frozen | toy `build_dist_fig` consumes RAW thr from `_norm_to_raw_thresholds`. For methodologies with non-monotone normalize_score (ACPAN clips), raw_thr lookup degenerates to extremes → vertical line stuck at boundary value |
| Dist vertical thr lines (selection overlay) | ✓ uses norm | `_append_selection_overlay` line 631 plots at `x=tau_norm` directly |
| ROC operating-point markers | ✗ frozen | toy `plot/roc.py:103 _op_point(y_c, raw_scores, raw_thr)` uses RAW thr from `_norm_to_raw_thresholds`. Same degeneracy as dist |
| ROC op label "device_xx @ thr=0.07" | ✗ shows raw not norm | label confuses user; shows raw cosine boundary not the slider's norm value |

## Root Cause

**Two threshold spaces, one slider:**
- Slider value lives in **normalized** [0,1] space
- Toy `build_dist_fig` + `build_roc_fig` both expect **raw** thresholds in pre-normalize cosine/score space
- Translation via `_norm_to_raw_thresholds` (argmin |norm-target|) is **non-injective** for clipped normalize_score (ACPAN, ArcCosineSVM): multiple raw values map to same norm value → freeze
- `_op_point(y_c, raw_scores, raw_thr)` then computes (FPR, TPR) at the frozen raw_thr → marker stuck

## Fix Plan (Priority Order)

### P0 — Single Source of Truth
Stop translating norm→raw. Push normalization INTO the toy plots so they accept norm thresholds directly.

1. **`plot/roc.py`**: pre-normalize raw scores via `rep.normalize_score` BEFORE entering toy. Compute `_op_point(y_c, normalize_score(raw_scores), norm_thr)`. Op marker now lands at the (FPR, TPR) corresponding to the slider's norm value, monotone for all methodologies.
2. **`plot/dist.py`**: same — render distribution X-axis in normalized space; vertical thr lines at `x=norm_thr`.
3. **`figures.py:_norm_to_raw_thresholds`**: deprecate. Keep as no-op identity for backward compat, remove all callers.
4. **`plot/roc.py:105 op_label`**: change to `f"{lbl} @ thr={norm_thr:.2f}"`.

### P1 — PCA Ring Visual Cap
- Cap displayed ring radius to `max_data_radius * 1.05` so rings never escape the viewport
- Rings exceeding cap → render as dashed at viewport edge with a "thr below all-pass" hint

### P2 — Step-Buttons Boundary Snap
- ◀/▶ buttons already use `np.unique(rep.normalize_score(votes[:, k]))` boundaries — verified working
- Add a **drift assertion** in dev mode: after slider update, assert `|norm_at_op_marker - slider_value| < 1e-6` for the just-snapped boundary

### P3 — CM Methodology Distinction (already correct)
Cosine-family methodologies sharing the cosine `_pair_kernel` produce identical CM by **mathematical definition**. Not a bug. Document in tooltip: "CM = `score_xy(v,u)` — methodologies sharing kernel produce identical pair-scores; predict() votes are where they diverge."

### P4 — Decouple ROC Sel-vs-Rest from Per-Class Slider
The selection-thr slider drives BOTH per-class lines AND the sel-vs-rest op marker. User wants:
- One slider drives all per-class indicators (current ✓)
- Sel-vs-rest op marker uses the SAME slider value (currently has its own thr=0.50 fallback at `pair-cos @ thr=0.50` legend)
- Drop the static `pair-cos @ thr=0.50` and `pair-norm @ thr=0.00` traces — they confuse vs the per-class sweep

## Verification Protocol

```python
# Probe at thr ∈ {0.0, 0.3, 0.7, 0.99}:
#   slider value
#   PCA ring max-x per class
#   dist vertical thr line x0 per class (norm space)
#   ROC op marker (FPR, TPR) per class
#   ROC op marker label "@ thr=X.XX"
# Assert:
#   all per-class slider values == slider value (already ✓)
#   monotone ring radius shrinkage as thr increases
#   monotone (FPR, TPR) movement on each class curve
#   op label X.XX == slider value
```

## Estimated Effort

- P0: 2h (touch plot/roc.py + plot/dist.py + figures.py)
- P1: 30m
- P2: 30m
- P3: 10m (docstring only)
- P4: 1h

Total: ~4h to fully sync all 4 panels in normalized space with provable drift = 0.
