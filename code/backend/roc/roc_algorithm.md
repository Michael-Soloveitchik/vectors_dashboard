# ROC Pipeline — Algorithmic Steps

## Overview

The ROC measures how well GMM centroids separate label groups using cosine similarity. Only **aggregated (selected) points** are scored. Centroids are fit on a 50% random split so half the positive scores are unbiased.

## Step-by-Step Algorithm

### 1. Centroid Fitting (gmm.py → `fit_centroids`)

For each label L in the dataset:

1. Collect all sessions with label L → `all_sids`, `all_vecs`
2. **Holdout split** (if N ≥ 4 sessions):
   - Randomly permute indices (seed=42 for reproducibility)
   - First 50% → `fit_sids` (used to train the GMM)
   - Remaining 50% → held out (centroids never saw these vectors)
   - Labels with < 4 sessions: all used for fitting
3. Fit `GaussianMixture(n_components=K)` on the **fit portion only**
4. Store:
   - `centroids[L]` = GMM means, shape (K, D)
   - `member_session_ids[L]` = ALL sessions of label L (ground truth)

### 2. Cosine Similarity Scoring (curves.py → `label_cosine_scores`)

Only **aggregated points** are scored (no aggregation → no ROC).

For each label L:

1. Get centroids: `means = centroids[L]`, shape (K, D)
2. L2-normalize all vectors and centroids
3. Similarity matrix: `sims = v_normed @ c_normed.T`, shape (N, K)
4. Per-session score = **max over K centroids**: `score[i] = max_k(sims[i, k])`
5. Ground truth: `is_member[i] = session_ids[i] in member_session_ids[L]`
   - ALL label members are positives — not just the held-out half
   - The held-out half provides unbiased scores; the fit half scores higher (optimistic) — together they produce a realistic, non-step-function ROC

### 3. ROC Curve Construction (curves.py → `roc_curve_cosine`)

1. `sklearn.metrics.roc_curve(is_member, scores)`
2. Sweeps all unique score values as thresholds
3. At each threshold T: TPR = TP/P, FPR = FP/N
4. AUC = area under the (FPR, TPR) curve
5. More aggregated points → smoother curve (fewer steps)

### 4. Threshold Marker (threshold.py → `threshold_marker_xy`)

The per-label threshold value maps to a dot on the ROC curve:

1. `i = argmin(|curve.thresholds - T|)`
2. Returns `(FPR[i], TPR[i])` — the operating point

### 5. Scatter Plot Threshold Circles (left_scatter_helpers.py)

For each label L with threshold T > 0:

1. `radius_2d = T * base_radius_2d` per centroid
2. Draw union outline as dashed-line trace
3. `uirevision="left-scatter"` preserves zoom/pan across redraws

## Data Flow

```
User selects points (aggregation) → fits centroids
  │
  ├─ _on_find:
  │    ├─ fit_centroids(holdout=0.5) → centroids on 50% of each label
  │    ├─ builds ROC figure instantly → fig_right (no threshold markers yet)
  │    └─ writes store_centroids
  │
  ├─ _compute_roc fires from store_centroids:
  │    ├─ scores ONLY aggregated points against centroids (cosine sim)
  │    ├─ ground truth = member_session_ids (ALL label members)
  │    └─ writes store_roc_curves
  │
  ├─ _render_right fires from store_roc_curves:
  │    ├─ reads store_thresholds_per_label
  │    └─ draws ROC with threshold marker dots
  │
  └─ On each new aggregated point:
       ├─ _compute_roc re-fires → new ROC on expanded set
       └─ _render_right redraws with markers
```

## Key Design Decisions

- **ROC scores only aggregated points**: no aggregation → empty ROC
- **Ground truth = all label members** (not just held-out): gives enough positives for a smooth curve; the holdout only affects centroid quality
- **Zoom preserved**: `uirevision` on the scatter layout prevents reset on control changes
- **Threshold circles**: dashed lines in 2D projected space (note: 2D distances ≠ high-D cosine distances due to projection distortion)
