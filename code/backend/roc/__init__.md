# __init__.py

## Purpose
Package initializer for the `roc` module. Re-exports the public API surface for ROC curve computation and threshold marker resolution.

## Motivation
Provides a single import point (`from backend.roc import label_cosine_scores, roc_curve_cosine, threshold_marker_xy`) so callers do not need to know the internal file layout.

## Re-exports
- `LabelScores`, `ROCCurve`, `label_cosine_scores`, `roc_curve_cosine` from `.curves`
- `threshold_marker_xy` from `.threshold`

## Dependencies
- `.curves`, `.threshold` (internal submodules)
