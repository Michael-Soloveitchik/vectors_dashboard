"""Per-label ROC curves driven by cosine similarity to GMM centroids.

Public surface:
    label_cosine_scores(centroid_set, session_ids, vectors) -> Scores
    roc_curve_cosine(scores) -> ROCCurve   # log-x ready
    threshold_marker_xy(curve, threshold)  -> (fpr, tpr, idx)
"""

from .curves import (
    LabelScores,
    ROCCurve,
    label_cosine_scores,
    roc_curve_cosine,
)
from .threshold import threshold_marker_xy

__all__ = [
    "LabelScores",
    "ROCCurve",
    "label_cosine_scores",
    "roc_curve_cosine",
    "threshold_marker_xy",
]
