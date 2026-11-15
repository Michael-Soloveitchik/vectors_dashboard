"""Hover-text and operating-point helpers shared across ROC figures."""
from __future__ import annotations

import numpy as np

from backend.representations.convention import threshold_compare_fn


__all__: list[str] = []  # helpers only — imported directly by dependents


def _hover(fpr, tpr, thr):
    return [
        f"thr={t:.2f}<br>fpr={f:.2f}<br>tpr={p:.2f}" if np.isfinite(t)
        else f"fpr={f:.2f}<br>tpr={p:.2f}"
        for t, f, p in zip(thr, fpr, tpr)
    ]


def _op_point(y_true: np.ndarray, scores: np.ndarray, thr: float) -> tuple[float, float]:
    """Compute (FPR, TPR) at a given threshold — positive iff score
    is convention-closer than thr (polarity-aware via
    ``threshold_compare_fn``)."""
    pred = threshold_compare_fn(scores, thr)
    pos = y_true == 1
    neg = ~pos
    tpr = float((pred & pos).sum() / max(pos.sum(), 1))
    fpr = float((pred & neg).sum() / max(neg.sum(), 1))
    return fpr, tpr
