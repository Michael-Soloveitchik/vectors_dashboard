"""Map a user-selected threshold onto the ROC curve.

The control panel slider gives a single distance value `t`. We find
the closest threshold on the per-label ROC curve and return the
(FPR, TPR, idx) so the front-end can drop a marker exactly there.
"""

from __future__ import annotations

import numpy as np

from .curves import ROCCurve


def threshold_marker_xy(
    curve: ROCCurve,
    threshold: float,
) -> tuple[float, float, int]:
    """Return (fpr, tpr, idx) interpolated on the ROC at ``threshold``.

    sklearn thresholds are sorted DESCENDING (highest first).
    We interpolate FPR/TPR linearly between the two bracketing
    thresholds so the big marker slides smoothly as the user drags
    the slider — even when there are few data points.
    """
    if curve.thresholds.size == 0:
        return (float("nan"), float("nan"), -1)
    thr = float(threshold)
    ts = curve.thresholds  # descending order
    # Clamp to range
    if thr >= ts[0]:
        return float(curve.fpr[0]), float(curve.tpr[0]), 0
    if thr <= ts[-1]:
        n = len(ts) - 1
        return float(curve.fpr[n]), float(curve.tpr[n]), n
    # Binary search: find idx where ts[idx-1] > thr >= ts[idx]
    idx = int(np.searchsorted(-ts, -thr))
    if idx == 0:
        idx = 1
    if idx >= len(ts):
        idx = len(ts) - 1
    t_hi = float(ts[idx - 1])
    t_lo = float(ts[idx])
    span = t_hi - t_lo
    if span < 1e-15:
        return float(curve.fpr[idx]), float(curve.tpr[idx]), idx
    alpha = (thr - t_lo) / span
    fpr = float(curve.fpr[idx]) + alpha * float(curve.fpr[idx - 1] - curve.fpr[idx])
    tpr = float(curve.tpr[idx]) + alpha * float(curve.tpr[idx - 1] - curve.tpr[idx])
    return fpr, tpr, idx
