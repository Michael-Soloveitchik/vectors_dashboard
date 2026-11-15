"""Color transforms applied to the confusion matrix before rendering.

Spec point 26: ABS first, THEN power, THEN clip to [lo, hi]. The clip
is enforced regardless of how the user set the power so the colormap
range is always meaningful.
"""

from __future__ import annotations

import numpy as np

from .matrix import ConfusionMatrix


def apply_color_transform(
    cm: ConfusionMatrix,
    abs_: bool,
    power: float,
    lo: float,
    hi: float,
) -> ConfusionMatrix:
    """Return a new ConfusionMatrix with abs/power/clip applied.

    Power preserves sign on negative inputs (since `(-x)**0.5` would
    NaN otherwise) — we apply `sign(m) * |m|**p`. When abs_ is on the
    sign is always positive so this collapses to the obvious form.
    """
    if cm.n == 0:
        return cm

    m = cm.matrix
    if abs_:
        m = np.abs(m)
    if power != 1.0:
        sign = np.sign(m)
        m = sign * (np.abs(m) ** float(power))

    lo_eff = float(lo)
    hi_eff = float(hi)
    if lo_eff > hi_eff:
        lo_eff, hi_eff = hi_eff, lo_eff
    m = np.clip(m, lo_eff, hi_eff)

    return ConfusionMatrix(
        matrix=m.astype(np.float32, copy=False),
        session_ids=list(cm.session_ids),
        is_full=cm.is_full,
        meta={
            **cm.meta,
            "abs": bool(abs_),
            "power": float(power),
            "color_lo": lo_eff,
            "color_hi": hi_eff,
        },
    )
