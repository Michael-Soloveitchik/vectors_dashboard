"""Pair-level operating-point marker builder — shared by build_roc_fig."""
from __future__ import annotations

import plotly.graph_objects as go

from .scoring import _op_point


__all__: list[str] = []


def _pair_op_trace(y_pair, scores, thr, *, color, name):
    """Operating-point fat marker for a pair-level metric.

    Mirrors the per-device pattern: trace always exists so widget callbacks
    can patch (x, y) in place; visibility flips when a threshold is supplied.
    """
    if isinstance(thr, (int, float)) and not isinstance(thr, bool):
        op_fpr, op_tpr = _op_point(y_pair, scores, float(thr))
        visible = True
        label = f"{name} @ thr={thr:.2f}"
        cd = [float(thr)]
    else:
        op_fpr, op_tpr = 0.0, 0.0
        visible = False
        label = f"{name} @ thr=—"
        cd = [float("nan")]
    return go.Scatter(
        x=[op_fpr], y=[op_tpr], mode="markers",
        name=label, legendgroup=name, showlegend=False,
        marker=dict(size=13, color=color, symbol="circle",
                    line=dict(width=2, color="rgba(255,255,255,0.9)")),
        hovertemplate=(f"<b>{name}</b> operating point<br>"
                       "thr=%{customdata:.3f}<br>"
                       "fpr=%{x:.3f}<br>tpr=%{y:.3f}<extra></extra>"),
        customdata=cd,
        visible=visible,
    )
