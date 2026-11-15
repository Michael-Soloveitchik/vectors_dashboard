"""Legacy Plotly updatemenu button style — kept for DELETE_ME_LATER/ previews.

Live toggles in the dashboard use tight-CSS ipywidgets (see ``toggles.py``);
this dict is only retained so the btn_preview sandbox scripts still run.
"""
from __future__ import annotations


__all__ = ["_BTN"]


_BTN = dict(
    bgcolor="rgba(56,60,68,0.92)",
    bordercolor="rgba(255,255,255,0.22)",
    borderwidth=1, showactive=False,
    font=dict(size=11, color="rgba(235,235,240,0.95)",
              family="ui-monospace, SFMono-Regular, Menlo, monospace"),
    pad=dict(l=8, r=8, t=4, b=4),
)
