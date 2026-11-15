"""Shared figure helpers."""
from __future__ import annotations

import hashlib
import json

import plotly.graph_objects as go

from ..theme import base_layout, tokens

THR = "thr"     # depends on thresholds → patched while a slider drags
SEL = "sel"     # depends on the selection
FOC = "foc"     # depends on the focused point / class


def dyn(**props: list[str]) -> dict:
    """Trace ``meta``: for each fast-path tag, the dotted trace props that
    must be resent, e.g. ``dyn(thr=["marker.symbol"], sel=["selectedpoints"])``.
    Everything else in the trace stays untouched on the client."""
    return {"dyn": props}


XY = ["x", "y"]


def sig(*parts) -> str:
    """Fingerprint of everything that fixes a figure's trace layout. A
    Patch is sent only when the browser's figure was built from the same
    fingerprint (it is echoed back as State) — never onto a missing or
    differently-shaped figure."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def home(xr, yr, slack: float = 0.5) -> dict:
    """Layout pieces for a fixed home view: explicit ranges, a zoom-out
    limit (``slack`` × the span beyond each side) and ``meta.home`` — what
    the ⌂ button / right-click restore (assets/sync.js)."""
    def axis(r):
        lo, hi = min(r), max(r)
        pad = slack * (hi - lo or 1.0)
        return dict(range=list(r), autorange=False,
                    minallowed=lo - pad, maxallowed=hi + pad)
    return {"xaxis": axis(xr), "yaxis": axis(yr),
            "meta": {"home": {"xaxis.range": list(xr), "yaxis.range": list(yr)}}}


def message_fig(text: str, theme: str | None) -> go.Figure:
    t = tokens(theme)
    fig = go.Figure()
    fig.update_layout(**base_layout(
        theme, xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=text, x=0.5, y=0.5, xref="paper",
                          yref="paper", showarrow=False,
                          font=dict(size=13, color=t["muted"]))]))
    return fig
