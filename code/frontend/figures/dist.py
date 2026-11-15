"""Per-class score distribution — one lane per class.

Lane ``c`` shows the normalized score every sample gets FROM class ``c``:
target samples (true class = c) vs non-target samples, the standard
genuine / impostor picture, with the class threshold as a gold line and
the resulting false-reject / false-accept rates printed in the lane.

    hist   target area in the class colour, non-target in grey
           (each scaled to its own peak)
    box    target / non-target as box plots (quartiles, whiskers 5–95 %)
    strip  individual samples coloured by their TRUE class — shows which
           class the impostors come from

``domain="raw"`` draws the methodology's raw scores instead (one common
axis, real values on the ticks); thresholds stay defined on normalized
scores and are shown where they fall among the raw ones.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..engine.analysis import Analysis
from ..theme import base_layout, class_colors, rgba, tokens
from .common import XY, dyn

BINS = 48
LANE_PX = {"hist": 62, "strip": 78, "box": 54}
RUG_CAP = 400


def lane_height(k: int, view: str) -> int:
    return 58 + max(k, 1) * LANE_PX.get(view, 62)


def _step_poly(h: np.ndarray, edges: np.ndarray, base: float):
    """Closed staircase polygon rising from ``base`` (y axis is reversed,
    so 'up' on screen is toward smaller y)."""
    xs = np.repeat(edges, 2)
    ys = np.concatenate([[0.0], np.repeat(h, 2), [0.0]])
    return (np.concatenate([xs, [edges[0]]]),
            np.concatenate([base - ys, [base]]))


def _rates(an: Analysis, P: np.ndarray, tv: np.ndarray, shown) -> list[str]:
    out = []
    for k in range(an.k):
        pos = an.true_idx == k
        acc = P[:, k] >= tv[k]
        fr = float((~acc & pos).sum() / max(pos.sum(), 1))
        fa = float((acc & ~pos).sum() / max((~pos).sum(), 1))
        out.append(f"thr {shown[k]:.3f}<br>FR {fr:.1%} · FA {fa:.1%}")
    return out


def _boundary(p: np.ndarray, x: np.ndarray, t: float) -> float:
    """Where the threshold ``p >= t`` falls on another (monotone) scale x."""
    acc = p >= t
    if acc.all() or not acc.any():
        rising = x[np.argmax(p)] >= x[np.argmin(p)]
        low = acc.all() == bool(rising)        # everything is above / below
        return float(x.min() if low else x.max())
    a, r = x[acc], x[~acc]
    return float((a.min() + r.max()) / 2 if a.min() >= r.max()
                 else (a.max() + r.min()) / 2)


def _five(v: np.ndarray) -> list:
    return (np.percentile(v, [5, 25, 50, 75, 95]).tolist() if v.size
            else [None] * 5)


def dist_fig(an: Analysis, *, alpha, thr: dict | None, theme,
             view: str = "hist", sel_rows: list[int] | None = None,
             focus_rows: list[int] | None = None,
             domain: str = "norm") -> go.Figure:
    t = tokens(theme)
    Pn = an.norm(alpha)
    K = an.k
    cmap = class_colors(sorted(set(an.y.tolist())))
    tv = an.thr_vector(thr)
    edges = np.linspace(0.0, 1.0, BINS + 1)
    raw = domain == "raw"
    if raw:           # raw scores on a common [0, 1] display scale
        R = np.asarray(an.raw, dtype=float)
        lo, hi = float(np.nanmin(R)), float(np.nanmax(R))
        P = (R - lo) / ((hi - lo) or 1.0)
        tx_k = [_boundary(Pn[:, k], P[:, k], tv[k]) for k in range(K)]
        real = lambda v: lo + np.asarray(v) * ((hi - lo) or 1.0)  # noqa: E731
    else:
        P, tx_k, real = Pn, [float(v) for v in tv], lambda v: v  # noqa: E731
    fig = go.Figure()
    # Invisible click targets along every lane: click = that class's
    # threshold moves there (callbacks/select.py). Drawn first, so real
    # points keep hover priority.
    gx = np.linspace(0.0, 1.0, 101)

    def as_thr(k: int) -> np.ndarray:      # display x → normalized threshold
        if not raw:
            return gx
        o = np.argsort(P[:, k])
        return np.interp(gx, P[o, k], Pn[o, k])
    fig.add_trace(go.Scatter(
        x=np.tile(gx, K), y=np.repeat(np.arange(K), gx.size) + 0.0,
        mode="markers", marker=dict(size=14, opacity=0.001, color=t["gold"]),
        customdata=[[lbl, float(v)] for k, lbl in enumerate(an.labels)
                    for v in as_thr(k)],
        hovertemplate="<b>%{customdata[0]}</b><br>click: threshold → "
                      "%{customdata[1]:.2f}<extra></extra>",
        showlegend=False, name="lane-click"))

    if view == "strip":
        rng = np.random.default_rng(42)
        cap = int(np.clip(60000 // max(K * K, 1), 40, 400))
        for lbl in sorted(set(an.y.tolist())):
            idx = np.flatnonzero(an.y == lbl)
            if idx.size > cap:
                idx = rng.choice(idx, cap, replace=False)
            own = an.labels.index(lbl) if lbl in an.labels else -1
            xs = P[idx, :].T.reshape(-1)                  # lane-major
            lane = np.repeat(np.arange(K), idx.size)
            off = np.where(lane == own, -0.2, 0.18)
            jit = rng.uniform(-0.13, 0.13, size=lane.size)
            fig.add_trace(go.Scattergl(
                x=xs, y=lane + off + jit, mode="markers",
                name=lbl, legendgroup=lbl,
                marker=dict(size=4, opacity=0.6,
                            color=cmap.get(lbl, t["neutral"])),
                customdata=np.asarray(an.labels, dtype=object)[lane],
                hovertemplate=(f"true <b>{lbl}</b><br>scored by "
                               "%{customdata}<br>%{x:.3f}<extra></extra>")))
    elif view == "box":
        non = np.array([_five(P[an.true_idx != k, k]) for k in range(K)],
                       dtype=object)
        keys = ("lowerfence", "q1", "median", "q3", "upperfence")
        fig.add_trace(go.Box(
            y=[k + 0.19 for k in range(K)], orientation="h", width=0.3,
            name="non-target", hoverinfo="x", showlegend=False,
            line=dict(color=rgba(t["neutral"], 0.9), width=1),
            fillcolor=rgba(t["neutral"], 0.28),
            **{q: non[:, i].tolist() for i, q in enumerate(keys)}))
        for k, lbl in enumerate(an.labels):
            col = cmap.get(lbl, t["neutral"])
            five = _five(P[an.true_idx == k, k])
            fig.add_trace(go.Box(
                y=[k - 0.19], orientation="h", width=0.3, hoverinfo="x",
                name=f"{lbl} target", legendgroup=lbl, showlegend=False,
                line=dict(color=col, width=1.4), fillcolor=rgba(col, 0.42),
                **{q: [five[i]] for i, q in enumerate(keys)}))
    else:
        nx, ny = [], []
        for k, lbl in enumerate(an.labels):
            pos = an.true_idx == k
            base = k + 0.42
            hn, _ = np.histogram(P[~pos, k], bins=edges)
            if hn.max() > 0:
                px, py = _step_poly(hn / hn.max() * 0.8, edges, base)
                nx += px.tolist() + [None]
                ny += py.tolist() + [None]
        fig.add_trace(go.Scatter(
            x=nx, y=ny, mode="lines", fill="toself", name="non-target",
            line=dict(color=rgba(t["neutral"], 0.9), width=1),
            fillcolor=rgba(t["neutral"], 0.28), hoverinfo="skip"))
        for k, lbl in enumerate(an.labels):
            pos = an.true_idx == k
            col = cmap.get(lbl, t["neutral"])
            ht, _ = np.histogram(P[pos, k], bins=edges)
            px, py = (_step_poly(ht / ht.max() * 0.8, edges, k + 0.42)
                      if ht.max() > 0 else ([], []))
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="lines", fill="toself",
                name=f"{lbl} target", legendgroup=lbl, showlegend=False,
                line=dict(color=col, width=1.4), fillcolor=rgba(col, 0.42),
                hoverinfo="skip"))

    tx, ty = [], []
    for k in range(K):
        tx += [tx_k[k], tx_k[k], None]
        ty += [k - 0.46, k + 0.46, None]
    fig.add_trace(go.Scatter(
        x=tx, y=ty, mode="lines", name="threshold", hoverinfo="skip",
        line=dict(color=t["gold"], width=2, dash="dot"), meta=dyn(thr=XY)))
    fig.add_trace(go.Scatter(
        x=[1.035] * K, y=list(range(K)), mode="text",
        text=_rates(an, Pn, tv, real(tx_k) if raw else tv),
        textposition="middle right",
        textfont=dict(size=10, color=t["muted"]),
        hoverinfo="skip", showlegend=False, name="rates",
        meta=dyn(thr=["text"])))

    sel = np.asarray(sorted(sel_rows or []), dtype=int)
    if sel.size > RUG_CAP:
        sel = np.random.default_rng(42).choice(sel, RUG_CAP, replace=False)
    lane = np.repeat(np.arange(K), sel.size)
    fig.add_trace(go.Scatter(
        x=P[sel, :].T.reshape(-1) if sel.size else [],
        y=lane + 0.42 if sel.size else [],
        mode="markers", name="selection", showlegend=bool(sel.size),
        marker=dict(symbol="line-ns", size=9,
                    line=dict(width=1.3, color=t["gold"])),
        hovertemplate="selected · %{x:.3f}<extra></extra>",
        meta=dyn(sel=XY + ["showlegend"])))

    # The inspected vector(s): where each sits in EVERY class's lane.
    fr = list(focus_rows or [])[:2]
    fig.add_trace(go.Scatter(
        x=[P[r, k] for r in fr for k in range(K)],
        y=[k for _ in fr for k in range(K)],
        mode="markers", name="inspected", showlegend=False,
        marker=dict(symbol=[s for s in ("diamond", "diamond-open")[:len(fr)]
                            for _ in range(K)],
                    size=11, color=t["ring"],
                    line=dict(width=1.6, color=t["ring"])),
        hovertemplate="inspected · %{x:.3f}<extra></extra>",
        meta=dyn(foc=XY + ["marker.symbol"])))

    fig.update_layout(**base_layout(
        theme,
        xaxis=dict(title=dict(
            text=f"raw {an.spec.methodology} score" if raw else
            f"normalized score   (α = "
            f"{float(alpha if alpha is not None else 1):.2f})"),
                   range=[-0.02, 1.33], zeroline=False, minallowed=-0.1,
                   maxallowed=1.45,
                   tickmode="array", tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   **({"ticktext": [f"{v:.3g}" for v in
                                    real([0, 0.2, 0.4, 0.6, 0.8, 1.0])]}
                      if raw else {})),
        yaxis=dict(tickmode="array", tickvals=list(range(K)),
                   ticktext=an.labels, range=[K - 0.5, -0.5],
                   minallowed=-0.6, maxallowed=K - 0.4,
                   showgrid=False, zeroline=False,
                   tickfont=dict(size=10, color=t["text"])),
        shapes=[dict(type="line", xref="paper", yref="y", x0=0, x1=1,
                     y0=k + 0.5, y1=k + 0.5,
                     line=dict(color=t["grid"], width=1))
                for k in range(K - 1)]
        + [dict(type="line", xref="x", yref="paper", x0=1.0, x1=1.0, y0=0,
                y1=1, line=dict(color=t["axis"], width=1))],
        meta={"home": {"xaxis.range": [-0.02, 1.33],
                       "yaxis.range": [K - 0.5, -0.5]}},
        showlegend=view == "strip",
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom",
                    xanchor="left"),
        margin=dict(l=8, r=10, t=26 if view == "strip" else 6, b=38),
        uirevision=f"dist|{an.frame.name}|{view}|{domain}|{K}"))
    return fig
