"""Matrix panel: pair scores (vectors), centroid scores (classes),
confusion (true × predicted at the current thresholds)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..engine import matrix as mx
from ..engine.analysis import Analysis, Decision
from ..theme import BLUES, base_layout, class_colors, tokens
from .common import message_fig

SEP = "  ·  "          # "<session_id>  ·  <class>" — parsed back on click
MAX_RUN_TICKS = 40
MAX_RUN_STRIPS = 120


def _zrange(Z: np.ndarray, auto: bool, lo: float, hi: float) -> tuple[float, float]:
    if not auto or Z.size == 0:
        return (min(lo, hi), max(lo, hi))
    a, b = float(np.nanmin(Z)), float(np.nanmax(Z))
    return (a, b) if b > a else (a - 0.5, a + 0.5)


def _square(theme, **over) -> dict:
    xa = {"showgrid": False, "zeroline": False, "constrain": "domain",
          "side": "bottom", **over.pop("xaxis", {})}
    ya = {"showgrid": False, "zeroline": False, "autorange": "reversed",
          "scaleanchor": "x", "constrain": "domain", **over.pop("yaxis", {})}
    return base_layout(theme, xaxis=xa, yaxis=ya,
                       **{"margin": dict(l=8, r=8, t=8, b=8), **over})


def _colorbar(t: dict, title: str) -> dict:
    return dict(thickness=9, len=0.8, outlinewidth=0,
                title=dict(text=title, side="right",
                           font=dict(size=10, color=t["muted"])),
                tickfont=dict(size=9, color=t["muted"]))


def vectors_fig(an: Analysis, rows: np.ndarray, *, theme, abs_on, power,
                auto_range, lo, hi,
                focus_rows: list[int] | None = None,
                strip_by: str | None = None) -> go.Figure:
    t = tokens(theme)
    n = rows.size
    if n == 0:
        return message_fig("empty selection", theme)
    Z = mx.transform(mx.pair_matrix(an, rows), abs_on, power)
    zmin, zmax = _zrange(Z, auto_range, lo, hi)
    cls = an.y[rows].astype(str)
    names = [f"{an.frame.sids[r]}{SEP}{c}" for r, c in zip(rows, cls)]
    fig = go.Figure(go.Heatmap(
        z=Z.astype(np.float32), x=names, y=names, colorscale=BLUES,
        zmin=zmin, zmax=zmax, colorbar=_colorbar(t, "score_XY"),
        hovertemplate="row  %{y}<br>col  %{x}<br>score  <b>%{z:.4f}</b>"
                      "<extra></extra>"))
    # Blocks (colour strips, boundaries, tick labels) follow the classes, or
    # the column the rows are sorted by — e.g. the user's own labels.
    if strip_by and strip_by in an.frame.df.columns:
        full = an.frame.df[strip_by].astype(str).to_numpy()
        grp, cmap = full[rows], class_colors(sorted(set(full.tolist())))
    else:
        grp, cmap = cls, class_colors(sorted(set(an.y.tolist())))
    rr = mx.runs(grp)
    shapes = []
    pad = max(n * 0.035, 0.6)      # gutter (data units) for the class strips
    a, b = -0.5 - 0.85 * pad, -0.5 - 0.25 * pad
    if len(rr) <= MAX_RUN_STRIPS:
        for s, e, lbl in rr:       # class colour strips, left + top
            col = cmap.get(lbl, t["neutral"])
            shapes.append(dict(type="rect", xref="x", yref="y", x0=a, x1=b,
                               y0=s - 0.5, y1=e - 0.5, fillcolor=col,
                               line=dict(width=0)))
            shapes.append(dict(type="rect", xref="x", yref="y", y0=a, y1=b,
                               x0=s - 0.5, x1=e - 0.5, fillcolor=col,
                               line=dict(width=0)))
    ticks = len(rr) <= MAX_RUN_TICKS
    if ticks:
        for s, _e, _ in rr[1:]:    # block boundaries
            for kw in (dict(x0=s - 0.5, x1=s - 0.5, y0=-0.5, y1=n - 0.5),
                       dict(y0=s - 0.5, y1=s - 0.5, x0=-0.5, x1=n - 0.5)):
                shapes.append(dict(type="line", xref="x", yref="y",
                                   line=dict(color="rgba(255,255,255,0.45)",
                                             width=0.7), **kw))
    where = {int(r): i for i, r in enumerate(rows)}
    for r in (focus_rows or []):          # crosshair on the inspected vector
        i = where.get(int(r))
        if i is None:
            continue
        for kw in (dict(x0=-0.5, x1=n - 0.5, y0=i, y1=i),
                   dict(y0=-0.5, y1=n - 0.5, x0=i, x1=i)):
            shapes.append(dict(type="line", xref="x", yref="y",
                               line=dict(color=t["gold"], width=1), **kw))
    big = [r for r in rr if r[1] - r[0] >= max(1, 0.035 * n)] if ticks else []
    centers = [names[(s + e - 1) // 2] for s, e, _ in big]
    axis = dict(showticklabels=bool(big), tickmode="array", tickvals=centers,
                ticktext=[r[2] for r in big],
                tickfont=dict(size=9, color=t["muted"]), ticks="")
    fig.update_layout(**_square(
        theme,
        xaxis=dict(axis, tickangle=-35, range=[-0.5 - pad, n - 0.5],
                   minallowed=-0.5 - 2 * pad, maxallowed=n - 0.5 + pad,
                   showticklabels=ticks and len(rr) <= 12),
        yaxis=dict(axis, autorange=False, range=[n - 0.5, -0.5 - pad],
                   minallowed=-0.5 - 2 * pad, maxallowed=n - 0.5 + pad),
        meta={"home": {"xaxis.range": [-0.5 - pad, n - 0.5],
                       "yaxis.range": [n - 0.5, -0.5 - pad]}},
        shapes=shapes,
        margin=dict(l=72, r=4, t=6, b=54 if len(rr) <= 12 else 8),
        uirevision=f"mx-v|{an.frame.name}|{n}"))
    return fig


def classes_fig(an: Analysis, *, theme, abs_on, power, auto_range,
                lo, hi) -> go.Figure:
    t = tokens(theme)
    M, names = mx.centroid_matrix(an)
    if not names:
        return message_fig("this methodology has no centroids", theme)
    Z = mx.transform(M, abs_on, power)
    zmin, zmax = _zrange(Z, auto_range, lo, hi)
    small = len(names) <= 12
    fig = go.Figure(go.Heatmap(
        z=Z, x=names, y=names, colorscale=BLUES, zmin=zmin, zmax=zmax,
        colorbar=_colorbar(t, "score_XY"),
        texttemplate="%{z:.2f}" if small else None,
        textfont=dict(size=10),
        hovertemplate="%{y}  ×  %{x}<br>score  <b>%{z:.4f}</b><extra></extra>"))
    axis = dict(tickfont=dict(size=9 if small else 8, color=t["muted"]),
                ticks="", showticklabels=len(names) <= 40)
    fig.update_layout(**_square(theme, xaxis=dict(axis, tickangle=-40),
                                yaxis=dict(axis),
                                uirevision=f"mx-c|{an.frame.name}"))
    return fig


TREE_SEP = "\x1f"       # joins the class names under a tree node


def tree_fig(an: Analysis, *, theme) -> go.Figure:
    """Class dendrogram. Click a junction → select every class under it."""
    t = tokens(theme)
    tr = mx.class_tree(an)
    if tr is None:
        return message_fig("need ≥ 2 classes with prototypes", theme)
    cmap = class_colors(sorted(set(an.y.tolist())))
    k, y, h = tr["k"], tr["y"], tr["h"]
    top = max(h.values()) or 1.0        # leaves sit at 0: pad so they are
                                        # whole and clickable
    lx, ly = [], []
    for node, (a, b) in tr["children"].items():     # one ⊐ shape per merge
        lx += [h[a], h[node], h[node], h[b], None]
        ly += [y[a], y[a], y[b], y[b], None]
    fig = go.Figure(go.Scatter(x=lx, y=ly, mode="lines", hoverinfo="skip",
                               line=dict(color=t["muted"], width=1.4),
                               showlegend=False))
    nodes = sorted(tr["children"])
    fig.add_trace(go.Scatter(
        x=[h[n] for n in nodes], y=[y[n] for n in nodes], mode="markers",
        marker=dict(size=9, color=t["panel"],
                    line=dict(width=1.6, color=t["accent"])),
        customdata=[[TREE_SEP.join(tr["members"][n]), len(tr["members"][n])]
                    for n in nodes],
        hovertemplate="%{customdata[1]} classes · cosine distance %{x:.3f}"
                      "<br>click to select them<extra></extra>",
        showlegend=False))
    fig.add_trace(go.Scatter(
        x=[0.0] * k, y=list(range(k)), mode="markers",
        marker=dict(size=10, color=[cmap.get(lb, t["neutral"])
                                    for lb in tr["labels"]]),
        customdata=[[lb, 1] for lb in tr["labels"]],
        hovertemplate="%{customdata[0]}<br>click to select<extra></extra>",
        showlegend=False))
    fig.update_layout(**base_layout(
        theme,
        xaxis=dict(title=dict(text="cosine distance between class prototypes"),
                   range=[-0.04 * top, 1.05 * top], zeroline=False),
        yaxis=dict(tickmode="array", tickvals=list(range(k)),
                   ticktext=tr["labels"], range=[k - 0.5, -0.5],
                   showgrid=False, zeroline=False,
                   tickfont=dict(size=10 if k <= 30 else 8, color=t["text"])),
        margin=dict(l=8, r=12, t=8, b=38),
        uirevision=f"mx-t|{an.frame.name}|{k}"))
    return fig


def confusion_fig(an: Analysis, d: Decision, sel_rows, *, theme) -> go.Figure:
    t = tokens(theme)
    counts, rlab, clab = mx.confusion(an, d, sel_rows)
    tot = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    rate = counts / tot                      # normalised by TRUE class count
    small = len(rlab) <= 14
    fig = go.Figure(go.Heatmap(
        z=rate, x=clab, y=rlab, colorscale=BLUES, zmin=0, zmax=1,
        colorbar=_colorbar(t, "share of true class"),
        text=counts, texttemplate="%{text}" if small else None,
        textfont=dict(size=10),
        hovertemplate="true  %{y}<br>predicted  %{x}<br>"
                      "<b>%{text}</b> samples · %{z:.1%}<extra></extra>"))
    axis = dict(tickfont=dict(size=9 if small else 8, color=t["muted"]),
                ticks="", showticklabels=len(rlab) <= 40)
    k = len(rlab)
    fig.update_layout(**base_layout(
        theme,
        xaxis=dict(axis, showgrid=False, tickangle=-40,
                   title=dict(text="predicted")),
        yaxis=dict(axis, showgrid=False, autorange="reversed",
                   title=dict(text="true")),
        shapes=[dict(type="line", xref="x", yref="y", x0=k - 0.5, x1=k - 0.5,
                     y0=-0.5, y1=k - 0.5,
                     line=dict(color=t["gold"], width=1, dash="dot"))],
        margin=dict(l=8, r=8, t=8, b=8),
        uirevision=f"mx-x|{an.frame.name}|{k}"))
    return fig
