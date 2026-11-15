"""Projection scatter: classes, centroids, accepted envelopes, focus.

Threshold feedback is per POINT (works in PCA, t-SNE and UMAP alike):
    ● filled   accepted by its own class and wins
    ○ hollow   rejected by its own class at the current threshold
    ✕ cross    accepted, but another class scores closer
    ◇ diamond  excluded from the fit (skip list)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..engine.analysis import Analysis, Decision
from ..engine.project import Projection, accepted_hulls, centroid_xy
from ..theme import BLUES, base_layout, class_colors, rgba, tokens
from .common import XY, dyn, home

GL_ABOVE = 2500
# A box drag makes Plotly mark points "selected" in EVERY marker trace;
# on the non-vector traces a selection update resets that (-> None).
SP = ["selectedpoints"]
OWN_SCORE = "own-class score"


def is_continuous(col: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(col)
            and not pd.api.types.is_bool_dtype(col) and col.nunique() > 12)
AXIS_NAME = {"pca": "PC", "tsne": "t-SNE ", "umap": "UMAP "}


def _symbols(an: Analysis, d: Decision, rows: np.ndarray) -> np.ndarray:
    sym = np.full(rows.size, "circle", dtype=object)
    wrong = d.own_ok[rows] & ~d.correct[rows]
    sym[wrong] = "x"
    sym[~d.own_ok[rows]] = "circle-open"
    sym[~an.fit_mask[rows]] = "diamond-open"
    return sym


def scatter_fig(an: Analysis, proj: Projection, d: Decision, *,
                theme: str | None, alpha: float | None,
                sel_rows: list[int] | None = None, sel_rev: str = "",
                focus_rows: list[int] | None = None,
                focus_labels: list[str] | None = None,
                neighbour_rows: list[int] | None = None,
                show_hull: bool = True, show_density: bool = False,
                color_by: str | None = None) -> go.Figure:
    t = tokens(theme)
    rows, xy = proj.rows, proj.xy
    y = an.y[rows]
    cmap = class_colors(sorted(set(an.y.tolist())))
    # Points are grouped (trace + legend + colour) by class, or by any other
    # column; markers, envelopes and centroids always speak about CLASSES.
    # A numeric column (or OWN_SCORE) colours on a continuous scale instead,
    # keeping the per-class traces.
    col = an.frame.df[color_by] if color_by in an.frame.df.columns else None
    cont = None
    if color_by == OWN_SCORE:
        cont = np.where(an.true_idx[rows] >= 0,
                        an.norm(alpha)[rows, np.maximum(an.true_idx[rows], 0)],
                        np.nan)
    elif col is not None and is_continuous(col):
        cont = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)[rows]
    by_class = cont is not None or col is None
    g_all = an.y if by_class else col.astype(str).to_numpy()
    g = g_all[rows]
    classes = sorted(set(g.tolist()))
    gmap = cmap if by_class else class_colors(sorted(set(g_all.tolist())))
    Scatter = go.Scattergl if rows.size > GL_ABOVE else go.Scatter
    size = 7 if rows.size <= 800 else (5 if rows.size <= GL_ABOVE else 4)
    P = an.norm(alpha)
    sel = set(sel_rows or [])
    sym_all = _symbols(an, d, rows)
    pred_lbl = np.array([an.labels[p] if p >= 0 else "— rejected"
                         for p in d.pred[rows]], dtype=object)
    own = np.where(an.true_idx[rows] >= 0,
                   P[rows, np.maximum(an.true_idx[rows], 0)], np.nan)
    fig = go.Figure()

    # Always present (hidden when off): a trace appearing / disappearing
    # makes Plotly forget the user's zoom.
    fig.add_trace(go.Histogram2dContour(
        x=xy[:, 0], y=xy[:, 1], ncontours=12, showscale=False,
        contours=dict(coloring="lines"), hoverinfo="skip",
        colorscale=[[0, rgba(t["gold"], 0.0)], [1, rgba(t["gold"], 0.65)]],
        line=dict(width=1), name="density", showlegend=False,
        visible=bool(show_density and rows.size >= 5)))

    def colour(lbl: str) -> str:
        return gmap[lbl] if (not by_class or lbl in an.labels) else t["neutral"]

    for lbl in classes:
        m = g == lbl
        color = colour(lbl)
        r = rows[m]
        selpts = (list(range(r.size)) if not sel
                  else [i for i, v in enumerate(r) if int(v) in sel])
        fig.add_trace(Scatter(
            x=xy[m, 0], y=xy[m, 1], mode="markers",
            name=lbl, legendgroup=lbl, showlegend=False,
            marker=dict(size=size, symbol=sym_all[m], opacity=0.9,
                        line=dict(width=1.2, color=color),
                        **(dict(color=color) if cont is None else
                           dict(color=cont[m], coloraxis="coloraxis"))),
            selectedpoints=selpts,
            selected=dict(marker=dict(opacity=0.95)),
            unselected=dict(marker=dict(opacity=0.12)),
            customdata=np.column_stack([
                r, np.asarray(an.frame.sids, dtype=object)[r],
                pred_lbl[m], np.round(own[m], 3)]),
            text=y[m],
            hovertemplate=("<b>%{text}</b>"
                           + ("" if by_class else f" · {color_by} {lbl}")
                           + "<br>%{customdata[1]}<br>"
                           "predicted: %{customdata[2]}<br>"
                           "own-class score: %{customdata[3]}<extra></extra>"),
            meta=dyn(thr=["marker.symbol", "customdata"],
                     sel=["selectedpoints"]),
        ))

    # Legend keys are separate point-less traces: a data trace's key would
    # copy its FIRST point's symbol (○ / ✕), which reads as a class property.
    for lbl in classes:
        skipped = by_class and lbl not in an.labels
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", legendgroup=lbl,
            name=f"{lbl}  ({int((g_all == lbl).sum()):,})"
                 + (" · skipped" if skipped else ""),
            marker=dict(size=9, symbol="circle", color=colour(lbl))))

    hulls = accepted_hulls(an, proj, d.own_ok) if show_hull else {}
    fill_alpha = 0.06 if an.k <= 8 else 0.0     # many classes: outline only
    for lbl in an.labels:           # one trace per class even when empty —
        hx, hy = hulls.get(lbl, ([], []))   # keeps trace indices stable
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode="lines", fill="toself",
            fillcolor=rgba(cmap.get(lbl, t["neutral"]), fill_alpha),
            line=dict(color=rgba(cmap.get(lbl, t["neutral"]), 0.75),
                      width=1.2, dash="dash"),
            hoverinfo="skip", showlegend=False,
            legendgroup=lbl if by_class else None,
            name=f"{lbl} accepted envelope", meta=dyn(thr=XY)))

    cents = centroid_xy(an, proj)
    emph = set(focus_labels or [])
    fig.add_trace(go.Scatter(
        x=[c[2] for c in cents], y=[c[3] for c in cents],
        mode="markers",
        marker=dict(symbol="cross", size=[22 if c[0] in emph else 15
                                           for c in cents],
                    color=[cmap.get(c[0], t["neutral"]) for c in cents],
                    line=dict(width=[3 if c[0] in emph else 1.6 for c in cents],
                              color=[t["gold"] if c[0] in emph else t["text"]
                                     for c in cents])),
        customdata=[[c[0], c[1] + 1] for c in cents],
        hovertemplate="<b>%{customdata[0]}</b> centroid "
                      "#%{customdata[1]}<extra></extra>",
        name=f"{an.spec.methodology} centroids", legendrank=2000,
        meta=dyn(foc=["marker"], sel=SP)))

    pos = proj.pos
    fpts = [pos[r] for r in (focus_rows or []) if r in pos]
    fig.add_trace(go.Scatter(
        x=xy[fpts, 0] if fpts else [], y=xy[fpts, 1] if fpts else [],
        mode="markers", hoverinfo="skip", showlegend=False, name="focus",
        marker=dict(symbol="circle-open", size=20,
                    line=dict(width=2.5, color=t["gold"])),
        meta=dyn(foc=XY, sel=SP)))
    npts = [pos[r] for r in (neighbour_rows or []) if r in pos]
    fig.add_trace(go.Scatter(
        x=xy[npts, 0] if npts else [], y=xy[npts, 1] if npts else [],
        mode="markers", hoverinfo="skip", showlegend=False, name="neighbours",
        marker=dict(symbol="circle-open", size=13,
                    line=dict(width=1.3, color=t["gold"])),
        meta=dyn(foc=XY, sel=SP)))

    ax = AXIS_NAME.get(proj.method, "dim ")
    ex = proj.explained
    xt = f"{ax}1" + (f"  ·  {ex[0]:.1%} var" if ex else "")
    yt = f"{ax}2" + (f"  ·  {ex[1]:.1%} var" if ex else "")
    # Fixed home view = the points' extent. With autorange the axes rescale
    # whenever a trace's extent changes and the picture jumps under the
    # user's hand. The user's own zoom is kept across updates by
    # `uirevision`; zooming OUT is limited to half a span beyond the data.
    def span(v: np.ndarray) -> list[float]:
        lo, hi = float(np.min(v)), float(np.max(v))
        pad = 0.06 * (hi - lo or 1.0)
        return [lo - pad, hi + pad]

    view = home(span(xy[:, 0]), span(xy[:, 1]))
    fig.update_layout(**base_layout(
        theme,
        xaxis=dict(title=dict(text=xt), zeroline=False, **view["xaxis"]),
        yaxis=dict(title=dict(text=yt), zeroline=False, **view["yaxis"]),
        meta=view["meta"],
        legend=dict(x=1.01 if cont is None else 1.07, y=1, xanchor="left",
                    yanchor="top"),
        **({} if cont is None else {"coloraxis": dict(
            colorscale=BLUES, colorbar=dict(
                thickness=8, len=0.6, x=1.0, xanchor="left", outlinewidth=0,
                tickfont=dict(size=9, color=t["muted"]),
                title=dict(text=color_by, side="right",
                           font=dict(size=10, color=t["muted"]))))}),
        margin=dict(l=48, r=8, t=8, b=38),
        dragmode="select", clickmode="event",
        # The STORE owns the selection. Plotly's persistent lasso outline is
        # a second source of truth (it re-derives the lit points from its
        # box on every draw, centroid markers included), so it is dropped
        # as soon as the gesture is captured; `selectedpoints` say the rest.
        # Plotly also KEEPS the user's own lit-points state across updates
        # while `selectionrevision` is unchanged — so every store-driven
        # selection update must carry a new revision, or it is ignored.
        selections=[], selectionrevision=sel_rev,
        uirevision=f"scatter|{an.frame.name}|{proj.method}|{rows.size}",
    ))
    return fig
