"""ROC / DET panel — per-class one-vs-rest, pair curves, selection overlay,
operating points at the current thresholds."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..engine import curves as cv
from ..engine.analysis import Analysis
from ..theme import base_layout, class_colors, tokens
from .common import XY, dyn

PAIR_STYLE = {"pair-cos": ("#ff9f43", "solid"), "pair-norm": ("#4ecdc4", "dot")}


def _xy(fpr, tpr, det: bool):
    return cv.to_det(np.asarray(fpr, float), np.asarray(tpr, float)) if det \
        else (np.asarray(fpr, float), np.asarray(tpr, float))


def roc_fig(an: Analysis, *, alpha, thr: dict | None, theme,
            sel_rows: list[int] | None = None, det: bool = False,
            show: str = "all", log_x: bool = False, log_y: bool = False,
            focus_labels: list[str] | None = None) -> go.Figure:
    t = tokens(theme)
    cmap = class_colors(sorted(set(an.y.tolist())))
    emph = set(focus_labels or [])
    fig = go.Figure()
    ht = ("thr %{customdata[1]:.3f}<br>FPR %{x:.4f}<br>TPR %{y:.4f}" if not det
          else "thr %{customdata[1]:.3f}")

    def add_curve(c: cv.Curve, color: str, dash: str, width: float, rank: int):
        f, tp, th = cv.decimate(c)
        x, y = _xy(f, tp, det)
        tip = "<br><i>click: set this threshold</i>" if c.kind == "class" else ""
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            customdata=[[c.key, float(v)] for v in th],   # click → threshold
            name=f"{c.name}   AUC {c.auc:.3f} · EER {c.eer:.1%}",
            legendgroup=c.key, legendrank=rank,
            line=dict(color=color, width=width, dash=dash),
            hovertemplate=(f"<b>{c.name}</b>  ·  minDCF {c.dcf:.3f}<br>{ht}"
                           f"{tip}<extra></extra>")))

    if show in ("all", "pair"):
        for c in cv.pair_curves(an, alpha):
            col, dash = PAIR_STYLE[c.key]
            add_curve(c, col, dash, 2.2, 10)
    classes = cv.class_curves(an, alpha) if show in ("all", "class") else []
    for c in classes:
        add_curve(c, cmap.get(c.key, t["neutral"]), "solid",
                  3.2 if c.key in emph else 1.6, 100)

    ch = np.linspace(1e-5, 1, 60)
    cx, cy = _xy(ch, ch, det)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", hoverinfo="skip",
                             showlegend=False, name="chance",
                             line=dict(color=t["axis"], width=1, dash="dash")))

    ops = cv.operating_points(an, thr, alpha) if classes else {}
    keys = [c.key for c in classes if c.key in ops]
    ox, oy = _xy([ops[k][0] for k in keys], [ops[k][1] for k in keys], det)
    fig.add_trace(go.Scatter(
        x=ox, y=oy, mode="markers", showlegend=False, name="operating points",
        marker=dict(size=[15 if k in emph else 11 for k in keys],
                    color=[cmap.get(k, t["neutral"]) for k in keys],
                    line=dict(width=[3 if k in emph else 1.8 for k in keys],
                              color=[t["gold"] if k in emph else t["text"]
                                     for k in keys])),
        customdata=[[k, ops[k][2], ops[k][0], ops[k][1]] for k in keys],
        hovertemplate=("<b>%{customdata[0]}</b> operating point<br>"
                       "thr %{customdata[1]:.3f}<br>FPR %{customdata[2]:.4f}"
                       "<br>TPR %{customdata[3]:.4f}<extra></extra>"),
        meta=dyn(thr=XY + ["customdata"], foc=["marker"])))

    sc = cv.selection_curve(an, alpha, sel_rows)
    if sc is not None:
        add_curve(sc, t["gold"], "solid", 2.6, 5)
        sel = np.zeros(an.frame.n, dtype=bool)
        sel[np.asarray(sel_rows, dtype=int)] = True
        acc = (an.norm(alpha) >= an.thr_vector(thr)[None, :]).any(axis=1)
        sx, sy = _xy([float((acc & ~sel).sum() / max((~sel).sum(), 1))],
                     [float((acc & sel).sum() / max(sel.sum(), 1))], det)
        fig.add_trace(go.Scatter(
            x=sx, y=sy, mode="markers", showlegend=False,
            name="selection operating point",
            marker=dict(size=13, symbol="diamond", color=t["gold"],
                        line=dict(width=1.5, color=t["bg"])),
            hovertemplate="<b>selection</b> — accepted by any class"
                          "<extra></extra>", meta=dyn(thr=XY)))

    if det:
        from scipy.stats import norm
        tv = norm.ppf(cv.DET_TICKS).tolist()
        rng = [tv[0] - 0.25, tv[-1] + 0.25]
        lim = dict(minallowed=rng[0] - 0.6, maxallowed=rng[1] + 0.6)
        axis = dict(tickmode="array", tickvals=tv, ticktext=cv.DET_LABELS,
                    range=rng, type="linear", **lim)
        xa = dict(axis, title=dict(text="false positive rate (probit)"))
        ya = dict(axis, title=dict(text="false negative rate (probit)"))
    else:
        lin = dict(type="linear", range=[-0.02, 1.02], minallowed=-0.25,
                   maxallowed=1.25)

        def log(lo: int) -> dict:      # decades as percentages, not "100µ"
            vals = [10.0 ** e for e in range(lo, 1)]
            return dict(type="log", range=[lo, 0.02], tickmode="array",
                        tickvals=vals, ticktext=[f"{v:.4g}" for v in vals])
        xa = dict(title=dict(text="false positive rate"),
                  **(log(-4) if log_x else lin))
        ya = dict(title=dict(text="true positive rate"),
                  **(log(-2) if log_y else lin))
    fig.update_layout(**base_layout(
        theme, xaxis=xa, yaxis=ya,
        meta={"home": {"xaxis.range": list(xa["range"]),
                       "yaxis.range": list(ya["range"])}},
        legend=dict(x=1.01, y=1, xanchor="left", yanchor="top"),
        margin=dict(l=48, r=8, t=8, b=38),
        uirevision=f"roc|{an.frame.name}|{det}|{log_x}|{log_y}"))
    return fig
