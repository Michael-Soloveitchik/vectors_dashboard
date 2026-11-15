"""Threshold callbacks — one writer, versioned by the fit token.

    global slider / EER / reset / ◀▶ ─► per-class slider VALUES ─► store_thr
                                                    {"token": fit token, "values"}

``store_thr`` is re-stamped after EVERY fit (even when the class list
did not change), and only when the slider rows match the fit's classes.
Panels draw only when the token matches — so no panel is ever painted
with the previous fit's thresholds, and each fit costs each panel one
render, not two.

Slider rows are re-created only when the class list changes; a drag is
never interrupted by its own re-render.
"""
from __future__ import annotations

import numpy as np
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from ..debug_log import trace
from ..engine import Engine, state
from ..engine import curves as cv
from ..ids import IDS
from ..theme import class_colors

SLIDER = {"type": "thr-slider", "label": ALL}
# Finest step the slider can take: below 1e-6 JavaScript prints the number
# in exponent form ("1e-7"), the slider then counts 0 decimals and snaps
# every drag to 0 or 1 — "all or nothing".
STEP = 0.000001


def _row(lbl: str, value: float, color: str) -> html.Div:
    return html.Div(className="thr-row", children=[
        html.Span([html.I(style={"background": color}), lbl],
                  className="thr-name", title=lbl),
        html.Button("◀", id={"type": "thr-step", "label": lbl, "dir": "L"},
                    n_clicks=0, className="step",
                    title="previous score boundary (one sample out)"),
        html.Div(dcc.Slider(
            id={"type": "thr-slider", "label": lbl}, min=0.0, max=1.0,
            step=STEP, value=value, marks=None, updatemode="drag",
            className="slider", allow_direct_input=False,
            tooltip={"placement": "bottom", "always_visible": False}),
            className="thr-slider"),
        html.Button("▶", id={"type": "thr-step", "label": lbl, "dir": "R"},
                    n_clicks=0, className="step",
                    title="next score boundary (one sample in)"),
        html.Span(f"{value:.3f}", id={"type": "thr-val", "label": lbl},
                  className="thr-val"),
    ])


def _clip(v) -> float:
    try:
        return float(min(1.0, max(0.0, float(v))))
    except (TypeError, ValueError):
        return 0.0


def register(app, engine: Engine) -> None:

    @app.callback(
        Output(IDS.threshold_panel, "children"),
        Input(IDS.store_fit, "data"), Input(IDS.store_kick, "data"),
        State(IDS.store_thr, "data"), State(IDS.ctrl_thr_all, "value"),
        State(SLIDER, "id"))
    @trace
    def _panel(fit, _kick, thr, thr_all, cur_ids):
        if not fit or fit.get("error"):
            return []
        labels = fit["labels"]
        if [i["label"] for i in (cur_ids or [])] == labels:
            return no_update
        an = engine.analysis(fit["spec"])
        cmap = class_colors(sorted(set(an.y.tolist())))
        old = state.thr_values(thr)        # same-named classes keep theirs
        return [_row(lbl, _clip(old.get(lbl, thr_all)), cmap.get(lbl, "#888"))
                for lbl in labels]

    @app.callback(
        Output(IDS.store_thr, "data"),
        Output({"type": "thr-val", "label": ALL}, "children"),
        Input(SLIDER, "value"), Input(IDS.store_fit, "data"),
        Input(IDS.store_kick, "data"),
        State(SLIDER, "id"), State(IDS.store_thr, "data"))
    def _to_store(values, fit, _kick, ids_, prev):
        if not fit:
            raise PreventUpdate
        if ctx.triggered_id == IDS.store_kick and state.thr_synced(fit, prev):
            raise PreventUpdate            # reconciler ping, already in step
        labels = [] if fit.get("error") else list(fit.get("labels") or [])
        have = [i["label"] for i in (ids_ or [])]
        if have != labels:
            raise PreventUpdate            # rows for this fit are on their way
        vals = [_clip(v) for v in (values or [])]
        return ({"token": fit.get("token"), "values": dict(zip(have, vals))},
                [f"{v:.3f}" for v in vals])

    @app.callback(
        Output(SLIDER, "value"), Output(IDS.ctrl_thr_all, "value"),
        Input(IDS.ctrl_thr_all, "value"), Input(IDS.btn_thr_eer, "n_clicks"),
        Input(IDS.btn_thr_dcf, "n_clicks"), Input(IDS.btn_thr_reset, "n_clicks"),
        Input({"type": "thr-step", "label": ALL, "dir": ALL}, "n_clicks"),
        Input(IDS.thr_pick, "data"),
        State(SLIDER, "id"), State(SLIDER, "value"),
        State(IDS.store_fit, "data"), State(IDS.ctrl_alpha, "value"),
        prevent_initial_call=True)
    @trace
    def _drive(thr_all, _eer, _dcf, _reset, _steps, pick, ids_, cur, fit,
               alpha):
        trig = ctx.triggered_id
        labels = [i["label"] for i in (ids_ or [])]
        if not labels or not ctx.triggered:
            raise PreventUpdate
        if trig == IDS.ctrl_thr_all:
            return [_clip(thr_all)] * len(labels), no_update
        if trig == IDS.thr_pick:               # click on a ROC curve / lane
            if not pick or pick.get("label") not in labels:
                raise PreventUpdate
            out = [_clip(v) for v in cur]
            out[labels.index(pick["label"])] = round(
                float(np.floor(_clip(pick.get("thr")) / STEP) * STEP), 6)
            return out, no_update
        if not ctx.triggered[0]["value"]:      # buttons just (re)created
            raise PreventUpdate
        if trig == IDS.btn_thr_reset:
            return [0.0] * len(labels), 0.0
        if not fit or fit.get("error") or fit.get("labels") != labels:
            raise PreventUpdate
        an = engine.analysis(fit["spec"])
        cur = [_clip(v) for v in cur]
        if trig in (IDS.btn_thr_eer, IDS.btn_thr_dcf):
            best = (cv.eer_thresholds if trig == IDS.btn_thr_eer
                    else cv.dcf_thresholds)(an, alpha)
            return ([round(best.get(lbl, c), 4) for lbl, c in zip(labels, cur)],
                    no_update)
        # ◀ / ▶ : jump to the neighbouring distinct score of that class —
        # exactly one sample changes side per click.
        lbl, direction = trig["label"], trig["dir"]
        if lbl not in an.labels:
            raise PreventUpdate
        uniq = np.unique(an.norm(alpha)[:, an.labels.index(lbl)])
        i = labels.index(lbl)
        nxt = uniq[uniq > cur[i]] if direction == "R" else uniq[uniq < cur[i]]
        if nxt.size:                 # floor to the slider grid: never above
            target = nxt[0] if direction == "R" else nxt[-1]   # the sample
            snapped = float(np.floor(target / STEP) * STEP)
            if direction == "R":     # samples closer than one step: still move
                snapped = max(snapped, cur[i] + STEP)
            cur[i] = round(min(1.0, snapped), 6)
        elif direction == "L":
            cur[i] = 0.0
        return cur, no_update
