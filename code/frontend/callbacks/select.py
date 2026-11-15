"""Focus (clicks) and the inspector. Selection lives in ``selection.py``.

``store_focus`` (what the inspector shows) survives a refit — the drawer
re-renders against the new methodology — and is dropped on model change.
"""
from __future__ import annotations

import time

from dash import ALL, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from .. import inspector as insp
from ..debug_log import trace
from ..engine import Engine, state
from ..figures import SEP, TREE_SEP
from ..ids import IDS
from .panels import panel
from .selection import class_intent, new_intent


def _points(data: dict | None) -> list[dict]:
    return (data or {}).get("points") or []


def register(app, engine: Engine) -> None:
    hub = engine.hub

    FIGS = (IDS.fig_scatter, IDS.fig_matrix, IDS.fig_roc, IDS.fig_dist)

    @app.callback(
        Output(IDS.store_focus, "data"), Output(IDS.int_click, "data"),
        Output(IDS.thr_pick, "data"),
        *[Output(f, "clickData") for f in FIGS],
        *[Input(f, "clickData") for f in FIGS],
        Input(IDS.btn_inspector_close, "n_clicks"),
        Input(IDS.ctrl_model, "value"),
        Input({"type": "sel-row", "row": ALL}, "n_clicks"),
        Input(IDS.op_pick, "data"),
        State(IDS.store_mods, "data"), State(IDS.ctrl_matrix_mode, "value"),
        State(IDS.store_focus, "data"), State(IDS.store_fit, "data"),
        State(IDS.ctrl_labels, "value"), State(IDS.ctrl_click_mode, "value"),
        prevent_initial_call=True)
    @trace
    def _click(sc_click, mx_click, roc_click, dist_click, _close, model,
               _picked, op, mods, mode, prev, fit, label_keys, click_mode):
        """The one consumer of every plot click (it also clears clickData so
        the SAME point can be clicked again).

            click            inspect the vector / pair / class
            Ctrl+click       add or remove it from the selection
                             (the "click" switch swaps these two)
            tree junction    select the classes under it
            ROC curve, distribution lane   that class's threshold moves there
        """
        trig = ctx.triggered_id

        def out(focus=no_update, intent=no_update, pick=no_update):
            return (focus, intent, pick,
                    *[None if f == trig else no_update for f in FIGS])

        if isinstance(trig, dict) and trig.get("type") == "sel-row":
            if not ctx.triggered[0]["value"] or model not in hub.models:
                raise PreventUpdate        # list just (re)drawn
            return out({"model": model, "rows": [int(trig["row"])]})
        if trig == IDS.op_pick:            # a ROC operating point (sync.js)
            if not op or model not in hub.models:
                raise PreventUpdate
            return out({"model": model, "labels": [str(op["label"])]})
        if trig in (IDS.btn_inspector_close, IDS.ctrl_model):
            if not prev:
                raise PreventUpdate
            return out(None)
        if not model or model not in hub.models:
            raise PreventUpdate
        pts = _points({IDS.fig_scatter: sc_click, IDS.fig_matrix: mx_click,
                       IDS.fig_roc: roc_click, IDS.fig_dist: dist_click}
                      .get(trig))
        if not pts:
            raise PreventUpdate            # our own clickData reset
        pt, fr = pts[0], hub.frame(model)
        cd = pt.get("customdata")
        ctrl = bool((mods or {}).get("ctrl")) != (click_mode == "select")

        def vectors(rows: list[int]):
            if ctrl:
                return out(intent=new_intent(model, "toggle", "rows", rows=rows))
            return out({"model": model, "rows": rows})

        def classes(labels: list[str]):
            labels = list(dict.fromkeys(labels))
            if ctrl:
                return out(intent=class_intent(fr, model, label_keys, labels))
            return out({"model": model, "labels": labels})

        if trig == IDS.fig_scatter:
            if isinstance(cd, list) and len(cd) == 4 and 0 <= int(cd[0]) < fr.n:
                return vectors([int(cd[0])])
            if isinstance(cd, list) and len(cd) == 2:      # a centroid
                return classes([str(cd[0])])
            return out()
        known = set((fit or {}).get("labels") or [])

        def threshold(label, value):           # every panel follows the store
            if str(label) not in known:
                return out()
            return out(pick={"label": str(label), "thr": float(value),
                             "seq": time.time_ns()})

        if trig == IDS.fig_roc:
            if isinstance(cd, list) and len(cd) == 4:      # operating point
                return classes([str(cd[0])])
            if isinstance(cd, list) and len(cd) == 2:      # a point ON a curve
                return threshold(cd[0], cd[1])
            return out()
        if trig == IDS.fig_dist:               # a spot in a class's lane
            if isinstance(cd, list) and len(cd) == 2:
                return threshold(cd[0], cd[1])
            return out()
        if mode == "tree":                     # a junction / leaf: select
            if isinstance(cd, list) and len(cd) == 2:
                return out(intent=class_intent(
                    fr, model, label_keys, str(cd[0]).split(TREE_SEP)))
            return out()
        ya, xa = str(pt.get("y", "")), str(pt.get("x", ""))
        if mode == "vectors":
            rows = [fr.row_of(v.split(SEP)[0]) for v in (ya, xa)]
            rows = list(dict.fromkeys(r for r in rows if r is not None))
            return vectors(rows) if rows else out()
        return classes([v if v in known else v.rsplit(" #", 1)[0]  # "lbl #2"
                        for v in (ya, xa) if v != "rejected"])  # = prototype

    @app.callback(
        Output(IDS.inspector_audio, "children"),
        Input(IDS.store_focus, "data"), State(IDS.store_fit, "data"))
    def _inspector_audio(focus, fit):
        """Audio depends on WHICH vectors only — it is not re-rendered (and
        playback not interrupted) when thresholds or the method change."""
        if not focus or not fit or fit.get("error"):
            return []
        try:
            an = engine.analysis(fit["spec"])
            rows, _ = state.focus_parts(an, focus)
            return insp.audio_view(hub, an, rows) if rows else []
        except Exception as exc:
            return insp.note(f"audio unavailable: {type(exc).__name__}: {exc}")

    closed = ("drawer", [], "Inspector")

    @panel(app, "inspector",
           [Output(IDS.inspector, "className"),
            Output(IDS.inspector_body, "children"),
            Output(IDS.inspector_title, "children")],
           on_error=lambda exc, v: (
               "drawer open", insp.note(f"{type(exc).__name__}: {exc}"),
               "Inspector"))
    def _inspector(v):
        fit, focus = v["fit"], v["focus"]
        if not focus or fit.get("error"):
            return closed
        an = engine.analysis(fit["spec"])
        rows, labels = state.focus_parts(an, focus)
        tv = state.thr_values(v["thr"])
        if rows:
            return ("drawer open", insp.rows_view(an, rows, tv, v["alpha"]),
                    "Vector" if len(rows) == 1 else "Pair of vectors")
        if labels:
            return ("drawer open", insp.labels_view(an, labels, tv, v["alpha"]),
                    "Class" if len(labels) == 1 else "Class pair")
        return closed
