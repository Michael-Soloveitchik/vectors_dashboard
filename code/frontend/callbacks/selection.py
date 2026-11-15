"""Selection = clauses per source, resolved by a strategy (after Mosaic).

    source              clause
    drag (box / lasso)  {"rows"}                 src "lasso"
    class picker, class-tree junction, Ctrl+click on a class,
      inspector "select its class"   {"classes", "label_keys"}   src "classes"
    inspector "select neighbours / speaker"      {"rows"}        src "rows"
    Ctrl+click a point / matrix cell   toggles those vectors in the
    Shift+drag                         adds the box to the  } current selection

    strategy (sidebar):  replace = newest clause only · add = union ·
                         intersect = rows in every clause

Why two stages. Dash applies only the LAST response when calls of one
callback overlap, so a single multi-source writer loses a real update
whenever a no-op call (an echo, a stale click) arrives just behind it.
Here every source has its OWN callback that only records an intent
``{"seq", …}``; one reducer folds all intents newer than the store's
``seq``. A dropped reducer response loses nothing — the next call still
sees the intents. Same-source overlap means "the newer gesture wins",
which is the correct outcome.
"""
from __future__ import annotations

import copy
import time

import numpy as np
from dash import ALL, Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from ..debug_log import dlog, trace
from ..engine import Engine, state
from ..ids import IDS

REDRAW_ECHO_S = 2.5
CAPTURE_ECHO_NS = 1_500_000_000
LIST_MAX = 60
INTENTS = [IDS.int_lasso, IDS.int_classes, IDS.int_click, IDS.int_insp,
           IDS.int_reset]


def _points(data: dict | None) -> list[dict]:
    return (data or {}).get("points") or []


def new_intent(model: str, op: str, src: str, **body) -> dict:
    return {"seq": time.time_ns(), "model": model, "op": op, "src": src, **body}


def keys_of(fr, label_keys) -> list[str]:
    return [k for k in (label_keys or []) if k in fr.df.columns]


def class_intent(fr, model, label_keys, names) -> dict:
    keys = keys_of(fr, label_keys)
    y = fr.labels(keys)
    names = [c for c in dict.fromkeys(str(n) for n in names) if (y == c).any()]
    if not names:
        raise PreventUpdate
    return new_intent(model, "set", "classes", classes=names, label_keys=keys)


def _clauses(sel: dict | None, model: str) -> dict:
    return dict((sel or {}).get("clauses") or {}) \
        if (sel or {}).get("model") == model else {}


def register(app, engine: Engine) -> None:
    hub = engine.hub

    def frame(model):
        if not model or model not in hub.models:
            raise PreventUpdate
        return hub.frame(model)

    intent = new_intent

    @app.callback(
        Output(IDS.int_lasso, "data"), Output(IDS.lasso_echo, "data"),
        Input(IDS.fig_scatter, "selectedData"),
        State(IDS.ctrl_model, "value"), State(IDS.drawn_scatter, "data"),
        State(IDS.store_mods, "data"), State(IDS.store_sel, "data"),
        prevent_initial_call=True)
    @trace
    def _intent_lasso(selected, model, drawn, mods, sel):
        """→ (intent, echo). Plotly has ALREADY re-dimmed the points for
        this gesture on its own; when the gesture is not accepted as a
        selection, ``echo`` makes the projection re-assert the store's."""
        fr = frame(model)
        ignore = (no_update, time.time_ns())
        cur = _clauses(sel, model).get("lasso")
        rows = sorted({int(p["customdata"][0]) for p in _points(selected)
                       if isinstance(p.get("customdata"), list)
                       and len(p["customdata"]) == 4
                       and 0 <= int(p["customdata"][0]) < fr.n})
        dragged = "range" in (selected or {}) or "lassoPoints" in (selected or {})
        if rows and not dragged:
            return ignore            # Plotly's click-select: a click inspects
        if not rows:
            # An empty event is the user's double-click "deselect" — unless
            # it is an echo: Plotly also emits one when the figure is
            # redrawn (theme, refit …) and when the app drops the lasso
            # outline right after capturing a selection.
            fresh = time.time_ns() - int((sel or {}).get("seq") or 0) \
                < CAPTURE_ECHO_NS
            if not _clauses(sel, model) or fresh \
                    or time.time() - float(drawn or 0) < REDRAW_ECHO_S:
                return ignore
            return intent(model, "reset", "clear"), no_update
        if (mods or {}).get("shift"):          # Shift+drag: add to selection
            return intent(model, "add", "rows", rows=rows), no_update
        if cur and rows == cur.get("rows"):
            raise PreventUpdate      # Plotly re-emits after a redraw
        return intent(model, "set", "lasso", rows=rows), no_update

    @app.callback(
        Output(IDS.int_classes, "data"), Input(IDS.ctrl_sel_classes, "value"),
        State(IDS.ctrl_model, "value"), State(IDS.ctrl_labels, "value"),
        State(IDS.store_sel, "data"), prevent_initial_call=True)
    @trace
    def _intent_classes(classes, model, label_keys, sel):
        fr = frame(model)
        cur = _clauses(sel, model).get("classes")
        if not classes:
            if cur is None:
                raise PreventUpdate
            return intent(model, "drop", "classes")
        new = class_intent(fr, model, label_keys, classes)
        if cur and cur.get("classes") == new["classes"] \
                and cur.get("label_keys") == new["label_keys"]:
            raise PreventUpdate      # the picker mirroring the store
        return new

    @app.callback(
        Output(IDS.int_insp, "data"),
        Input({"type": "insp-act", "act": ALL}, "n_clicks"),
        State(IDS.store_focus, "data"), State(IDS.ctrl_model, "value"),
        State(IDS.ctrl_labels, "value"), prevent_initial_call=True)
    @trace
    def _intent_inspector(_n, focus, model, label_keys):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not ctx.triggered[0]["value"]:
            raise PreventUpdate      # buttons just (re)drawn
        fr = frame(model)
        rows = [int(r) for r in (focus or {}).get("rows") or []
                if (focus or {}).get("model") == model and 0 <= int(r) < fr.n]
        if not rows:
            raise PreventUpdate
        row, act = rows[0], trig.get("act")
        if act == "class":
            y = fr.labels(keys_of(fr, label_keys))
            return class_intent(fr, model, label_keys, [y[row]])
        if act == "nbrs":
            picked = [row] + [r for r, _ in state.neighbours(fr, row)]
        elif act == "speaker" and "speaker_id" in fr.df.columns:
            col = fr.df["speaker_id"].astype(str).to_numpy()
            picked = np.flatnonzero(col == col[row]).tolist()
        else:
            raise PreventUpdate
        return intent(model, "set", "rows", rows=sorted(set(picked)))

    @app.callback(
        Output(IDS.int_reset, "data"), Input(IDS.btn_sel_clear, "n_clicks"),
        Input(IDS.ctrl_model, "value"), Input(IDS.ctrl_labels, "value"),
        prevent_initial_call=True)
    @trace
    def _intent_reset(_clear, model, _labels):
        why = {IDS.btn_sel_clear: "clear", IDS.ctrl_model: "model",
               IDS.ctrl_labels: "labels"}.get(ctx.triggered_id)
        if not why or not model:
            raise PreventUpdate
        return intent(model, "reset", why)

    @app.callback(Output(IDS.sel_list, "children"),
                  Input(IDS.store_sel, "data"), State(IDS.ctrl_model, "value"))
    def _sel_list(sel, model):
        """The selected vectors as a clickable list (first ``LIST_MAX``):
        click = inspect / listen."""
        if not model or model not in hub.models:
            return []
        fr = hub.frame(model)
        rows = state.resolve(fr, sel)
        out = [html.Button(fr.sids[r], id={"type": "sel-row", "row": r},
                           n_clicks=0, className="sel-item", title="inspect")
               for r in rows[:LIST_MAX]]
        if len(rows) > LIST_MAX:
            out.append(html.Span(f"+ {len(rows) - LIST_MAX:,} more",
                                 className="note"))
        return out

    @app.callback(
        Output(IDS.store_sel, "data"), Output(IDS.badge_selection, "children"),
        Output(IDS.badge_selection, "className"),
        Output(IDS.ctrl_sel_classes, "value"),
        *[Input(i, "data") for i in INTENTS],
        Input(IDS.ctrl_sel_mode, "value"),
        State(IDS.store_sel, "data"), State(IDS.ctrl_model, "value"),
        prevent_initial_call=True)
    @trace
    def _reduce(*args):
        intents, mode = args[:len(INTENTS)], args[len(INTENTS)] or "single"
        prev, model = args[-2], args[-1]
        fr = frame(model)
        sel = copy.deepcopy(prev) if (prev or {}).get("model") == model else \
            {"model": model, "clauses": {}, "seq": 0}
        todo = sorted((i for i in intents if i and i.get("model") == model
                       and i["seq"] > sel["seq"]), key=lambda i: i["seq"])
        for it in todo:              # every intent not folded in yet, in order
            cl = sel["clauses"]
            if it["op"] == "reset":
                if it["src"] == "labels":
                    cl.pop("classes", None)     # class names no longer apply
                else:
                    cl.clear()
            elif it["op"] == "drop":
                cl.pop(it["src"], None)
            elif it["op"] in ("toggle", "add"):   # edit the CURRENT selection
                base = set(state.resolve(fr, {**sel, "mode": mode}))
                picked = set(state._rows(fr.n, it.get("rows")))
                rows = base ^ picked if it["op"] == "toggle" else base | picked
                cl.clear()
                if rows:
                    cl["rows"] = {"rows": sorted(rows), "seq": it["seq"]}
            else:
                body = {k: v for k, v in it.items()
                        if k not in ("model", "op", "src")}
                if mode == "single":
                    cl.clear()
                cl[it["src"]] = body
            sel["seq"] = it["seq"]
        if mode == "single" and len(sel["clauses"]) > 1:   # strategy switched
            newest = max(sel["clauses"], key=lambda s: sel["clauses"][s]["seq"])
            sel["clauses"] = {newest: sel["clauses"][newest]}
        sel["mode"] = mode
        if sel == prev:
            raise PreventUpdate
        n = len(state.resolve(fr, sel))
        parts = " ∩ " if mode == "intersect" else " ∪ "
        note = (f"  ({parts.join(sel['clauses'])})"
                if len(sel["clauses"]) > 1 else "")
        dlog.event("selection", clauses=list(sel["clauses"]), mode=mode, n=n,
                   folded=[f"{i['op']}:{i['src']}" for i in todo])
        return (sel, f"{n:,} selected{note}", "badge on" if n else "badge",
                (sel["clauses"].get("classes") or {}).get("classes") or [])
