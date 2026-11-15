"""The user's own labels: name the current selection.

The labels live in one extra column (``my_label``, see ``DataHub``), so
they work everywhere a column does: as class labels for the fit, as the
projection colour, as a matrix sort key (the colour strips then follow
it), in the selection composition and in predictions.csv. They are saved
to ``dumps/my_labels_<model>.csv`` and restored on start.

``store_labels = {"model", "rev"}`` is what the rest of the app listens
to; the fit includes ``rev`` only when ``my_label`` is a class label, so
labelling never refits an analysis that does not use it.
"""
from __future__ import annotations

from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from ..debug_log import trace
from ..engine import Engine, state
from ..engine.data import MY_LABEL, UNLABELLED
from ..ids import IDS


def register(app, engine: Engine) -> None:
    hub = engine.hub

    @app.callback(
        Output(IDS.store_labels, "data"), Output(IDS.label_status, "children"),
        Input(IDS.btn_label_apply, "n_clicks"),
        Input(IDS.btn_label_clear, "n_clicks"), Input(IDS.ctrl_model, "value"),
        State(IDS.ctrl_label_name, "value"), State(IDS.store_sel, "data"))
    @trace
    def _labels(_apply, _clear, model, name, sel):
        if not model or model not in hub.models:
            raise PreventUpdate
        fr, trig = hub.frame(model), ctx.triggered_id
        note = ""
        if trig == IDS.btn_label_apply:
            rows = state.resolve(fr, sel)
            if not rows:
                return no_update, "select some vectors first, then label them"
            hub.set_user_label(model, rows, name)
            what = f"“{name.strip()}”" if (name or "").strip() else "unlabelled"
            note = f"{len(rows):,} vectors → {what} · "
        elif trig == IDS.btn_label_clear:
            hub.set_user_label(model, None, None)
            note = "all labels removed · "
        col = (fr.df[MY_LABEL].to_numpy(dtype=object)
               if MY_LABEL in fr.df.columns else None)
        n = int((col != UNLABELLED).sum()) if col is not None else 0
        kinds = len(set(col[col != UNLABELLED].tolist())) if n else 0
        hint = (f"{n:,} labelled in {kinds} label(s) — use “{MY_LABEL}” as "
                "class labels, colour, or matrix sort") if n else \
            "no own labels yet"
        return {"model": model, "rev": hub.labels_rev(model)}, note + hint
