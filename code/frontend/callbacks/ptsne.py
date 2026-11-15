"""Parametric t-SNE: the "train" button.

``store_ptsne`` = {"model", "perplexity", "points", "rounds"} — what the
server holds for the current settings. The projection panel is drawn
from it, so a finished round redraws the panel like any other input."""
from __future__ import annotations

from dash import Input, Output, ctx, no_update
from dash.exceptions import PreventUpdate

from ..debug_log import trace
from ..engine import Engine, ptsne
from ..engine.project import norm_method, norm_points
from ..ids import IDS


def register(app, engine: Engine) -> None:
    hub = engine.hub

    @app.callback(
        Output(IDS.store_ptsne, "data"), Output(IDS.ptsne_status, "children"),
        Input(IDS.btn_ptsne_train, "n_clicks"),
        Input(IDS.ctrl_reduction, "value"), Input(IDS.ctrl_model, "value"),
        Input(IDS.ctrl_perplexity, "value"),
        Input(IDS.ctrl_max_points, "value"))
    @trace
    def _train(n, reduction, model, perplexity, max_points):
        if norm_method(reduction) != "ptsne" or model not in hub.models:
            raise PreventUpdate
        fr = hub.frame(model)
        perp, pts = float(perplexity or 30), norm_points(max_points)
        try:
            st = (ptsne.train(engine, fr, perp, pts)
                  if ctx.triggered_id == IDS.btn_ptsne_train and n
                  else ptsne.info(engine, fr, perp, pts))
        except Exception as exc:            # e.g. PyTorch missing
            return no_update, f"{type(exc).__name__}: {exc}"
        text = (f"{st['steps']:,} steps · KL {st['loss']:.3f}" if st["rounds"]
                else "not trained")
        return ({"model": model, "perplexity": perp, "points": pts,
                 "rounds": st["rounds"]}, text)
