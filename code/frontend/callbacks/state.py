"""Controls → fit spec, plus small UI gates, theme and the tab id.

Cascade (each arrow is exactly one callback run per user action):

    model ─► label options/value ─► class options ─► _fit ─► store_fit
                                                       (token = hash(spec))
"""
from __future__ import annotations

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .. import figures as F
from ..debug_log import dlog, trace
from ..engine import MULTI_CENTROID_ARG, Engine, FitSpec, state
from ..engine.concurrency import Superseded
from ..engine.matrix import SYNTHETIC_SORTS
from ..ids import IDS
from .panels import spec_from_controls

THEME_JS = """function(n, cur) {
    const KEY = 'vectors-dashboard-theme';
    let t = cur;
    if (!n) {                     // first paint: explicit > saved > OS
        if (t !== 'light' && t !== 'dark') {
            try { t = window.localStorage.getItem(KEY); } catch (e) { t = null; }
        }
        if (t !== 'light' && t !== 'dark') {
            t = (window.matchMedia &&
                 window.matchMedia('(prefers-color-scheme: light)').matches)
                ? 'light' : 'dark';
        }
    } else {
        t = (cur === 'dark') ? 'light' : 'dark';
        try { window.localStorage.setItem(KEY, t); } catch (e) {}
    }
    document.documentElement.setAttribute('data-theme', t);
    return [t, t === 'dark' ? 'switch to light' : 'switch to dark'];
}"""

CLIENT_JS = """function(_) {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 10);
}"""


def register(app, engine: Engine) -> None:
    hub = engine.hub

    @app.callback(
        Output(IDS.ctrl_labels, "options"), Output(IDS.ctrl_labels, "value"),
        Output(IDS.ctrl_matrix_sort, "options"),
        Output(IDS.ctrl_matrix_sort, "value"),
        Output(IDS.ctrl_color_by, "options"), Output(IDS.ctrl_color_by, "value"),
        Input(IDS.ctrl_model, "value"), Input(IDS.store_labels, "data"),
        State(IDS.ctrl_labels, "value"), State(IDS.ctrl_color_by, "value"),
        State(IDS.ctrl_matrix_sort, "value"))
    @trace
    def _model_changed(model, _labels, cur, color_by, sort):
        if not model or model not in hub.models:
            raise PreventUpdate
        cols = hub.label_columns(model)
        keep = [c for c in (cur or []) if c in cols] or hub.default_labels(model)
        df = hub.frame(model).df
        extra = [c for c in df.columns if c != "model_name"]
        colour = (["class", F.OWN_SCORE] + cols
                  + [c for c in extra if c not in cols and F.is_continuous(df[c])])
        sorts = list(SYNTHETIC_SORTS) + extra
        new_model = ctx.triggered_id in (IDS.ctrl_model, None)
        # None = never set (two start-up runs of this callback can overlap,
        # and only the last answer is applied): class blocks by default.
        sort = ["class"] if new_model or sort is None else \
            [k for k in sort if k in sorts]
        return (cols, keep, sorts, sort,
                colour, color_by if color_by in colour else "class")

    @app.callback(
        Output(IDS.ctrl_skip, "options"), Output(IDS.ctrl_skip, "value"),
        Output(IDS.ctrl_sel_classes, "options"),
        Input(IDS.ctrl_model, "value"), Input(IDS.ctrl_labels, "value"),
        Input(IDS.store_labels, "data"), State(IDS.ctrl_skip, "value"))
    @trace
    def _class_options(model, label_keys, _labels, skip):
        if not model or model not in hub.models or not label_keys:
            return [], [], []
        fr = hub.frame(model)
        keys = [k for k in label_keys if k in fr.df.columns]
        classes = sorted(set(fr.labels(keys).tolist()))
        opts = set(classes)
        if len(keys) > 1:              # also allow skipping one column value
            for c in keys:
                opts.update(fr.df[c].astype(str).unique().tolist())
        opts = sorted(opts)
        for c in hub.label_columns(model):     # filters: any column=value
            vals = fr.df[c].astype(str).unique().tolist()
            if c not in keys and len(vals) <= 40:
                opts += [f"{c}={v}" for v in sorted(vals)]
        return opts, [s for s in (skip or []) if s in opts], classes

    @app.callback(
        Output(IDS.ctrl_aggregator, "disabled"),
        Output(IDS.ctrl_n_centroids, "disabled"),
        Input(IDS.ctrl_methodology, "value"))
    def _gate_method_args(methodology):
        from backend.representations import METHOD_REGISTRY
        cls = METHOD_REGISTRY.get(methodology)
        uses_agg = bool(getattr(cls, "_uses_aggregator", False))
        return (not uses_agg), (methodology not in MULTI_CENTROID_ARG)

    @app.callback(
        Output(IDS.store_fit, "data"), Output(IDS.fit_status, "children"),
        Output(IDS.fit_status, "className"),
        Input(IDS.ctrl_model, "value"), Input(IDS.ctrl_labels, "value"),
        Input(IDS.ctrl_methodology, "value"),
        Input(IDS.ctrl_aggregator, "value"), Input(IDS.ctrl_skip, "value"),
        Input(IDS.ctrl_n_centroids, "value"),
        Input(IDS.store_per_class_n, "data"), Input(IDS.store_labels, "data"),
        Input(IDS.store_kick, "data"),
        State(IDS.store_client, "data"), State(IDS.store_fit, "data"))
    @trace
    def _fit(model, label_keys, methodology, aggregator, skip, n_global, pcn,
             labels, _kick, client, prev):
        sd = spec_from_controls(model, label_keys, methodology, aggregator,
                                skip, n_global, pcn, labels)
        if sd is None:
            raise PreventUpdate
        tok = state.token(sd)          # same token the wanted-state uses
        if ctx.triggered_id in (IDS.store_kick, IDS.store_labels) \
                and (prev or {}).get("token") == tok:
            raise PreventUpdate        # ping / labels not used by this fit
        try:
            an = engine.analysis(FitSpec.from_dict(sd), client=client)
            labels, error = an.labels, an.error
        except Superseded:             # a newer fit from this tab is queued
            raise PreventUpdate
        except Exception as exc:       # bad control state must not kill the UI
            dlog.event("fit_error", err=f"{type(exc).__name__}: {exc}")
            data = {"spec": sd, "labels": [], "token": tok,
                    "error": f"{type(exc).__name__}: {exc}"}
            return data, data["error"], "fit-status err"
        data = {"spec": sd, "labels": labels, "error": error, "token": tok}
        if error:
            return data, error, "fit-status err"
        msg = (f"{sd['methodology']} · {an.k} classes · fitted on "
               f"{int(an.fit_mask.sum()):,} of {an.frame.n:,} vectors · "
               f"dim {an.frame.dim}")
        return data, msg, "fit-status"

    # ── per-class centroid counts ───────────────────────────────────
    @app.callback(
        Output(IDS.per_class_n_panel, "children"),
        Input(IDS.store_fit, "data"), Input(IDS.ctrl_n_centroids, "value"),
        State(IDS.store_per_class_n, "data"),
        State({"type": "n-class", "label": ALL}, "id"))
    @trace
    def _per_class_panel(fit, n_global, pcn, cur_ids):
        if not fit or fit.get("error"):
            return []
        if fit["spec"]["methodology"] not in MULTI_CENTROID_ARG:
            return html.Span("this methodology has one prototype per class",
                             className="note")
        labels = fit["labels"]
        reset = ctx.triggered_id == IDS.ctrl_n_centroids
        if [i["label"] for i in (cur_ids or [])] == labels and not reset:
            return no_update
        out = []
        for lbl in labels:
            v = int(n_global or 1) if reset else int(
                (pcn or {}).get(lbl, n_global or 1))
            out += [html.Span(lbl, className="thr-name", title=lbl),
                    dcc.Input(id={"type": "n-class", "label": lbl},
                              type="number", value=v, min=1, max=12, step=1,
                              debounce=True, className="num",
                              style={"width": "58px"})]
        return out

    @app.callback(
        Output(IDS.store_per_class_n, "data"),
        Input({"type": "n-class", "label": ALL}, "value"),
        State({"type": "n-class", "label": ALL}, "id"),
        State(IDS.ctrl_n_centroids, "value"),
        State(IDS.store_per_class_n, "data"), prevent_initial_call=True)
    def _per_class_store(values, ids_, n_global, prev):
        g = int(n_global or 1)
        out = {i["label"]: int(v) for i, v in zip(ids_ or [], values or [])
               if v is not None and int(v) != g}
        return no_update if out == (prev or {}) else out

    # ── small UI gates ──────────────────────────────────────────────
    @app.callback(
        Output(IDS.wrap_tsne, "className"), Output(IDS.wrap_umap, "className"),
        Output(IDS.wrap_ptsne, "className"), Output(IDS.wrap_sigma, "className"),
        Input(IDS.ctrl_reduction, "value"), Input(IDS.ctrl_scatter_opts, "value"))
    def _reduction_params(method, opts):
        m = (method or "").lower().replace("-", "")
        show = lambda on: "tool-param" if on else "tool-param hidden"  # noqa: E731
        return (show(m in ("tsne", "ptsne")), show(m == "umap"),
                show(m == "ptsne"), show("ellipses" in (opts or [])))

    @app.callback(Output(IDS.wrap_matrix_sort, "className"),
                  Input(IDS.ctrl_matrix_mode, "value"))
    def _sort_visible(mode):
        return "tool-param grow" + ("" if mode == "vectors" else " hidden")

    app.clientside_callback(
        THEME_JS,
        Output(IDS.store_theme, "data"), Output(IDS.btn_theme, "title"),
        Input(IDS.btn_theme, "n_clicks"), State(IDS.store_theme, "data"))
    app.clientside_callback(
        CLIENT_JS, Output(IDS.store_client, "data"),
        Input(IDS.btn_theme, "id"))
