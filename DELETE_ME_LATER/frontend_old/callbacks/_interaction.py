"""Interaction callbacks — selection, view mode, threshold sliders."""
from __future__ import annotations

import pandas as pd
from dash import ALL, Input, Output, State, ctx, dcc, html

from ..debug_log import trace
from ..ids import IDS


def register(app, df: pd.DataFrame) -> None:
    @app.callback(
        Output(IDS.store_aggregated, "data"),
        Output(IDS.badge_selection, "children"),
        Input(IDS.fig_pca, "selectedData"),
        Input(IDS.btn_reset_aggregation, "n_clicks"),
        Input(IDS.ctrl_label_allow, "value"),
        Input(IDS.btn_reset_labels, "n_clicks"),
        State(IDS.store_aggregated, "data"),
        State(IDS.ctrl_model, "value"),
        State(IDS.ctrl_labels, "value"),
    )
    @trace
    def _selection_to_agg(selected, reset_clicks, allow_labels,
                          reset_labels_clicks, prev, model, label_keys):
        triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
        df_m = (df[df["model_name"] == model].reset_index(drop=True)
                if model else df.iloc[0:0])
        total = len(df_m)
        if triggered == IDS.btn_reset_aggregation:
            return [], f"0/{total}"
        if triggered == IDS.btn_reset_labels:
            return [], f"0/{total}"
        if triggered == IDS.ctrl_label_allow and label_keys:
            allow = set(allow_labels or [])
            if not allow:
                return [], f"0/{total}"
            ys = df_m[label_keys].astype(str).agg(" | ".join, axis=1)
            sids = [str(s) for s, y in zip(df_m["session_id"].astype(str), ys)
                    if y in allow]
            return sids, f"{len(sids)}/{total}"
        if not selected or not selected.get("points"):
            return prev or [], f"{len(prev or [])}/{total}"
        sids = []
        for pt in selected["points"]:
            cd = pt.get("customdata")
            if isinstance(cd, list) and len(cd) >= 2:
                sids.append(str(cd[1]))
            elif "hovertext" in pt:
                sids.append(str(pt["hovertext"]))
        sids = list(dict.fromkeys(sids))
        return sids, f"{len(sids)}/{total}"

    @app.callback(
        Output(IDS.store_view_mode, "data"),
        Input(IDS.ctrl_view_mode, "n_clicks"),
        Input(IDS.btn_view_centroids, "n_clicks"),
        Input(IDS.btn_view_mode_matrix, "n_clicks"),
        State(IDS.store_view_mode, "data"),
    )
    @trace
    def _view_mode_sync(vec_clicks, cen_clicks, matrix_clicks, prev):
        """Unambiguous view-mode selector using two explicit buttons.

        Each button click is a definitive "set to X" signal — there is no
        RadioItems ambiguity where re-clicking the already-selected option
        emits no event.  The matrix header button retains its toggle
        behaviour for quick switching from the confusion-matrix panel.
        """
        triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
        if triggered == IDS.ctrl_view_mode:
            return "vectors"
        if triggered == IDS.btn_view_centroids:
            return "centroids"
        if triggered == IDS.btn_view_mode_matrix and matrix_clicks:
            return "centroids" if (prev or "vectors") == "vectors" else "vectors"
        return prev or "vectors"

    @app.callback(
        Output(IDS.store_cm_click_pair, "data"),
        Input(IDS.fig_cm, "clickData"),
        prevent_initial_call=True,
    )
    @trace
    def _cm_click(click_data):
        if not click_data or not click_data.get("points"):
            return None
        pt = click_data["points"][0]
        cd = pt.get("customdata")
        if isinstance(cd, list) and len(cd) >= 2:
            return {"row": str(cd[0]), "col": str(cd[1])}
        return None

    @app.callback(
        Output(IDS.ctrl_matrix_sort, "options"),
        Output(IDS.ctrl_matrix_sort, "value"),
        Input(IDS.ctrl_labels, "value"),
    )
    @trace
    def _sort_options(label_keys):
        # Sort dropdown lists EVERY df column (so user can sort CM by
        # any field, not just the active label_keys). Centroid column
        # excluded — non-sortable vector. `random` + `hierarchical` are
        # synthetic options that always live at the top.
        base = ["random", "hierarchical"]
        all_cols = [c for c in df.columns if c != "centroid"]
        opts = [{"label": v, "value": v} for v in base + all_cols]
        # Pre-fill sort with the active label_keys so the CM is grouped
        # by class on first paint instead of arbitrary row order.
        return opts, list(label_keys or [])

    @app.callback(
        Output(IDS.ctrl_view_mode, "style"),
        Output(IDS.btn_view_centroids, "style"),
        Input(IDS.store_view_mode, "data"),
    )
    @trace
    def _view_mode_button_styles(mode):
        on = {"background": "#4ecdc4", "color": "#000",
              "border": "none", "borderRadius": "4px",
              "padding": "4px 10px", "fontSize": "11px",
              "cursor": "pointer", "fontWeight": "600"}
        off = {"background": "#2a2e38", "color": "#e6e8ee",
               "border": "none", "borderRadius": "4px",
               "padding": "4px 10px", "fontSize": "11px",
               "cursor": "pointer"}
        is_centroids = (mode or "vectors") == "centroids"
        return (off if is_centroids else on,
                on if is_centroids else off)

    @app.callback(
        Output(IDS.ctrl_n_centroids, "disabled"),
        Output(IDS.ctrl_centroid_approach, "options"),
        Output(IDS.ctrl_centroid_approach, "value"),
        Output(IDS.ctrl_centroid_approach, "disabled"),
        Input(IDS.ctrl_methodology, "value"),
    )
    @trace
    def _gate_multi_centroid(methodology):
        # Per-methodology: which split approaches are theoretically valid.
        # Methodologies w/ no multi-centroid notion get disabled inputs.
        per_method = {
            "NCM": ["auto", "kmeans-sphere", "kmedoids",
                    "agglomerative-cos"],
            "VMFMixture": ["auto", "vmf-em"],
            "MultiProxyAnchor": ["auto", "kmeans-sphere"],
        }
        if methodology in per_method:
            opts = per_method[methodology]
            return (False,
                    [{"label": o, "value": o} for o in opts],
                    "auto", False)
        # Single-centroid (kernel / KNN / cosine / etc.) — disable both.
        return (True,
                [{"label": "auto", "value": "auto"}],
                "auto", True)

    @app.callback(
        Output("tsne-param-wrap", "style"),
        Output("umap-param-wrap", "style"),
        Input(IDS.ctrl_reduction, "value"),
    )
    @trace
    def _toggle_reduction_params(method):
        hidden = {"display": "none"}
        shown = {"display": "flex", "marginTop": "6px",
                 "alignItems": "center", "gap": "8px"}
        m = (method or "").strip().lower().replace("-", "")
        return (shown if m == "tsne" else hidden,
                shown if m == "umap" else hidden)

    @app.callback(
        Output(IDS.ctrl_label_allow, "options"),
        Output(IDS.ctrl_label_allow, "value"),
        Input(IDS.ctrl_model, "value"),
        Input(IDS.ctrl_labels, "value"),
        Input(IDS.btn_reset_labels, "n_clicks"),
    )
    @trace
    def _populate_label_allow(model, label_keys, _reset_clicks):
        if not model or not label_keys:
            return [], []
        df_m = df[df["model_name"] == model]
        if df_m.empty:
            return [], []
        ys = df_m[label_keys].astype(str).agg(" | ".join, axis=1)
        opts = sorted(set(ys))
        return [{"label": v, "value": v} for v in opts], []

    @app.callback(
        Output(IDS.threshold_panel, "children"),
        Input(IDS.store_centroids, "data"),
        Input(IDS.store_thresholds, "data"),
    )
    @trace
    def _render_thresholds(state, thresholds):
        if not state or not state.get("centroids"):
            return []
        labels = sorted(state["centroids"].keys())
        # Only use threshold values for classes that belong to the current
        # centroid set — drop stale keys from a previous label selection.
        current_label_set = set(labels)
        thr = {k: v for k, v in (thresholds or {}).items()
               if k in current_label_set}
        rows = []
        for lbl in labels:
            v = float(thr.get(lbl, 0.0))
            rows.append(html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "8px", "marginBottom": "4px"},
                children=[
                    html.Span(lbl, style={"fontSize": "10px",
                                          "minWidth": "60px",
                                          "color": "#8a8f99"}),
                    html.Button("◀", id={"type": "thr-step",
                                          "label": lbl, "dir": "L"},
                                n_clicks=0,
                                style={"background": "#2a2e38",
                                       "color": "#e6e8ee",
                                       "border": "none", "padding": "0 4px",
                                       "fontSize": "10px",
                                       "borderRadius": "3px",
                                       "cursor": "pointer"}),
                    html.Div(style={"flex": 1}, children=[
                        dcc.Slider(
                            id={"type": "thr-slider", "label": lbl},
                            min=0.0, max=1.0, step=0.005, value=v,
                            marks=None,
                            updatemode="drag",
                            tooltip={"placement": "bottom",
                                     "always_visible": True},
                        ),
                    ]),
                    html.Button("▶", id={"type": "thr-step",
                                          "label": lbl, "dir": "R"},
                                n_clicks=0,
                                style={"background": "#2a2e38",
                                       "color": "#e6e8ee",
                                       "border": "none", "padding": "0 4px",
                                       "fontSize": "10px",
                                       "borderRadius": "3px",
                                       "cursor": "pointer"}),
                ],
            ))
        return rows

    @app.callback(
        Output(IDS.store_thresholds, "data", allow_duplicate=True),
        Input({"type": "thr-step", "label": ALL, "dir": ALL}, "n_clicks"),
        State({"type": "thr-step", "label": ALL, "dir": ALL}, "id"),
        State(IDS.store_thresholds, "data"),
        State(IDS.store_centroids, "data"),
        State(IDS.ctrl_model, "value"),
        prevent_initial_call=True,
    )
    @trace
    def _thr_step_buttons(clicks, ids_, prev, state, model):
        # ◀/▶ buttons advance to the NEXT score boundary (one point in
        # or out per click).  Algorithm:
        #   1. Build per-class normalized scores via rep API
        #   2. Sort unique scores
        #   3. j = bisect for current threshold
        #   4. ▶ → j+1, ◀ → j−1 (clamped)
        # This keeps one-point-per-click semantics regardless of how
        # smooth the score distribution is.
        import numpy as np
        triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
        if not isinstance(triggered, dict):
            return prev or {}
        out = dict(prev or {})
        lbl = triggered.get("label")
        dr = triggered.get("dir")
        cur = float(out.get(lbl, 0.0))
        if not state or not model:
            return out
        df_m = df[df["model_name"] == model].reset_index(drop=True)
        if df_m.empty:
            return out
        try:
            from ._fit import build_rep_from_state, _stack
            V, _ = _stack(df_m)
            rep = build_rep_from_state(state)
            ordered = list(rep.class_order_fn())
            if lbl not in [str(o) for o in ordered]:
                return out
            k = [str(o) for o in ordered].index(lbl)
            _, _, votes = rep.predict(V)
            raw = np.asarray(votes[:, k], dtype=np.float64)
            norm = np.asarray(rep.normalize_score(raw), dtype=np.float64)
            uniq = np.unique(np.round(norm, 6))
            j = int(np.searchsorted(uniq, cur, side="left"))
            if dr == "R":
                j = min(len(uniq) - 1, j + 1)
            else:
                j = max(0, j - 1)
            out[lbl] = float(uniq[j])
        except Exception as exc:
            print(f"[thr-step] fallback {exc}", flush=True)
            delta = 0.01 if dr == "R" else -0.01
            out[lbl] = max(0.0, min(1.0, cur + delta))
        return out

    @app.callback(
        Output(IDS.store_thresholds, "data", allow_duplicate=True),
        Input({"type": "thr-slider", "label": ALL}, "value"),
        State({"type": "thr-slider", "label": ALL}, "id"),
        State(IDS.store_thresholds, "data"),
        prevent_initial_call=True,
    )
    @trace
    def _threshold_slider_to_store(values, ids_, prev):
        out = dict(prev or {})
        current_labels = set(out.keys())
        for vid, val in zip(ids_ or [], values or []):
            lbl = vid["label"]
            # Only update classes that currently exist in the threshold store.
            # This prevents stale sliders from a previous label selection
            # from re-injecting their values after a multilabel change resets
            # the store to the new class set.
            if val is not None and lbl in current_labels:
                out[lbl] = float(val)
        return out

    @app.callback(
        Output(IDS.store_thresholds, "data", allow_duplicate=True),
        Input(IDS.ctrl_pair_cos_thr, "value"),
        Input(IDS.store_centroids, "data"),
        State(IDS.store_thresholds, "data"),
        prevent_initial_call=True,
    )
    @trace
    def _selection_thr_fans_out(sel_thr, centroid_state, prev):
        # Single "selection thr (sel-vs-rest)" slider mirrors its value
        # to EVERY per-class threshold key so all class lines on
        # PCA / dist / ROC move in lockstep with the selection line.
        #
        # Listens BOTH to slider changes AND to store_centroids changes
        # (i.e. methodology / labels switch). On every fire:
        #   - if slider has a value → write {lbl: v} for ALL labels
        #     known from centroid state (the authoritative class list)
        #   - else fall back to prev or {}
        # This eliminates the bug where switching methodology cleared
        # store_thresholds, then the next slider move had `prev={}`
        # and fan-out returned empty — leaving rings stuck at thr=0
        # default ("cover all points") regardless of slider value.
        if sel_thr is None:
            return prev or {}
        # Authoritative label list = current methodology's classes.
        labels: list = []
        if centroid_state and isinstance(centroid_state, dict):
            cents = centroid_state.get("centroids") or {}
            labels = sorted(cents.keys())
        if not labels and prev:
            labels = list(prev.keys())
        if not labels:
            return prev or {}
        v = float(sel_thr)
        return {str(lbl): v for lbl in labels}


    @app.callback(
        Output(IDS.per_class_n_panel, "children"),
        Input(IDS.store_centroids, "data"),
        Input(IDS.ctrl_methodology, "value"),
        State(IDS.ctrl_n_centroids, "value"),
        State(IDS.store_per_class_n, "data"),
    )
    @trace
    def _render_per_class_n(state, methodology, n_global, per_class_n):
        from ._fit import _multi_arg
        from backend.representations import METHOD_REGISTRY
        if not state or not methodology:
            return []
        cls = METHOD_REGISTRY.get(methodology)
        if cls is None or _multi_arg(cls) is None:
            return [html.Div("(methodology has no multi-centroid arg)",
                             style={"fontSize": "10px",
                                    "color": "#8a8f99",
                                    "fontStyle": "italic",
                                    "marginTop": "4px"})]
        labels = sorted(state.get("centroids", {}).keys())
        cur = per_class_n or {}
        rows = []
        for lbl in labels:
            v = int(cur.get(lbl, n_global or 1))
            rows.append(html.Div(
                style={"display": "flex", "alignItems": "center",
                       "gap": "6px"},
                children=[
                    html.Span(str(lbl), style={"fontSize": "10px",
                                               "minWidth": "70px",
                                               "color": "#8a8f99"}),
                    dcc.Input(
                        id={"type": "n-class", "label": str(lbl)},
                        type="number", value=v, min=1, max=12, step=1,
                        debounce=True,
                        style={"width": "55px", "fontSize": "11px"},
                    ),
                ],
            ))
        return rows

    @app.callback(
        Output(IDS.store_per_class_n, "data"),
        Input({"type": "n-class", "label": ALL}, "value"),
        State({"type": "n-class", "label": ALL}, "id"),
        prevent_initial_call=True,
    )
    @trace
    def _per_class_n_to_store(values, ids_):
        out = {}
        for vid, val in zip(ids_ or [], values or []):
            if val is not None:
                out[vid["label"]] = int(val)
        return out


__all__ = ["register"]
