"""Render callbacks — one per subplot.

Each callback fires on its narrow Inputs only so dragging a threshold
slider doesn't refit centroids.

Fast path: when only `store_aggregated` triggers `_render_pca`, return a
`dash.Patch` that updates `selectedpoints` + clears layout.selections in
place — no PCA recompute, no figure rebuild.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, ctx, no_update

from .. import figures as F
from ..debug_log import trace
from ..ids import IDS
from ._fit import (build_rep_from_state, _stack, get_cached_rep,
                    cached_score_xy_matrix)


def _blank() -> go.Figure:
    return go.Figure().update_layout(template="plotly_dark")


def _patch_pca_selection(agg, current_fig):
    """In-place selectedpoints + clear stale lasso rectangles."""
    if not current_fig or not current_fig.get("data"):
        return no_update
    agg_set = {str(s) for s in (agg or [])}
    is_full = not agg_set
    patched = Patch()
    n_patched = 0
    for i, trace in enumerate(current_fig["data"]):
        if trace.get("mode") != "markers":
            continue
        cd = trace.get("customdata") or []
        if not cd:
            continue
        try:
            row0_len = len(cd[0])
        except (TypeError, IndexError):
            continue
        if row0_len < 2:
            continue
        if is_full:
            patched["data"][i]["selectedpoints"] = list(range(len(cd)))
        else:
            patched["data"][i]["selectedpoints"] = [
                j for j, row in enumerate(cd) if str(row[1]) in agg_set
            ]
        n_patched += 1
    patched["layout"]["selections"] = []
    patched["layout"]["dragmode"] = "select"
    print(f"[fast-patch] n_patched={n_patched}  is_full={is_full}  "
          f"agg_size={len(agg_set)}", flush=True)
    return patched


def register(app, df: pd.DataFrame) -> None:
    @app.callback(  # _render_pca
        Output(IDS.fig_pca, "figure"),
        Input(IDS.store_centroids, "data"),
        Input(IDS.store_thresholds, "data"),
        Input(IDS.ctrl_labels, "value"),
        Input(IDS.store_cm_click_pair, "data"),
        Input(IDS.ctrl_show_ellipses, "value"),
        Input(IDS.ctrl_ellipse_n_sigma, "value"),
        Input(IDS.store_aggregated, "data"),
        Input(IDS.ctrl_reduction, "value"),
        Input(IDS.ctrl_perplexity, "value"),
        Input(IDS.ctrl_n_neighbors, "value"),
        Input(IDS.ctrl_min_dist, "value"),
        Input(IDS.ctrl_density_contour, "value"),
        Input(IDS.ctrl_alpha, "value"),
        State(IDS.fig_pca, "figure"),
        State(IDS.ctrl_model, "value"),
    )
    @trace
    def _render_pca(state, thresholds, label_keys, cm_click,
                    show_ellipses, n_sigma, agg, reduction,
                    perplexity, n_neighbors, min_dist, density, alpha,
                    current_fig, model):
        # Fast path — selection alone: patch selectedpoints, no rebuild.
        triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
        if (triggered == IDS.store_aggregated
                and current_fig and current_fig.get("data")):
            return _patch_pca_selection(agg, current_fig)
        if not state or not model:
            return _blank()
        df_m = df[df["model_name"] == model].reset_index(drop=True)
        if df_m.empty:
            return _blank()
        V, sids = _stack(df_m)
        rep = (get_cached_rep(model, state, V, label_keys, df_m)
               or build_rep_from_state(state))
        try:
            rep.impl.NORM_ALPHA = float(alpha if alpha is not None else 0.5)
        except Exception:
            pass
        red_params: dict = {"random_state": 42}
        rmethod = (reduction or "PCA").strip().lower().replace("-", "")
        if rmethod == "tsne":
            red_params["perplexity"] = float(perplexity or 30)
        elif rmethod == "umap":
            red_params["n_neighbors"] = int(n_neighbors or 15)
            red_params["min_dist"] = float(min_dist or 0.1)
        fig = F.pca_fig(
            rep=rep, df_m=df_m, V=V, sids=sids, state=state,
            method_label=state.get("methodology", "?"),
            model_name=model, class_thresholds=thresholds or {},
            label_keys=label_keys,
            show_ellipses=bool(show_ellipses and "on" in show_ellipses),
            n_sigma=float(n_sigma or 2.0),
            agg=agg,
            reduction=reduction or "PCA",
            reduction_params=red_params,
            density=bool(density and "on" in density),
            excluded_sids=(state or {}).get("excluded_sids") or [],
        )
        if cm_click and isinstance(cm_click, dict):
            try:
                fig = F.add_cm_highlight(
                    fig, df_m=df_m, V=V, sids=sids, state=state,
                    label_keys=label_keys,
                    row_id=str(cm_click.get("row")),
                    col_id=str(cm_click.get("col")),
                )
            except Exception as exc:
                print(f"[render_pca] cm-highlight skipped: {exc}", flush=True)
        return fig

    @app.callback(
        # CM SHRINKS to lasso selection. Empty selection ⇒ full CM.
        # Matrix = rep.score_XY(V, V) cached once per (methodology, α);
        # selection just slices indices. α propagated so normalize_score
        # is consistent with dist + ROC + PCA.
        Output(IDS.fig_cm, "figure"),
        Input(IDS.store_centroids, "data"),
        Input(IDS.store_view_mode, "data"),
        Input(IDS.store_aggregated, "data"),
        Input(IDS.ctrl_color_abs, "value"),
        Input(IDS.ctrl_color_power, "value"),
        Input(IDS.ctrl_color_range, "value"),
        Input(IDS.ctrl_matrix_sort, "value"),
        Input(IDS.ctrl_labels, "value"),
        Input(IDS.ctrl_alpha, "value"),
        State(IDS.ctrl_model, "value"),
    )
    @trace
    def _render_cm(state, view_mode, agg, abs_v, power, color_range,
                   sort_keys, _label_keys, alpha, model):
        if not model:
            return _blank()
        df_m = df[df["model_name"] == model].reset_index(drop=True)
        if df_m.empty:
            return _blank()
        V, sids = _stack(df_m)
        # Per spec: CM = V_score @ V_score.T where V_score comes ONLY
        # via the methodology API (rep.predict + rep.normalize_score).
        # Switching methodology now changes CM because each rep maps V
        # → its own (N, K) normalized vote matrix.
        V_score = V
        score_mat = None
        if state:
            try:
                import numpy as np
                # Prefer cached fitted rep (kept by autofit); falls back
                # to centroid-only rebuild for legacy state dicts.
                rep = (get_cached_rep(model, state, V, _label_keys, df_m)
                       or build_rep_from_state(state))
                methodology = state.get("methodology", "?")
                # PER USER SPEC: CM matrix = rep.score_XY(V, V) (rep's
                # pair-primitive). Cached once per (model, methodology,
                # α); slicing on selection is downstream.
                try:
                    score_mat = cached_score_xy_matrix(
                        rep, V, model, methodology, alpha)
                except Exception as exc:
                    print(f"[render_cm] score_XY fallback to votes-V: "
                          f"{exc}", flush=True)
                    score_mat = None
                _, _, votes = rep.predict(V)
                votes = np.asarray(votes, dtype=np.float64)
                if votes.ndim != 2 or votes.shape[0] == 0:
                    raise ValueError(f"votes shape {votes.shape}")
                cols = []
                for k in range(votes.shape[1]):
                    col = votes[:, k]
                    try:
                        nc = rep.normalize_score(col)
                        nc = np.asarray(nc, dtype=np.float64).ravel()
                        if nc.size != col.size:
                            nc = col   # fallback: raw col
                    except Exception:
                        nc = col
                    cols.append(nc)
                V_score = np.column_stack(cols).astype(np.float64)
            except Exception as exc:
                print(f"[render_cm] score-V fallback to raw V: {exc}",
                      flush=True)
                V_score = V
        lo = float((color_range or [-1, 1])[0])
        hi = float((color_range or [-1, 1])[1])
        method_label = (state or {}).get("methodology", "?")
        return F.cm_fig(
            df_m=df_m, V=V_score, sids=sids, state=state,
            view_mode=view_mode or "vectors",
            method_label=method_label, model_name=model,
            abs_on=bool(abs_v and "on" in abs_v),
            power=float(power or 1.0), lo=lo, hi=hi, agg=agg,
            sort_keys=sort_keys or [],
            score_matrix=score_mat,
        )

    @app.callback(
        Output(IDS.fig_dist, "figure"),
        Output(IDS.fig_dist, "style"),
        Input(IDS.store_centroids, "data"),
        Input(IDS.store_thresholds, "data"),
        Input(IDS.store_view_mode, "data"),
        Input(IDS.store_aggregated, "data"),
        Input(IDS.ctrl_score_domain, "value"),
        Input(IDS.ctrl_dist_view, "value"),
        Input(IDS.ctrl_labels, "value"),
        Input(IDS.ctrl_alpha, "value"),
        State(IDS.ctrl_model, "value"),
    )
    @trace
    def _render_dist(state, thresholds, view_mode, agg, score_domain,
                     dist_view, label_keys, alpha, model):
        if not state or not model:
            return _blank(), {"height": "300px", "width": "100%"}
        df_m = df[df["model_name"] == model].reset_index(drop=True)
        if df_m.empty:
            return _blank(), {"height": "300px", "width": "100%"}
        V, sids = _stack(df_m)
        rep = (get_cached_rep(model, state, V, label_keys, df_m)
               or build_rep_from_state(state))
        try:
            rep.impl.NORM_ALPHA = float(alpha if alpha is not None else 0.5)
        except Exception:
            pass
        n_classes = max(len(state.get("centroids", {})), 1)
        # +1 lane for the (SELECTED) sub-row when selection non-empty.
        n_lanes = n_classes + (1 if (agg or []) else 0)
        height_px = max(420, 130 * n_lanes)
        fig = F.dist_fig(
            rep=rep, df_m=df_m, V=V, sids=sids, state=state,
            view_mode=view_mode or "vectors",
            method_label=state.get("methodology", "?"), model_name=model,
            class_thresholds=thresholds or {}, agg=agg,
            label_keys=label_keys,
            score_domain=score_domain or "raw",
            view=dist_view or "box",
        )
        return fig, {"height": f"{height_px}px", "width": "100%"}

    @app.callback(
        Output(IDS.fig_roc, "figure"),
        Input(IDS.store_centroids, "data"),
        Input(IDS.store_thresholds, "data"),
        Input(IDS.store_aggregated, "data"),
        Input(IDS.ctrl_roc_log_x, "value"),
        Input(IDS.ctrl_roc_log_y, "value"),
        Input(IDS.ctrl_roc_mode, "value"),
        Input(IDS.ctrl_labels, "value"),
        Input(IDS.ctrl_pair_cos_thr, "value"),
        Input(IDS.ctrl_pair_norm_thr, "value"),
        Input(IDS.ctrl_alpha, "value"),
        Input(IDS.store_cm_click_pair, "data"),
        Input(IDS.ctrl_curve_mode, "value"),
        State(IDS.ctrl_model, "value"),
    )
    @trace
    def _render_roc(state, thresholds, agg, log_x, log_y, mode,
                    label_keys, pair_cos, pair_norm, alpha,
                    cm_click, curve_mode, model):
        if not state or not model:
            return _blank()
        df_m = df[df["model_name"] == model].reset_index(drop=True)
        if df_m.empty:
            return _blank()
        V, sids = _stack(df_m)
        rep = (get_cached_rep(model, state, V, label_keys, df_m)
               or build_rep_from_state(state))
        try:
            rep.impl.NORM_ALPHA = float(alpha if alpha is not None else 0.5)
        except Exception:
            pass

        # CM-click → resolve a session_id to its class label so the ROC
        # panel can emphasise that class's operating-point marker.
        # The CM stores {"row": session_id, "col": session_id} (vectors mode)
        # or {"row": class_label, "col": class_label} (centroids mode).
        # In centroids mode the id IS the class label; in vectors mode we
        # look up device_id from df_m.  We take the row id arbitrarily —
        # any cell in the clicked row belongs to the same class.
        selected_class: str | None = None
        if cm_click and isinstance(cm_click, dict):
            row_id = str(cm_click.get("row", ""))
            if row_id:
                centroid_labels = sorted((state.get("centroids") or {}).keys())
                if row_id in centroid_labels:
                    # centroids mode: row_id is already a class label
                    selected_class = row_id
                else:
                    # vectors mode: map session_id → device_id via df_m
                    lk = list(label_keys or []) or None
                    if lk:
                        match = df_m[df_m["session_id"].astype(str) == row_id]
                        if not match.empty:
                            selected_class = " | ".join(
                                str(match.iloc[0][k]) for k in lk
                                if k in match.columns
                            )
                    else:
                        match = df_m[df_m["session_id"].astype(str) == row_id]
                        if not match.empty and "device_id" in match.columns:
                            selected_class = str(match.iloc[0]["device_id"])

        return F.roc_fig(
            rep=rep, df_m=df_m, V=V, sids=sids, state=state,
            method_label=state.get("methodology", "?"), model_name=model,
            class_thresholds=thresholds or {}, agg=agg,
            label_keys=label_keys,
            log_x=bool(log_x and "on" in log_x),
            log_y=bool(log_y and "on" in log_y),
            mode=mode or "all",
            pair_cos_thr=float(pair_cos) if pair_cos is not None else None,
            pair_norm_thr=float(pair_norm) if pair_norm is not None else None,
            selected_class=selected_class,
            curve_mode=curve_mode or "ROC",
        )


__all__ = ["register"]
