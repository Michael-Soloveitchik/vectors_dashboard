"""CM (confusion matrix) panel adapter — split from figures.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plot.cm import (build_cm_fig, apply_cm_transform,
                     _build_customdata, _class_labels_from_plot_df,
                     _MAX_TICK_LABELS)

from .figures_common import _restrict, _centroids


def _hierarchical_sort(df_v: pd.DataFrame,
                       V_v: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Reorder rows using scipy average-linkage on cosine distance."""
    from scipy.cluster.hierarchy import linkage, leaves_list
    n = V_v.shape[0]
    if n < 3:
        return df_v.reset_index(drop=True), V_v
    Vn = V_v / np.maximum(np.linalg.norm(V_v, axis=1, keepdims=True), 1e-12)
    sim = np.clip(Vn @ Vn.T, -1.0, 1.0)
    dist = 1.0 - sim
    iu = np.triu_indices(n, k=1)
    cond = np.maximum(dist[iu], 0.0)
    Z = linkage(cond, method="average")
    order = leaves_list(Z)
    df_w = df_v.iloc[order].reset_index(drop=True)
    return df_w, V_v[order]


def _sort_df_for_cm(df_v: pd.DataFrame, V_v: np.ndarray,
                    sort_keys: list[str] | None) -> tuple[pd.DataFrame, np.ndarray]:
    if not sort_keys:
        return df_v, V_v
    df_w = df_v.copy()
    if "random" in sort_keys:
        rng = np.random.default_rng(42)
        df_w = df_w.iloc[rng.permutation(len(df_w))].reset_index(drop=True)
    if "hierarchical" in sort_keys:
        df_w, V_w = _hierarchical_sort(
            df_w, V_v if "random" not in sort_keys
            else V_v[df_w.index.to_numpy()])
        df_v = df_w
        V_v = V_w
    real_keys = [k for k in sort_keys
                 if k not in ("random", "hierarchical") and k in df_w.columns]
    if real_keys:
        df_w = df_w.sort_values(real_keys, kind="stable").reset_index(drop=True)
    sid_to_idx = {str(s): i for i, s in enumerate(df_v["session_id"].astype(str))}
    perm = np.array([sid_to_idx[str(s)]
                     for s in df_w["session_id"].astype(str)], dtype=int)
    return df_w, V_v[perm]


def cm_fig(*, df_m, V, sids, state, view_mode, method_label, model_name,
           abs_on, power, lo, hi, agg=None, sort_keys=None,
           score_matrix=None):
    """CM restricts to lasso selection (`agg` = list of session_ids).
    Empty selection → full data.

    `score_matrix` (N×N) optional override. When provided, the heatmap
    z-values come from `rep.score_XY` instead of cosine V@V.T.
    """
    rows, labels = _centroids(state)
    if view_mode == "centroids" and rows:
        fig = build_cm_fig(
            V_emb=V, centroid_rows=rows, centroid_labels=labels,
            plot_df=df_m, method_label=method_label, model_name=model_name,
            mode="centroids", abs_on=abs_on, power=power,
            clip_lo=lo, clip_hi=hi,
        )
    else:
        df_v, V_v, sids_v = _restrict(df_m, V, sids, agg)
        if V_v.shape[0] == 0:
            return go.Figure().update_layout(template="plotly_dark",
                                             title="empty selection")
        df_v, V_v = _sort_df_for_cm(df_v, V_v, sort_keys)
        if score_matrix is not None and view_mode != "centroids":
            try:
                fig = _cm_fig_from_score_matrix(
                    score_matrix=score_matrix, full_sids=sids,
                    sub_sids=sids_v, df_sub=df_v,
                    method_label=method_label, model_name=model_name,
                    abs_on=abs_on, power=power, lo=lo, hi=hi)
            except Exception as exc:
                print(f"[cm_fig] score_matrix path fallback "
                      f"to V@V.T: {exc}", flush=True)
                fig = build_cm_fig(
                    V_emb=V_v, centroid_rows=rows, centroid_labels=labels,
                    plot_df=df_v, method_label=method_label,
                    model_name=model_name, mode="vectors",
                    abs_on=abs_on, power=power,
                    clip_lo=lo, clip_hi=hi)
        else:
            fig = build_cm_fig(
                V_emb=V_v, centroid_rows=rows, centroid_labels=labels,
                plot_df=df_v, method_label=method_label, model_name=model_name,
                mode="vectors", abs_on=abs_on, power=power,
                clip_lo=lo, clip_hi=hi,
            )
    n_sel = len(agg) if agg else 0
    rev = f"cm-{view_mode}-{tuple(sort_keys or ())}-{model_name}-n{n_sel}"
    fig.update_layout(autosize=True, width=None, height=None,
                      uirevision=rev,
                      margin=dict(t=40, l=10, r=10, b=10))
    return fig


def _cm_fig_from_score_matrix(*, score_matrix, full_sids, sub_sids,
                              df_sub, method_label, model_name,
                              abs_on, power, lo, hi):
    """Build CM heatmap directly from a precomputed N×N score matrix
    (output of `rep.score_XY`). Slices rows/cols by `sub_sids`."""
    M = np.asarray(score_matrix, dtype=np.float64)
    full_sids_l = list(full_sids or [])
    sub_sids_l = list(sub_sids or [])
    if not sub_sids_l:
        sub_sids_l = full_sids_l
    sid_to_idx = {sid: i for i, sid in enumerate(full_sids_l)}
    idx = [sid_to_idx[sid] for sid in sub_sids_l if sid in sid_to_idx]
    if not idx:
        return go.Figure().update_layout(template="plotly_dark",
                                         title="empty selection")
    M_sub = M[np.ix_(idx, idx)]
    M_t = apply_cm_transform(M_sub, abs_on, power, lo, hi)
    n = M_t.shape[0]
    cls_labels = _class_labels_from_plot_df(df_sub, sub_sids_l)
    show_ticks = n <= _MAX_TICK_LABELS
    tick_text = cls_labels if show_ticks else []
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=M_t,
        x=list(range(n)), y=list(range(n)),
        colorscale="Viridis",
        zmin=float(lo), zmax=float(hi), zauto=False,
        customdata=_build_customdata(sub_sids_l, cls_labels) if n > 0 else [],
        hovertemplate=(
            "<b>row:</b> %{customdata[0]}"
            " <span style='color:#aaa'>(%{customdata[2]})</span><br>"
            "<b>col:</b> %{customdata[1]}"
            " <span style='color:#aaa'>(%{customdata[3]})</span><br>"
            "<b>score:</b> %{z:.4f}<extra></extra>"
        ),
        hoverongaps=False,
        colorbar=dict(thickness=10, len=0.7),
        name="vectors",
    ))
    title = (f"score_XY matrix (vectors, n={n}) — "
             f"{method_label} — {model_name}")
    axis_common = dict(
        showgrid=False,
        tickmode="array" if show_ticks else "auto",
        tickvals=list(range(n)) if show_ticks else [],
        ticktext=tick_text,
        tickfont=dict(size=9),
        showticklabels=show_ticks,
    )
    fig.update_layout(
        template="plotly_dark",
        width=720, height=720,
        margin=dict(t=70, l=50, r=40, b=40),
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=14)),
        xaxis=dict(**axis_common, title="col →"),
        yaxis=dict(**axis_common, title="← row",
                   autorange="reversed", scaleanchor="x"),
        meta={"mode": "vectors", "ids": sub_sids_l,
              "cls_labels": cls_labels, "src": "score_XY"},
    )
    return fig
