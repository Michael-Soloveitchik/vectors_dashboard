"""Per-class score distribution adapter — split from figures.py."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from plot.dist import build_dist_fig

from .figures_common import palette_for, _centroids, _y_from_label_keys


def dist_fig(*, rep, df_m, V, sids, state, view_mode, method_label,
             model_name, class_thresholds, agg=None, label_keys=None,
             score_domain: str = "raw", view: str = "box"):
    """Distribution panel — full data baseline + selection overlay
    when `agg` non-empty. Thresholds live in NORMALIZED [0,1] space.
    Reserved SELECTED lane sits at the bottom (always present)."""
    use_raw = (score_domain or "raw").lower() != "normalized"
    rows, labels = _centroids(state)
    toy_thr = class_thresholds or {}
    if view_mode == "centroids" and rows:
        Vc = np.vstack([np.atleast_2d(r) for r in rows]).astype(np.float64)
        src = []
        for r, lbl in zip(rows, labels):
            n = int(np.atleast_2d(r).shape[0])
            src.extend([str(lbl)] * n)
        srcs_sorted = sorted(set(src))
        cmap = palette_for(srcs_sorted)
        fig = build_dist_fig(
            rep=rep, V_te=Vc, dev_te=np.asarray(src),
            devices_sorted=srcs_sorted, cmap=cmap,
            method_label=f"{method_label} (centroids)",
            model_name=model_name, use_raw=True,
            class_thresholds=toy_thr,
            class_labels=labels,
        )
    else:
        y_full = _y_from_label_keys(df_m, label_keys)
        devices_sorted = sorted(set(y_full))
        cmap = palette_for(devices_sorted)
        fig = build_dist_fig(
            rep=rep, V_te=V, dev_te=np.asarray(y_full),
            devices_sorted=devices_sorted, cmap=cmap,
            method_label=method_label, model_name=model_name,
            use_raw=use_raw, class_thresholds=toy_thr,
            class_labels=labels,
        )
        agg_set = {str(s) for s in (agg or [])}
        if agg_set and len(agg_set) < len(sids):
            mask = np.array([str(s) in agg_set for s in sids], dtype=bool)
            if mask.any():
                df_sel = df_m.iloc[mask].reset_index(drop=True)
                V_sel = V[mask]
                y_sel = _y_from_label_keys(df_sel, label_keys)
                _append_selection_overlay(
                    fig, rep=rep, V_sel=V_sel, y_sel=np.asarray(y_sel),
                    devices_sorted=devices_sorted, cmap=cmap,
                    use_raw=use_raw,
                    class_thresholds=class_thresholds or {},
                )
    fig.update_layout(autosize=True, width=None,
                      uirevision=f"dist-{model_name}-{view_mode}-{view}-{score_domain}",
                      margin=dict(t=120, l=80, r=20, b=40))
    meta = getattr(fig.layout, "meta", None) or {}
    if view == "hist" and meta.get("vis_hist"):
        for tr, vis in zip(fig.data, meta["vis_hist"]):
            tr.visible = vis
    return fig


def _append_selection_overlay(fig, *, rep, V_sel, y_sel, devices_sorted,
                              cmap, use_raw: bool,
                              class_thresholds=None) -> None:
    """Selection overlay:
       • golden marker per (class, device) for selected samples
       • dotted vertical thr line per class
       • SELECTED lane Box at bottom (already reserved by build_dist_fig)
    """
    from frontend.callbacks._fit import cached_predict
    methodology = getattr(rep, "_method_name", "?")
    cls_arr, scores, votes = cached_predict(rep, V_sel, methodology)
    ordered = list(rep.class_order_fn())
    thr_map = class_thresholds or {}
    rng = np.random.default_rng(42)
    for i, lbl in enumerate(ordered):
        col = votes[:, i]
        for j, dev in enumerate(devices_sorted):
            mask = (y_sel == dev)
            if not mask.any():
                continue
            xs = col[mask]
            ys = [str(lbl)] * len(xs)
            fig.add_trace(go.Scatter(
                x=xs.tolist(),
                y=ys,
                mode="markers",
                marker=dict(color=cmap.get(dev, "#ffd700"),
                            size=6, opacity=0.85,
                            line=dict(color="#000", width=0.4)),
                hoverinfo="x",
                name=f"{dev} (selected)",
                legendgroup=f"selected-{dev}",
                showlegend=(i == 0),
            ))
        tau_norm = thr_map.get(str(lbl))
        if tau_norm is not None:
            fig.add_trace(go.Scatter(
                x=[float(tau_norm)], y=[str(lbl)],
                mode="markers",
                marker=dict(symbol="line-ns-open", size=40,
                            color="#ffd700",
                            line=dict(color="#ffd700", width=3)),
                name=f"sel-thr {lbl}",
                legendgroup="selection-overlay",
                showlegend=False,
                hovertemplate=(f"<b>{lbl}</b> sel-thr<br>"
                               "norm=%{x:.3f}<extra></extra>"),
            ))
    # Overwrite the reserved SELECTED-lane Box (always present per
    # plot.dist.build_dist_fig) with selection's normalized max-vote.
    SEL_LANE_LABEL = "SELECTED"
    if votes.shape[0] > 0:
        try:
            v_norm = np.column_stack([
                rep.normalize_score(votes[:, k])
                for k in range(votes.shape[1])
            ])
            best_norm = v_norm.max(axis=1)
        except Exception:
            best_norm = votes.max(axis=1)
        n_classes = len(ordered)
        sel_y_idx = float(n_classes)
        sel_box_idx = next(
            (i for i, t in enumerate(fig.data)
             if getattr(t, "name", "") == SEL_LANE_LABEL),
            None,
        )
        if sel_box_idx is not None:
            tr = fig.data[sel_box_idx]
            tr.x = best_norm.tolist()
            tr.y = [sel_y_idx] * len(best_norm)
        else:
            fig.add_trace(go.Box(
                x=best_norm.tolist(),
                y=[sel_y_idx] * len(best_norm),
                orientation="h",
                marker=dict(color="#ffd700", size=6, opacity=0.95,
                            line=dict(color="#000", width=0.5)),
                line=dict(width=1.5, color="#ffd700"),
                fillcolor="rgba(255,215,0,0.15)",
                boxpoints="all", jitter=0.4, pointpos=0.0,
                hoverinfo="x",
                name=SEL_LANE_LABEL,
                legendgroup="selection-overlay",
                showlegend=True,
            ))
