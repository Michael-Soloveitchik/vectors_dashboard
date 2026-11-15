"""ROC panel adapter — split from figures.py."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from plot.roc import build_roc_fig

from .figures_common import (palette_for, _centroids, _y_from_label_keys,
                              _trapz)


def roc_fig(*, rep, df_m, V, sids, state, method_label, model_name,
            class_thresholds, agg=None, label_keys=None,
            log_x: bool = False, log_y: bool = False,
            mode: str = "all",
            pair_cos_thr: float | None = None,
            pair_norm_thr: float | None = None,
            selected_class: str | None = None,
            curve_mode: str = "ROC"):
    """ROC panel with normalized-thr op markers + selection overlay."""
    rows, labels = _centroids(state)
    if V.shape[0] == 0 or not labels:
        return go.Figure().update_layout(template="plotly_dark", height=420)
    y = _y_from_label_keys(df_m, label_keys)
    devices_sorted = sorted(set(y) & set(labels)) or sorted(set(y))
    cmap = palette_for(devices_sorted)
    from frontend.callbacks._fit import cached_predict
    methodology = getattr(rep, "_method_name", "?")
    _, _, P_te = cached_predict(rep, V, methodology)
    dev_arr = np.asarray(y)
    n = len(dev_arr)
    V_v = V
    if n < 2:
        return go.Figure().update_layout(template="plotly_dark", height=420)
    ri, ci = np.triu_indices(n, k=1)
    y_pair = (dev_arr[ri] == dev_arr[ci]).astype(np.int32)
    safe_labels = (list(labels)[:P_te.shape[1]] if P_te.ndim == 2
                   else list(labels))
    safe_devices = ([d for d in devices_sorted if d in safe_labels]
                    or safe_labels)
    norm_thr = {str(k): float(v)
                for k, v in (class_thresholds or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    fig = build_roc_fig(
        rep=rep, V_te=V_v, dev_te=dev_arr, devices_sorted=safe_devices,
        cmap=cmap, method_label=method_label, model_name=model_name,
        y_pair=y_pair, P_te=P_te, n_tr=V_v.shape[0],
        class_labels=safe_labels, class_thresholds=norm_thr,
        pair_cos_thr=pair_cos_thr, pair_norm_thr=pair_norm_thr,
    )
    if log_x:
        fig.update_xaxes(type="log", range=[-4, 0])
    if log_y:
        fig.update_yaxes(type="log", range=[-4, 0])
    fig.update_layout(uirevision=f"roc-{model_name}-{log_x}-{log_y}-{mode}")
    if mode in ("per-class", "pair"):
        for tr in fig.data:
            grp = (getattr(tr, "legendgroup", "") or "")
            is_pair = grp.startswith("pair")
            if mode == "per-class" and is_pair:
                tr.visible = False
            if mode == "pair" and not is_pair and grp != "":
                tr.visible = False

    if selected_class:
        op_x, op_y = None, None
        for tr in fig.data:
            grp = (getattr(tr, "legendgroup", "") or "")
            if (grp == str(selected_class)
                    and getattr(tr, "mode", "") == "markers"
                    and getattr(tr, "showlegend", True) is False):
                xs = list(tr.x or [])
                ys = list(tr.y or [])
                if xs and ys:
                    op_x, op_y = float(xs[0]), float(ys[0])
                break
        if op_x is not None:
            fig.add_trace(go.Scatter(
                x=[op_x], y=[op_y],
                mode="markers",
                name=f"{selected_class} (selected)",
                showlegend=False,
                hovertemplate=(
                    f"<b>{selected_class}</b> — CM-click emphasis<br>"
                    "fpr=%{x:.3f}<br>tpr=%{y:.3f}<extra></extra>"
                ),
                marker=dict(
                    size=22,
                    symbol="circle-open",
                    color="rgba(0,0,0,0)",
                    line=dict(width=3, color="#ffd700"),
                ),
            ))

    agg_set = {str(s) for s in (agg or [])}
    if agg_set:
        n_sel_dbg = sum(1 for s in sids if str(s) in agg_set)
        print(f"[roc] overlay agg_set={len(agg_set)} sids_full={len(sids)} "
              f"matched={n_sel_dbg}", flush=True)
        _append_roc_selection_overlay(
            fig, rep=rep, V_full=V, dev_full=dev_arr, sids_full=sids,
            agg_set=agg_set, log_x=log_x, pair_cos_thr=pair_cos_thr,
        )

    if (curve_mode or "ROC").upper() == "DET":
        _convert_to_det(fig)

    fig.update_layout(autosize=True, width=None, height=None,
                      margin=dict(t=80, l=50, r=10, b=40))
    return fig


def _convert_to_det(fig) -> None:
    """In-place: rewrite each ROC trace from (FPR, TPR) → (Φ⁻¹(FPR),
    Φ⁻¹(FNR)) using the standard DET probit transform."""
    from scipy.stats import norm
    eps = 1e-6
    for tr in fig.data:
        if not getattr(tr, "x", None) or not getattr(tr, "y", None):
            continue
        try:
            fpr = np.asarray(tr.x, dtype=float)
            tpr = np.asarray(tr.y, dtype=float)
        except Exception:
            continue
        grp = getattr(tr, "legendgroup", "") or ""
        if "selected" in (tr.name or "") and grp == "":
            continue
        fnr = 1.0 - tpr
        fpr_c = np.clip(fpr, eps, 1 - eps)
        fnr_c = np.clip(fnr, eps, 1 - eps)
        tr.x = norm.ppf(fpr_c).tolist()
        tr.y = norm.ppf(fnr_c).tolist()
    pct = [0.001, 0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99, 0.999]
    ticks = norm.ppf(pct).tolist()
    labels = ["0.1%", "1%", "5%", "20%", "50%", "80%", "95%", "99%", "99.9%"]
    fig.update_xaxes(title_text="false-pos rate (probit)",
                     tickmode="array", tickvals=ticks, ticktext=labels,
                     range=[ticks[0] - 0.2, ticks[-1] + 0.2], type="linear")
    fig.update_yaxes(title_text="false-neg rate (probit)",
                     tickmode="array", tickvals=ticks, ticktext=labels,
                     range=[ticks[0] - 0.2, ticks[-1] + 0.2], type="linear")
    fig.update_layout(title_text=fig.layout.title.text.replace("ROC", "DET")
                      if fig.layout.title and fig.layout.title.text
                      else "DET curves")


def _append_roc_selection_overlay(fig, *, rep, V_full, dev_full, sids_full,
                                  agg_set, log_x: bool, pair_cos_thr=None):
    """Selection ROC overlay — sel-vs-rest curve + diamond op marker."""
    from sklearn.metrics import roc_curve
    if not agg_set:
        return
    sids_arr = np.asarray([str(s) for s in sids_full])
    in_sel = np.array([s in agg_set for s in sids_arr], dtype=bool)
    n_sel = int(in_sel.sum())
    if n_sel < 2 or n_sel >= len(sids_full):
        return
    _, _, votes_full = rep.predict(V_full)
    votes_norm = np.column_stack([
        rep.normalize_score(votes_full[:, k])
        for k in range(votes_full.shape[1])
    ])
    s_norm = votes_norm.max(axis=1)
    fpr1, tpr1, thr1 = roc_curve(in_sel.astype(int), s_norm)
    auc1 = float(_trapz(tpr1, fpr1))
    fig.add_trace(go.Scatter(
        x=fpr1, y=tpr1, mode="lines",
        name=f"sel-vs-rest (AUC={auc1:.3f}, n={n_sel})",
        line=dict(color="#ffd700", width=2.6, dash="solid"),
        legendgroup="selection-overlay",
        hovertemplate=("<b>selection vs rest</b><br>"
                       "fpr=%{x:.4f}<br>tpr=%{y:.4f}"
                       "<extra></extra>"),
    ))
    op_thr_norm = float(pair_cos_thr) if pair_cos_thr is not None else 0.0
    idx = int(np.argmin(np.abs(thr1 - op_thr_norm)))
    fig.add_trace(go.Scatter(
        x=[fpr1[idx]], y=[tpr1[idx]], mode="markers",
        marker=dict(size=14, color="#ffd700", symbol="diamond",
                    line=dict(color="#000", width=1.5)),
        name=f"sel-op @ thr={op_thr_norm:.2f}",
        legendgroup="selection-overlay", showlegend=False,
        hovertemplate=("<b>selection op-point</b><br>"
                       "thr=%{customdata:.3f}<br>"
                       "fpr=%{x:.4f}<br>tpr=%{y:.4f}<extra></extra>"),
        customdata=[float(thr1[idx])],
    ))
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=fpr1[idx], x1=fpr1[idx], y0=0, y1=1,
                  line=dict(color="#ffd700", width=1, dash="dot"))
