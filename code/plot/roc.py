"""build_roc_fig — pair·cosine + pair·normalized + per-device ROC figure."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import auc, roc_curve

from .scoring import _hover, _op_point
from .roc_trace import _pair_op_trace
from .style import _BTN


__all__ = ["build_roc_fig"]


def build_roc_fig(*, rep, V_te, dev_te, devices_sorted, cmap,
                  method_label: str, model_name: str, y_pair, P_te, n_tr: int,
                  class_labels, class_thresholds: dict | None = None,
                  pair_cos_thr: float | None = None,
                  pair_norm_thr: float | None = None):
    fig = go.Figure()

    # pair·cosine — methodology-owned pipeline on rep.
    pc = rep.cosine_pairs(V_te)
    fp, tp, th = roc_curve(y_pair, pc, drop_intermediate=False)
    fig.add_trace(go.Scatter(
        x=fp, y=tp, mode="lines+markers",
        name=f"pair·cosine (AUC={auc(fp, tp):.3f})", legendgroup="pair-cos",
        line=dict(width=2, color="#ff9f43"),
        marker=dict(size=4, color="#ff9f43", opacity=0.75),
        text=_hover(fp, tp, th),
        hovertemplate="<b>pair·cosine</b><br>%{text}<extra></extra>",
    ))
    # Static pair-cos op marker dropped — confused vs per-class sweep that
    # follows the same selection-thr slider via fan-out (see P4 in
    # threshold_sync_plan.md).

    # pair·normalized — inline chain: per-class normalize(predict) →
    # P_norm @ P_norm.T → LOCAL rank on the pair vector. The pair
    # distribution is not the class-score distribution, so the final
    # normalize must rank within the pair vector itself — otherwise
    # most pair values saturate at 1.0 against the class _norm_ref and
    # the slider grid collapses near the top of the ROC curve.
    from scipy.stats import rankdata as _rankdata
    _, _, _raw = rep.predict(V_te)
    _raw = np.asarray(_raw, dtype=np.float64)
    _P_norm = np.column_stack([rep.normalize_score(_raw[:, k])
                               for k in range(_raw.shape[1])])
    _ri, _ci = np.triu_indices(_P_norm.shape[0], k=1)
    _pair_raw = (_P_norm @ _P_norm.T)[_ri, _ci]
    pp = ((_rankdata(_pair_raw, method="average") - 1.0) /
          max(_pair_raw.size - 1, 1))
    fp, tp, th = roc_curve(y_pair, pp, drop_intermediate=False)
    fig.add_trace(go.Scatter(
        x=fp, y=tp, mode="lines+markers",
        name=f"pair·normalized (AUC={auc(fp, tp):.3f})", legendgroup="pair-norm",
        line=dict(width=2, color="#4ecdc4", dash="dot"),
        marker=dict(size=4, color="#4ecdc4", opacity=0.75),
        text=_hover(fp, tp, th),
        hovertemplate="<b>pair·normalized</b><br>%{text}<extra></extra>",
    ))
    # Static pair-norm op marker dropped (see P4 in threshold_sync_plan.md).

    # per-device one-vs-rest + (optional) threshold operating-point marker.
    # Each per-class curve is two traces in the same legendgroup:
    #   [curve]       — the line+markers of the full ROC
    #   [op-point]    — a fat marker at (FPR, TPR) for the current threshold.
    #                   Exists even when no threshold is set (hidden then);
    #                   future widget callbacks update its (x, y) only.
    class_labels_list = list(class_labels)
    thr_map = class_thresholds if isinstance(class_thresholds, dict) else {}
    for lbl in devices_sorted:
        if lbl not in class_labels_list:
            continue
        ci = class_labels_list.index(lbl)
        raw_scores = P_te[:, ci]                              # native posterior
        s = rep.normalize_score(raw_scores)                   # rank-normed for ROC
        y_c = (dev_te == lbl).astype(np.int32)
        if y_c.sum() in (0, len(y_c)):
            continue
        fp, tp, th = roc_curve(y_c, s, drop_intermediate=False)
        col = cmap[lbl]
        fig.add_trace(go.Scatter(
            x=fp, y=tp, mode="lines+markers",
            name=f"{lbl} (AUC={auc(fp, tp):.3f})", legendgroup=lbl,
            line=dict(width=1.8, color=col),
            marker=dict(size=4, color=col, opacity=0.75),
            text=_hover(fp, tp, th),
            hovertemplate=f"<b>{lbl}</b><br>%{{text}}<extra></extra>",
        ))
        # Operating-point fat marker — driven by the SLIDER threshold which
        # lives in NORMALIZED [0,1] space (single source of truth). Compute
        # (FPR, TPR) against `s` (already rep.normalize_score'd) using the
        # raw slider value. Eliminates the norm→raw round-trip degeneracy
        # on clipped methodologies (ACPAN, ArcCosineSVM) where multiple raw
        # values map to the same norm and the marker froze.
        thr_v = thr_map.get(lbl)
        if isinstance(thr_v, (int, float)) and not isinstance(thr_v, bool):
            if float(thr_v) <= 0.0:
                op_fpr, op_tpr = 1.0, 1.0
            elif float(thr_v) >= 1.0:
                op_fpr, op_tpr = 0.0, 0.0
            else:
                op_fpr, op_tpr = _op_point(y_c, s, float(thr_v))
            op_visible = True
            op_label = f"{lbl} @ thr={thr_v:.2f}"
        else:
            op_fpr, op_tpr = 1.0, 1.0
            op_visible = False
            op_label = f"{lbl} @ thr=—"
        fig.add_trace(go.Scatter(
            x=[op_fpr], y=[op_tpr], mode="markers",
            name=op_label, legendgroup=lbl, showlegend=False,
            marker=dict(size=13, color=col, symbol="circle",
                        line=dict(width=2, color="rgba(255,255,255,0.9)")),
            hovertemplate=(f"<b>{lbl}</b> operating point<br>"
                           "thr=%{customdata:.3f}<br>"
                           "fpr=%{x:.3f}<br>tpr=%{y:.3f}<extra></extra>"),
            customdata=[float(thr_v) if isinstance(thr_v, (int, float))
                        and not isinstance(thr_v, bool) else float("nan")],
            visible=op_visible,
        ))

    fig.add_trace(go.Scatter(
        x=[1e-4, 1], y=[1e-4, 1], mode="lines", name="chance",
        line=dict(dash="dash", width=1, color="rgba(255,255,255,0.5)"),
        hoverinfo="skip", showlegend=False,
    ))

    # Slide-toggle buttons: label shows the NEXT action; args/args2 swap
    # on successive clicks so the label always tells you what will happen.
    x_toggle = dict(type="buttons", direction="right", x=0.01, y=1.02,
                    xanchor="left", yanchor="bottom", **_BTN, buttons=[dict(
                        label="x → log", method="update",
                        args=[{}, {"xaxis.type": "log",
                                   "xaxis.range": [-4, 0],
                                   "updatemenus[0].buttons[0].label": "x → lin"}],
                        args2=[{}, {"xaxis.type": "linear",
                                    "xaxis.range": [-0.02, 1.02],
                                    "updatemenus[0].buttons[0].label": "x → log"}],
                    )])
    y_toggle = dict(type="buttons", direction="right", x=0.14, y=1.02,
                    xanchor="left", yanchor="bottom", **_BTN, buttons=[dict(
                        label="y → log", method="update",
                        args=[{}, {"yaxis.type": "log",
                                   "yaxis.range": [-4, 0],
                                   "updatemenus[1].buttons[0].label": "y → lin"}],
                        args2=[{}, {"yaxis.type": "linear",
                                    "yaxis.range": [-0.02, 1.02],
                                    "updatemenus[1].buttons[0].label": "y → log"}],
                    )])
    fig.update_layout(
        updatemenus=[x_toggle, y_toggle],
        title=f"ROC — {method_label} — {model_name}",
        xaxis=dict(title="False positive rate", range=[-0.02, 1.02]),
        yaxis=dict(title="True positive rate", range=[-0.02, 1.02]),
        template="plotly_dark", width=900, height=620,
        legend=dict(font=dict(size=10), tracegroupgap=3), margin=dict(t=110),
    )
    return fig
