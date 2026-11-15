"""build_pca_fig — sample + centroid scatter with cosine-τ rings."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from .ring import _ring_xy


__all__ = ["build_pca_fig"]


def build_pca_fig(*, rep, V_emb, centroid_rows, centroid_labels,
                  plot_df, cmap, method_label, model_name, run_pca,
                  class_thresholds: dict | None = None):
    joint = (np.concatenate([V_emb, *centroid_rows], axis=0)
             if centroid_rows else V_emb)
    Z = run_pca(joint, {"random_state": 42})
    Z_s, Z_c = Z[:len(V_emb)], Z[len(V_emb):]
    plot_df = plot_df.copy()
    plot_df["PC1"], plot_df["PC2"] = Z_s[:, 0], Z_s[:, 1]

    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="device_id",
        color_discrete_map=cmap,
        hover_data=["voice_id", "session_id"],
        template="plotly_dark",
        title=f"PCA — {method_label} — {model_name}",
        width=820, height=620,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.82))

    if Z_c.shape[0] > 0:
        fig.add_trace(go.Scatter(
            x=Z_c[:, 0], y=Z_c[:, 1], mode="markers+text",
            name=f"{method_label} centroids",
            text=[str(l) for l in centroid_labels],
            textposition="top center", textfont=dict(size=10),
            marker=dict(size=17, symbol="x",
                        line=dict(width=2, color="rgba(255,255,255,0.85)"),
                        color=[cmap.get(l, "#ffffff") for l in centroid_labels]),
            hovertemplate="<b>%{text}</b><br>centroid<extra></extra>",
        ))

        # pca_scale: RMS spread of projected data in PCA space.
        # The ring radius formula sqrt(2-2τ) lives on the unit sphere.
        # After PCA projection the coordinate magnitudes shrink, so we
        # multiply by pca_scale to keep rings visually commensurate with
        # the scatter cluster extents. Using the joint pts array (samples
        # + centroids) ensures the scale accounts for the full displayed
        # range.
        _pts_for_scale = np.concatenate([Z_s, Z_c], axis=0)
        _rms = float(np.sqrt(np.mean(_pts_for_scale ** 2))) if _pts_for_scale.size else 1.0
        pca_scale = max(_rms, 1e-6)

        thr_map = class_thresholds if isinstance(class_thresholds, dict) else {}
        for i, lbl in enumerate(centroid_labels):
            cx, cy = float(Z_c[i, 0]), float(Z_c[i, 1])
            tau = thr_map.get(lbl)
            # 1) Empirical envelope — max-accepted same-class PCA radius.
            #    Solid line. Overwritten dynamically by interactive.py to
            #    track the predict-set as the slider moves.
            rx, ry = _ring_xy(cx, cy, tau, pca_scale=pca_scale)
            fig.add_trace(go.Scatter(
                x=rx, y=ry, mode="lines",
                name="accepted envelope (dash)", legendgroup="threshold_ring",
                showlegend=(i == 0), meta=[cx, cy, str(lbl)],
                line=dict(color=cmap.get(lbl, "#ffffff"),
                          width=1.6, dash="dash"),
                hovertemplate=(f"<b>{lbl}</b> accepted envelope<extra></extra>"),
            ))

    pts = np.concatenate([Z_s, Z_c], axis=0) if Z_c.shape[0] > 0 else Z_s
    if pts.shape[0] > 0:
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        x_pad = (x_max - x_min) * 0.05 or 1.0
        y_pad = (y_max - y_min) * 0.05 or 1.0
        x_lo, x_hi = x_min - x_pad, x_max + x_pad
        y_lo, y_hi = y_min - y_pad, y_max + y_pad
        cx, hw = (x_lo + x_hi) / 2, (x_hi - x_lo) / 2
        cy, hh = (y_lo + y_hi) / 2, (y_hi - y_lo) / 2
        zoom = 1.5
        fig.update_xaxes(range=[cx - hw * zoom, cx + hw * zoom])
        fig.update_yaxes(range=[cy - hh * zoom, cy + hh * zoom])

    fig.update_layout(legend_title_text="device")
    return fig
