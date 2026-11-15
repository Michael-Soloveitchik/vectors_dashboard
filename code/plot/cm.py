"""build_cm_fig — cosine-similarity matrix heatmap (vectors / centroids).

Mirrors the abs/power/clip pipeline used by the main dashboard
(``backend/confusion_matrix/transforms.py``) but operates on the SAME
inputs the PCA panel consumes — so a click on a CM cell can be mapped
directly to a point in the PCA scatter.

Two modes:
- ``vectors``   : cosine of all sample vectors against each other.
                  ids = ``plot_df['session_id']``.
- ``centroids`` : cosine of all centroid rows against each other.
                  ids = ``centroid_labels``.

Both modes L2-renormalize before matmul so the matrix is always
cosine, regardless of whether the source vectors arrived unit-norm.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


__all__ = ["build_cm_fig", "compute_cm_matrix", "apply_cm_transform"]

_L2_EPS = 1e-12


def _l2norm(V: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(n, _L2_EPS)


def apply_cm_transform(M: np.ndarray, abs_on: bool, power: float,
                       clip_lo: float, clip_hi: float) -> np.ndarray:
    """abs → sign-preserving power → clip. Same order as backend.

    Sign-preserving power keeps ``(-x)**0.5`` from going NaN —
    matches ``backend/confusion_matrix/transforms.py``.
    """
    if M.size == 0:
        return M
    if abs_on:
        M = np.abs(M)
    p = float(power)
    if p != 1.0:
        sign = np.sign(M)
        M = sign * (np.abs(M) ** p)
    lo, hi = (clip_lo, clip_hi) if clip_lo <= clip_hi else (clip_hi, clip_lo)
    return np.clip(M, float(lo), float(hi))


def compute_cm_matrix(*, V_emb, centroid_rows, centroid_labels, plot_df, mode):
    """Return ``(M, ids)``: cosine matrix + row/col session-or-centroid id list."""
    if mode == "centroids":
        if not centroid_rows or len(centroid_rows) == 0:
            return np.zeros((0, 0), dtype=np.float64), []
        rows = []
        for r in centroid_rows:
            a = np.asarray(r, dtype=np.float64)
            rows.append(a.reshape(1, -1) if a.ndim == 1 else a)
        C = np.vstack(rows)
        Cn = _l2norm(C)
        M = (Cn @ Cn.T).astype(np.float64)
        ids = [str(l) for l in centroid_labels]
        return M, ids

    Vn = _l2norm(np.asarray(V_emb, dtype=np.float64))
    M = (Vn @ Vn.T).astype(np.float64)
    if plot_df is not None and "session_id" in plot_df.columns:
        ids = [str(s) for s in plot_df["session_id"].tolist()]
    else:
        ids = [str(i) for i in range(M.shape[0])]
    return M, ids


def _build_customdata(ids, class_labels=None):
    """customdata[i][j] = [row_id, col_id, row_class, col_class].

    class_labels: list of class-label strings parallel to ids.
    When None, class fields fall back to the id itself.
    """
    n = len(ids)
    cls = class_labels if (class_labels and len(class_labels) == n) else ids
    return [[[ids[i], ids[j], cls[i], cls[j]] for j in range(n)]
            for i in range(n)]


def _class_labels_from_plot_df(plot_df, ids):
    """Return a per-row class-label list aligned to ids.

    Looks for ``device_id`` in plot_df (the column the toy pipeline always
    populates with the label used for training). Falls back to ids when the
    column is absent.
    """
    if plot_df is None or "device_id" not in plot_df.columns:
        return list(ids)
    sid_to_cls = {str(r["session_id"]): str(r["device_id"])
                  for _, r in plot_df.iterrows()
                  if "session_id" in plot_df.columns}
    return [sid_to_cls.get(sid, sid) for sid in ids]


_MAX_TICK_LABELS = 40   # suppress tick-labels on very large matrices


def build_cm_fig(*, V_emb, centroid_rows, centroid_labels, plot_df,
                 method_label, model_name, mode="vectors",
                 abs_on=False, power=1.0,
                 clip_lo=-1.0, clip_hi=1.0):
    M_raw, ids = compute_cm_matrix(
        V_emb=V_emb, centroid_rows=centroid_rows,
        centroid_labels=centroid_labels, plot_df=plot_df, mode=mode,
    )
    M = apply_cm_transform(M_raw, abs_on, power, clip_lo, clip_hi)
    n = M.shape[0]
    lo, hi = (clip_lo, clip_hi) if clip_lo <= clip_hi else (clip_hi, clip_lo)

    # Per-row class labels ————————————————————————————————————————————
    # centroids mode: the id IS the class label already.
    # vectors mode:   look up device_id from plot_df so hover + tick
    #                 labels show the class rather than a raw session id.
    if mode == "centroids":
        cls_labels = list(ids)
    else:
        cls_labels = _class_labels_from_plot_df(plot_df, ids)

    show_ticks = n <= _MAX_TICK_LABELS
    tick_text = cls_labels if show_ticks else []

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=M,
        x=list(range(n)), y=list(range(n)),
        colorscale="Viridis",
        zmin=float(lo), zmax=float(hi), zauto=False,
        customdata=_build_customdata(ids, cls_labels) if n > 0 else [],
        hovertemplate=(
            "<b>row:</b> %{customdata[0]}"
            " <span style='color:#aaa'>(%{customdata[2]})</span><br>"
            "<b>col:</b> %{customdata[1]}"
            " <span style='color:#aaa'>(%{customdata[3]})</span><br>"
            "<b>cosine:</b> %{z:.4f}<extra></extra>"
        ),
        hoverongaps=False,
        colorbar=dict(thickness=10, len=0.7),
        name=mode,
    ))

    title = (f"cosine matrix ({mode}, n={n}) — "
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
        xaxis=dict(**axis_common, title="predicted →"),
        yaxis=dict(**axis_common, title="← true",
                   autorange="reversed", scaleanchor="x"),
        meta={"mode": mode, "ids": ids, "cls_labels": cls_labels},
    )
    return fig
