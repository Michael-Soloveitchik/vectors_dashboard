"""PCA panel adapter (split from figures.py)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from plot.pca import build_pca_fig

from .figures_common import (palette_for, _centroids, _y_from_label_keys,
                              _make_reducer)


def _override_pca_rings(fig, V_full, rep, labels, centroid_rows,
                        thr_norm_dict: dict) -> None:
    """Replace toy ring traces (`legendgroup="threshold_ring"`) with
    data-driven radii. r_class = max(d_to_centroid where norm_score ≥
    thr_norm). Covers all points at thr=0; collapses to 0 at thr=1.
    Uses ONLY rep API (predict + normalize_score)."""
    from sklearn.decomposition import PCA
    if V_full is None or len(centroid_rows) == 0:
        return
    joint = np.concatenate([V_full, *centroid_rows], axis=0)
    Z = PCA(n_components=2, random_state=42).fit_transform(joint)
    Z_s = Z[:V_full.shape[0]]
    Z_c = Z[V_full.shape[0]:]
    try:
        _, _, votes = rep.predict(V_full)
    except Exception as exc:
        print(f"[ring-override] predict failed: {exc}", flush=True)
        return
    cur_alpha = (getattr(rep.impl, "NORM_ALPHA", "?")
                 if hasattr(rep, "impl") else "?")
    print(f"[ring-override] votes shape={votes.shape} alpha={cur_alpha} "
          f"thr_dict={thr_norm_dict}", flush=True)
    ordered = list(rep.class_order_fn())
    cidx = {str(lbl): i for i, lbl in enumerate(labels)}
    new_traces = []
    theta = np.linspace(0, 2 * np.pi, 96)
    if V_full.shape[0] > 0:
        d_global_max = float(np.linalg.norm(
            Z_s - Z_s.mean(axis=0, keepdims=True), axis=1).max())
        ring_cap = d_global_max * 1.05
    else:
        ring_cap = 1.0
    for tr in fig.data:
        if (getattr(tr, "legendgroup", "") or "") != "threshold_ring":
            new_traces.append(tr)
            continue
        meta = getattr(tr, "meta", None) or []
        if len(meta) < 3:
            new_traces.append(tr)
            continue
        cx, cy, lbl_str = float(meta[0]), float(meta[1]), str(meta[2])
        if lbl_str not in [str(o) for o in ordered]:
            new_traces.append(tr)
            continue
        k = [str(o) for o in ordered].index(lbl_str)
        thr_norm = float(thr_norm_dict.get(lbl_str, 0.0))
        try:
            norm = rep.normalize_score(votes[:, k])
        except Exception:
            new_traces.append(tr)
            continue
        if thr_norm >= 1.0 - 1e-9:
            r = 0.0
        elif thr_norm <= 0.0:
            d = np.hypot(Z_s[:, 0] - cx, Z_s[:, 1] - cy)
            r = float(d.max()) if d.size else 0.0
        else:
            accepted = np.asarray(norm) >= thr_norm
            if accepted.any():
                d = np.hypot(Z_s[:, 0] - cx, Z_s[:, 1] - cy)
                r = float(d[accepted].max())
            else:
                r = 0.0
        r = min(r, ring_cap)
        rx = cx + r * np.cos(theta)
        ry = cy + r * np.sin(theta)
        tr.x = rx.tolist()
        tr.y = ry.tolist()
        new_traces.append(tr)
    fig.data = tuple(new_traces)


def pca_fig(*, rep, df_m, V, sids, state, method_label, model_name,
            class_thresholds, label_keys=None,
            show_ellipses: bool = True, n_sigma: float = 2.0,
            agg: list[str] | None = None,
            reduction: str = "PCA",
            reduction_params: dict | None = None,
            density: bool = False,
            excluded_sids: list[str] | None = None):
    """Toy `build_pca_fig` already draws the cosine-τ dashed ring per
    centroid. `_override_pca_rings` replaces those rings with
    data-driven radii in normalized score space."""
    del show_ellipses, n_sigma   # toy owns the ring geometry
    rows, labels = _centroids(state)
    # Bug D: rescale centroids to median V norm so they share the same
    # shell as data points in joint-PCA.  Many `_centroid_under_score`
    # implementations return raw arithmetic mean (norm < 1) which lands
    # centroids in the cloud's interior, far from clusters.
    if rows and V.shape[0] > 0:
        v_med = float(np.median(np.linalg.norm(V, axis=1)))
        if np.isfinite(v_med) and v_med > 0:
            rescaled = []
            for r in rows:
                arr = np.atleast_2d(r)
                n = np.linalg.norm(arr, axis=1, keepdims=True)
                n = np.maximum(n, 1e-12)
                rescaled.append(arr * (v_med / n))
            rows = rescaled
    y = _y_from_label_keys(df_m, label_keys)
    cmap = palette_for(sorted(set(y)))
    plot_df = df_m.copy()
    plot_df["device_id"] = y
    if "voice_id" not in plot_df.columns and "speaker_id" in plot_df.columns:
        plot_df["voice_id"] = plot_df["speaker_id"]
    runner = _make_reducer(reduction, reduction_params or {})
    is_pca = (reduction or "PCA").strip().lower() == "pca"
    rings_thr: dict = {}
    fig = build_pca_fig(
        rep=rep, V_emb=V, centroid_rows=rows, centroid_labels=labels,
        plot_df=plot_df, cmap=cmap, method_label=method_label,
        model_name=model_name, run_pca=runner,
        class_thresholds=rings_thr,
    )
    fig.update_layout(title_text=f"{reduction} — {method_label} — {model_name}")
    if not is_pca:
        fig.data = tuple(
            t for t in fig.data
            if (getattr(t, "legendgroup", "") or "") != "threshold_ring"
        )
    elif class_thresholds:
        _override_pca_rings(fig, V, rep, labels, rows,
                            class_thresholds or {})
    if density:
        _add_density_contour(fig)
    _apply_pca_interaction(fig, sids, agg,
                           f"{model_name}-{reduction}", method_label,
                           excluded_sids=excluded_sids)
    fig.update_layout(autosize=True, width=None, height=None,
                      margin=dict(t=40, l=10, r=10, b=10))
    return fig


def _add_density_contour(fig) -> None:
    xs, ys = [], []
    for tr in fig.data:
        if getattr(tr, "mode", None) != "markers":
            continue
        if tr.customdata is None:
            continue
        if len(tr.customdata) and len(tr.customdata[0]) >= 2:
            xs.extend(list(tr.x))
            ys.extend(list(tr.y))
    if len(xs) < 5:
        return
    contour = go.Histogram2dContour(
        x=xs, y=ys, contours=dict(coloring="lines"),
        line=dict(width=1, color="rgba(255,215,0,0.4)"),
        showscale=False, hoverinfo="skip",
        name="density",
    )
    fig.data = (contour,) + tuple(t for t in fig.data if t is not contour)


def _apply_pca_interaction(fig, sids, agg, model_name, method_label,
                           excluded_sids=None):
    rev = f"pca-{model_name}-{method_label}-{len(sids)}"
    fig.update_layout(
        uirevision=rev,
        dragmode="select",
        selections=[],
        clickmode="event+select",
    )
    agg_set = {str(s) for s in (agg or [])}
    excl_set = {str(s) for s in (excluded_sids or [])}
    for tr in fig.data:
        if not isinstance(tr, go.Scatter) or tr.mode != "markers":
            continue
        cd = tr.customdata
        if cd is None or len(cd) == 0:
            continue
        try:
            row0_len = len(cd[0])
        except TypeError:
            continue
        if row0_len < 2:
            continue
        n = len(cd)
        sids_in_trace = [str(row[1]) for row in cd]
        if not agg_set:
            tr.selectedpoints = list(range(n))
        else:
            tr.selectedpoints = [
                i for i, sid in enumerate(sids_in_trace) if sid in agg_set
            ]
        tr.selected = dict(marker=dict(opacity=1.0))
        tr.unselected = dict(marker=dict(opacity=0.15))
        if excl_set:
            tr.marker.opacity = [0.25 if sid in excl_set else 0.85
                                 for sid in sids_in_trace]


def add_cm_highlight(fig, *, df_m, V, sids, state, label_keys,
                     row_id: str, col_id: str):
    """Project (row_id, col_id) into PCA space + add a highlight overlay."""
    from sklearn.decomposition import PCA
    rows, labels = _centroids(state)
    sid_to_idx = {str(s): i for i, s in enumerate(sids)}
    centroid_to_idx = {str(lbl): j for j, lbl in enumerate(labels)}
    joint = (np.concatenate([V, *rows], axis=0)
             if rows else V).astype(np.float64)
    Z = PCA(n_components=2, random_state=42).fit_transform(joint)
    n_sample = V.shape[0]

    def _resolve(_id):
        if _id in sid_to_idx:
            return Z[sid_to_idx[_id]]
        if _id in centroid_to_idx:
            return Z[n_sample + centroid_to_idx[_id]]
        return None

    pts = []
    for _id in (row_id, col_id):
        z = _resolve(_id)
        if z is not None:
            pts.append((float(z[0]), float(z[1]), _id))
    if not pts:
        return fig
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pts], y=[p[1] for p in pts],
        mode="markers+text",
        text=[p[2] for p in pts],
        textposition="bottom center",
        marker=dict(size=22, color="rgba(255,215,0,0.0)",
                    line=dict(color="#ffd700", width=3)),
        name="cm-click",
        hovertemplate="<b>%{text}</b><br>cm-click<extra></extra>",
        showlegend=True,
    ))
    return fig
