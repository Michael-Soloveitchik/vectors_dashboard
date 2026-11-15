"""Per-slider update + bulk rebuild handlers — operate on a state ns.

State (``s``) is a ``SimpleNamespace`` populated by ``builder.py`` with
the figure widgets, index dicts, score caches, and method-of-record
references. These functions are intentionally side-effecty: each one
writes back into the figures or caches it owns.
"""
from __future__ import annotations

import numpy as np

from ..scoring import _op_point
from ..roc import build_roc_fig
from ..dist import build_dist_fig
from ..pca import build_pca_fig
from ._utils import (
    rebuild_slider_options, opts_from_scores, rank01, copy_traces_in_place,
    class_opts,
)
from ._ring import precompute_ring_data


__all__ = [
    "refresh_cache", "current_thr",
    "update_class_op", "update_class_dist_line", "update_class_ring",
    "update_class", "update_pair_cos", "update_pair_norm",
    "apply_thresholds",
    "rebuild_posterior_dependent", "rebuild_after_refit", "rebuild_after_alpha",
]


# ── score cache ─────────────────────────────────────────────────────
def refresh_cache(s, P_mat):
    s.per_dev_cache.clear()
    for lbl in s.devices_sorted:
        if lbl not in s.class_labels_list:
            continue
        ci = s.class_labels_list.index(lbl)
        y_c = (s.dev_te == lbl).astype(np.int32)
        if y_c.sum() in (0, len(y_c)):
            continue
        s.per_dev_cache[lbl] = (y_c, P_mat[:, ci].astype(np.float64))
        sl = s.sliders.get(lbl)
        if sl is not None:
            rebuild_slider_options(
                sl, class_opts(P_mat, lbl, s.class_labels_list, s.rep)
            )


def current_thr(s) -> dict:
    return {c: float(sl.value) for c, sl in s.sliders.items()}


# ── per-class targeted updates ──────────────────────────────────────
def update_class_op(s, lbl, v):
    idx = s.op_point_idxs.get(lbl)
    cached = s.per_dev_cache.get(lbl)
    if idx is None or cached is None:
        return
    y_c, raw_scores = cached
    # Init / neutral position: slider at 0 → op-point at (0, 0). Matches the
    # toy-example convention so PCA ring + dist dash + ROC op-point all line
    # up at "0" before the user has moved the slider. After any non-zero
    # move, compute the real (FPR, TPR) at the rank-mapped threshold.
    if float(v) == 0.0:
        op_fpr, op_tpr = 0.0, 0.0
    else:
        norm_scores = s.rep.normalize_score(raw_scores)
        j = int(np.argmin(np.abs(raw_scores - float(v))))
        norm_v = float(norm_scores[j])
        op_fpr, op_tpr = _op_point(y_c, norm_scores, norm_v)
    tr = s.f1.data[idx]
    with s.f1.batch_update():
        tr.x = [op_fpr]
        tr.y = [op_tpr]
        tr.name = f"{lbl} @ thr={v:.2f}"
        tr.customdata = [float(v)]


def update_class_dist_line(s, lbl, v):
    if lbl not in s.class_labels_list:
        return
    i = s.class_labels_list.index(lbl)
    cached = s.per_dev_cache.get(lbl)
    if cached is None:
        return
    # Plot dash directly at the slider value clipped to the [0, 1] x-range.
    # Previously the rank-map nearest-raw lookup placed v=0 at the median
    # rank (~0.2-0.5 depending on class), which made the init "0" position
    # look like 0.2.
    norm_thr = float(np.clip(float(v), 0.0, 1.0))
    if i >= len(s.norm_thr_idxs):
        return
    tr = s.f2.data[s.norm_thr_idxs[i]]
    with s.f2.batch_update():
        tr.x = [norm_thr, norm_thr]


def update_class_ring(s, lbl, v):
    if s.f3 is None:
        return
    idx = s.pca_ring_idxs.get(lbl)
    if idx is None:
        return
    meta = s.f3.data[idx].meta
    cx, cy = float(meta[0]), float(meta[1])
    d = s.ring_dist.get(lbl)
    sc = s.ring_score.get(lbl)
    if d is None or sc is None or d.size == 0:
        r = 0.0
    else:
        cached = s.per_dev_cache.get(lbl)
        if cached is None:
            accepted = sc >= float(v)
        else:
            _, raw_col = cached
            j = int(np.argmin(np.abs(raw_col - float(v))))
            norm_v = float(s.rep.normalize_score(raw_col)[j])
            norm_s = s.rep.normalize_score(sc)
            accepted = norm_s >= norm_v
        r = float(d[accepted].max()) if accepted.any() else 0.0
    rx = cx + r * s.cos96
    ry = cy + r * s.sin96
    with s.f3.batch_update():
        s.f3.data[idx].x = rx
        s.f3.data[idx].y = ry


def update_class(s, lbl, v):
    update_class_op(s, lbl, v)
    update_class_dist_line(s, lbl, v)
    update_class_ring(s, lbl, v)


def update_pair_cos(s, v):
    idx = s.pair_op_idxs.get("pair-cos")
    if idx is None:
        return
    if float(v) == 0.0:
        fpr, tpr = 0.0, 0.0
    else:
        fpr, tpr = _op_point(s.y_pair, s.pair_cos_scores, float(v))
    tr = s.f1.data[idx]
    with s.f1.batch_update():
        tr.x = [fpr]
        tr.y = [tpr]
        tr.name = f"pair-cos @ thr={v:.2f}"
        tr.customdata = [float(v)]


def update_pair_norm(s, v):
    idx = s.pair_op_idxs.get("pair-norm")
    if idx is None:
        return
    if float(v) == 0.0:
        fpr, tpr = 0.0, 0.0
    else:
        fpr, tpr = _op_point(s.y_pair, s.pair_norm_scores[0], float(v))
    tr = s.f1.data[idx]
    with s.f1.batch_update():
        tr.x = [fpr]
        tr.y = [tpr]
        tr.name = f"pair-norm @ thr={v:.2f}"
        tr.customdata = [float(v)]


def apply_thresholds(s, thr_dict):
    for lbl in s.class_labels_list:
        v = thr_dict.get(lbl)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            update_class(s, lbl, float(v))
    update_pair_cos(s, float(s.pair_cos_slider.value))
    update_pair_norm(s, float(s.pair_norm_slider.value))


# ── bulk rebuilds ───────────────────────────────────────────────────
def _recompute_pair_norm(s):
    raw2 = np.asarray(s.rep.predict(s.V_te)[2], dtype=np.float64)
    P_norm = np.column_stack([s.rep.normalize_score(raw2[:, k])
                              for k in range(raw2.shape[1])])
    s.pair_norm_scores[0] = rank01((P_norm @ P_norm.T)[s.ri_pair, s.ci_pair])
    rebuild_slider_options(s.pair_norm_slider,
                           opts_from_scores(s.pair_norm_scores[0]))


def _rebuild_top_two(s, P_for_roc):
    thr = current_thr(s)
    new_f1 = build_roc_fig(
        rep=s.rep, V_te=s.V_te, dev_te=s.dev_te,
        devices_sorted=s.devices_sorted, cmap=s.cmap,
        method_label=s.method_label, model_name=s.model_name, y_pair=s.y_pair,
        P_te=P_for_roc, n_tr=s.n_tr, class_labels=s.class_labels,
        class_thresholds=thr,
        pair_cos_thr=float(s.pair_cos_slider.value),
        pair_norm_thr=float(s.pair_norm_slider.value),
    )
    new_f2 = build_dist_fig(
        rep=s.rep, V_te=s.V_te, dev_te=s.dev_te,
        devices_sorted=s.devices_sorted, cmap=s.cmap,
        method_label=s.method_label, model_name=s.model_name,
        use_raw=s.use_raw, class_thresholds=thr, class_labels=s.class_labels,
    )
    copy_traces_in_place(s.f1, new_f1)
    copy_traces_in_place(s.f2, new_f2)
    return thr


def rebuild_posterior_dependent(s):
    if s.has_temp:
        s.rep.temperature = float(s.temp_slider.value)
    _, _, P_new = s.rep.predict(s.V_te)
    refresh_cache(s, P_new)
    _recompute_pair_norm(s)
    thr = _rebuild_top_two(s, P_new)
    apply_thresholds(s, thr)


def rebuild_after_refit(s):
    _, _, P_new = s.rep.predict(s.V_te)
    refresh_cache(s, P_new)
    _recompute_pair_norm(s)
    thr = _rebuild_top_two(s, P_new)
    if s.f3 is not None and s.pca_inputs is not None:
        fresh_pca = s.get_pca_inputs()
        s.initial_pca.update(fresh_pca)
        new_f3 = build_pca_fig(
            rep=s.rep, **fresh_pca, cmap=s.cmap,
            method_label=s.method_label, model_name=s.model_name,
            class_thresholds=thr,
        )
        copy_traces_in_place(s.f3, new_f3)
        for idx, tr in enumerate(s.f3.data):
            lg = getattr(tr, "legendgroup", None)
            m = getattr(tr, "meta", None)
            if m is None or len(m) < 3:
                continue
            if lg == "threshold_ring":
                s.pca_ring_idxs[m[2]] = idx
        precompute_ring_data(
            f3=s.f3, initial_pca=s.initial_pca,
            devices_sorted=s.devices_sorted,
            class_labels_list=s.class_labels_list,
            pca_ring_idxs=s.pca_ring_idxs, ring_dist=s.ring_dist,
            ring_score=s.ring_score, ring_same_class=s.ring_same_class,
            rep=s.rep,
        )
        if s.cm is not None:
            s.cm.populate_xy_cache()
            s.cm.ensure_highlight_trace()
            s.cm.rebuild()
    apply_thresholds(s, thr)


def rebuild_after_alpha(s, _change=None):
    s.impl_for_alpha.NORM_ALPHA = float(s.alpha_slider.value)
    refresh_cache(s, s.P_live)
    _recompute_pair_norm(s)
    thr = _rebuild_top_two(s, s.P_live)
    apply_thresholds(s, thr)
