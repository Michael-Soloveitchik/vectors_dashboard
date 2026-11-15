"""Per-class PCA-ring distance/score precompute.

Reads the live PCA figure to recover sample 2-D positions, then scores
every sample against every class so the per-class ring radius can be
computed from a simple ``score >= threshold`` mask.

Stays a free function (not a class) — the builder owns its three
dictionaries, this just fills them in place.
"""
from __future__ import annotations

import numpy as np


__all__ = ["precompute_ring_data"]


def precompute_ring_data(
    *, f3, initial_pca: dict | None, devices_sorted, class_labels_list,
    pca_ring_idxs: dict, ring_dist: dict, ring_score: dict,
    ring_same_class: dict, rep,
) -> None:
    """Fill ``ring_*`` dicts with per-class arrays read off the PCA panel.

    1. Collect xy positions of every sample-marker trace, grouped by
       device, then stitch back together in plot-df row order.
    2. Score all samples against every class via ``rep.predict``; stash
       per-class (distance, score, same-class mask) tuples keyed by the
       class label.
    """
    if f3 is None or initial_pca is None:
        return
    ring_dist.clear()
    ring_score.clear()
    ring_same_class.clear()

    V_emb_all = initial_pca["V_emb"]
    plot_dev = initial_pca["plot_df"]["device_id"].to_numpy()
    xy_by_dev: dict[object, np.ndarray] = {}
    for tr in f3.data:
        if (tr.mode == "markers"
                and tr.name in devices_sorted
                and getattr(tr, "legendgroup", None) != "threshold_ring"):
            xy_by_dev[tr.name] = np.column_stack([
                np.asarray(tr.x, dtype=np.float64),
                np.asarray(tr.y, dtype=np.float64),
            ])
    xy_all = np.zeros((plot_dev.size, 2), dtype=np.float64)
    for d in np.unique(plot_dev):
        xy_d = xy_by_dev.get(d)
        if xy_d is None:
            continue
        idx = np.flatnonzero(plot_dev == d)
        n = min(len(idx), xy_d.shape[0])
        xy_all[idx[:n]] = xy_d[:n]

    _, _, values = rep.predict(V_emb_all)
    labels_pred = list(rep.class_order_fn())
    class_idx_map = {lbl: i for i, lbl in enumerate(labels_pred)}
    for lbl in class_labels_list:
        ring_idx_trace = pca_ring_idxs.get(lbl)
        if ring_idx_trace is None or lbl not in class_idx_map:
            continue
        meta = f3.data[ring_idx_trace].meta
        cx, cy = float(meta[0]), float(meta[1])
        d_all = np.hypot(xy_all[:, 0] - cx, xy_all[:, 1] - cy)
        ring_dist[lbl] = d_all
        ring_score[lbl] = values[:, class_idx_map[lbl]].astype(np.float64)
        mask_dc = np.asarray(plot_dev) == lbl
        if not bool(mask_dc.any()):
            mask_dc = np.asarray([str(d) == str(lbl) for d in plot_dev],
                                 dtype=bool)
        ring_same_class[lbl] = mask_dc
