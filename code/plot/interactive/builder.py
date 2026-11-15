"""build_interactive — composes ROC/dist/PCA/CM as a live FigureWidget dashboard.

Per-class sliders mutate ONLY their threshold-dependent traces (op-point
marker, dist dash, PCA decision-ring). Pair sliders mutate one op-point
each. Aggregator (mean↔median) refit is the only path that wholesale
rebuilds PCA traces — every other interaction is in-place.

State lives on a ``SimpleNamespace`` (``s``) so per-slider/rebuild
handlers stay top-level functions in ``_handlers.py`` rather than
closures. Widget construction lives in ``_widgets.py``; the CM figure
controller lives in ``_cm_panel.py``; ring-data precompute lives in
``_ring.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import plotly.graph_objects as go

from ..roc import build_roc_fig
from ..dist import build_dist_fig
from ..pca import build_pca_fig
from ._utils import opts_from_scores, rank01
from ._nudge import NUDGE_CSS_HTML, make_nudged_row
from ._snapshot import make_download_widget
from ._cm_panel import CmPanel, make_cm_controls_box
from ._ring import precompute_ring_data
from ._widgets import (
    make_class_sliders, make_alpha_slider, make_temp_slider,
    make_pair_sliders, make_aggregator_toggle,
)
from . import _handlers as H


__all__ = ["build_interactive"]


def build_interactive(*, rep, V_te, dev_te, devices_sorted, cmap,
                      method_label: str, model_name: str, y_pair, P_te, n_tr: int,
                      class_labels, use_raw: bool,
                      init_thresholds: dict | None = None,
                      pca_inputs=None,
                      V_tr=None, dev_tr=None):
    import ipywidgets as widgets

    init = dict(init_thresholds or {})
    class_labels_list = list(class_labels)
    ri_pair, ci_pair = np.triu_indices(len(V_te), k=1)
    _, _, P_live = rep.predict(V_te)

    # ── widgets ──────────────────────────────────────────────────────
    sliders = make_class_sliders(class_labels, P_live, class_labels_list,
                                 rep, init)
    alpha_slider, alpha_note, impl_for_alpha = make_alpha_slider(rep)
    temp_slider, temp_note, has_temp = make_temp_slider(rep)
    pair_cos_slider, pair_cos_scores, pair_norm_slider, pair_norm_scores = \
        make_pair_sliders(rep, V_te, ri_pair, ci_pair, init)

    # ── initial figures ──────────────────────────────────────────────
    def get_pca_inputs():
        return pca_inputs() if callable(pca_inputs) else pca_inputs

    initial_pca = get_pca_inputs()
    f1 = go.FigureWidget(build_roc_fig(
        rep=rep, V_te=V_te, dev_te=dev_te, devices_sorted=devices_sorted, cmap=cmap,
        method_label=method_label, model_name=model_name, y_pair=y_pair,
        P_te=P_live, n_tr=n_tr, class_labels=class_labels,
        class_thresholds={c: float(sl.value) for c, sl in sliders.items()},
        pair_cos_thr=float(pair_cos_slider.value),
        pair_norm_thr=float(pair_norm_slider.value),
    ))
    f2 = go.FigureWidget(build_dist_fig(
        rep=rep, V_te=V_te, dev_te=dev_te, devices_sorted=devices_sorted, cmap=cmap,
        method_label=method_label, model_name=model_name, use_raw=use_raw,
        class_thresholds={c: float(sl.value) for c, sl in sliders.items()},
        class_labels=class_labels,
    ))

    f3 = None
    pca_ring_idxs: dict[object, int] = {}
    ring_dist: dict[object, np.ndarray] = {}
    ring_score: dict[object, np.ndarray] = {}
    ring_same_class: dict[object, np.ndarray] = {}
    theta = np.linspace(0.0, 2.0 * np.pi, 96)

    if initial_pca is not None:
        f3 = go.FigureWidget(build_pca_fig(
            rep=rep, **initial_pca, cmap=cmap,
            method_label=method_label, model_name=model_name,
            class_thresholds={c: float(sl.value) for c, sl in sliders.items()},
        ))
        for idx, tr in enumerate(f3.data):
            lg = getattr(tr, "legendgroup", None)
            meta = getattr(tr, "meta", None)
            if meta is not None and len(meta) >= 3 and lg == "threshold_ring":
                pca_ring_idxs[meta[2]] = idx
        precompute_ring_data(
            f3=f3, initial_pca=initial_pca,
            devices_sorted=devices_sorted, class_labels_list=class_labels_list,
            pca_ring_idxs=pca_ring_idxs, ring_dist=ring_dist,
            ring_score=ring_score, ring_same_class=ring_same_class, rep=rep,
        )

    cm = (CmPanel(f3=f3, initial_pca_ref=initial_pca or {},
                  devices_sorted=devices_sorted,
                  method_label=method_label, model_name=model_name)
          if initial_pca is not None else None)

    # ── trace-index lookups ──────────────────────────────────────────
    op_point_idxs: dict[object, int] = {}
    pair_op_idxs: dict[str, int] = {}
    norm_thr_idxs: list[int] = []

    for idx, tr in enumerate(f1.data):
        lg = getattr(tr, "legendgroup", None)
        if getattr(tr, "mode", "") != "markers":
            continue
        if lg in class_labels_list:
            op_point_idxs[lg] = idx
        elif lg in ("pair-cos", "pair-norm"):
            pair_op_idxs[lg] = idx
    for i, tr in enumerate(f2.data):
        if getattr(tr, "legendgroup", None) == "threshold":
            norm_thr_idxs.append(i)

    # ── shared state namespace ───────────────────────────────────────
    s = SimpleNamespace(
        rep=rep, V_te=V_te, dev_te=dev_te, devices_sorted=devices_sorted,
        cmap=cmap, method_label=method_label, model_name=model_name,
        y_pair=y_pair, n_tr=n_tr, class_labels=class_labels,
        class_labels_list=class_labels_list, use_raw=use_raw,
        sliders=sliders, per_dev_cache={},
        alpha_slider=alpha_slider, temp_slider=temp_slider,
        impl_for_alpha=impl_for_alpha, has_temp=has_temp,
        pair_cos_slider=pair_cos_slider, pair_cos_scores=pair_cos_scores,
        pair_norm_slider=pair_norm_slider, pair_norm_scores=pair_norm_scores,
        ri_pair=ri_pair, ci_pair=ci_pair,
        f1=f1, f2=f2, f3=f3, cm=cm,
        op_point_idxs=op_point_idxs, pair_op_idxs=pair_op_idxs,
        norm_thr_idxs=norm_thr_idxs, pca_ring_idxs=pca_ring_idxs,
        ring_dist=ring_dist, ring_score=ring_score,
        ring_same_class=ring_same_class,
        cos96=np.cos(theta), sin96=np.sin(theta),
        P_live=P_live, initial_pca=initial_pca, pca_inputs=pca_inputs,
        get_pca_inputs=get_pca_inputs,
    )
    H.refresh_cache(s, P_live)
    thr0 = H.current_thr(s)

    # ── observers ────────────────────────────────────────────────────
    for c, sl in sliders.items():
        sl.observe(
            lambda change, _c=c: H.update_class(s, _c, float(change["new"])),
            names="value",
        )
    pair_cos_slider.observe(
        lambda change: H.update_pair_cos(s, float(change["new"])), names="value",
    )
    pair_norm_slider.observe(
        lambda change: H.update_pair_norm(s, float(change["new"])), names="value",
    )
    temp_slider.observe(lambda _: H.rebuild_posterior_dependent(s), names="value")
    alpha_slider.observe(lambda c: H.rebuild_after_alpha(s, c), names="value")

    has_aggregator = bool(getattr(rep, "_uses_aggregator", False))
    can_refit = has_aggregator and V_tr is not None and dev_tr is not None
    agg_toggle, agg_note = make_aggregator_toggle(rep, can_refit)

    def on_agg(change):
        if not can_refit:
            return
        rep.fit(V_tr, dev_tr, aggregator=change["new"])
        H.rebuild_after_refit(s)

    agg_toggle.observe(on_agg, names="value")
    H.apply_thresholds(s, thr0)

    # ── headers + download ───────────────────────────────────────────
    header = widgets.HTML(
        f"<b>per-class thresholds</b> — "
        f"<span style='color:#aaa'>"
        f"domain: {'raw' if use_raw else 'posterior'} · "
        f"options = unique rep.normalize_score(predict) per class · "
        f"drag or ◀/▶ to move the op-point along the ROC steps"
        f"</span>"
    )
    pair_header = widgets.HTML(
        "<b>pair-level thresholds</b> — "
        "<span style='color:#aaa'>"
        "op-point markers on pair·cosine and pair·normalized ROC curves"
        "</span>"
    )
    download_btn, download_out = make_download_widget(
        rep=rep, method_label=method_label, model_name=model_name,
        sliders=sliders, per_dev_cache=s.per_dev_cache,
        pair_cos_slider=pair_cos_slider, pair_norm_slider=pair_norm_slider,
        y_pair=y_pair, pair_cos_scores=pair_cos_scores,
        pair_norm_scores=pair_norm_scores,
    )

    rows = [
        NUDGE_CSS_HTML(),
        widgets.VBox([agg_toggle, agg_note]),
        header,
        widgets.VBox([make_nudged_row(sliders[c]) for c in class_labels]),
        pair_header,
        widgets.VBox([make_nudged_row(pair_cos_slider),
                      make_nudged_row(pair_norm_slider)]),
        widgets.VBox([make_nudged_row(temp_slider), temp_note]),
        widgets.VBox([make_nudged_row(alpha_slider), alpha_note]),
        widgets.HBox([download_btn, download_out]),
        f1, f2,
    ]
    if f3 is not None:
        rows.append(f3)
    if cm is not None and cm.f4 is not None:
        rows.append(make_cm_controls_box(cm))
        rows.append(cm.f4)
    return widgets.VBox(rows)
