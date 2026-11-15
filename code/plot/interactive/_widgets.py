"""Widget factories — sliders, toggles, and HTML notes."""
from __future__ import annotations

import numpy as np

from ._utils import opts_from_scores, class_opts, pick_init, rank01


__all__ = [
    "make_class_sliders",
    "make_alpha_slider",
    "make_temp_slider",
    "make_pair_sliders",
    "make_aggregator_toggle",
]


def make_class_sliders(class_labels, P_live, class_labels_list, rep, init):
    import ipywidgets as widgets
    sliders: dict[object, widgets.SelectionSlider] = {}
    for c in class_labels:
        opts = class_opts(P_live, c, class_labels_list, rep)
        sliders[c] = widgets.SelectionSlider(
            options=opts,
            value=pick_init(opts, init.get(c)),
            description=str(c),
            continuous_update=False,
            layout=widgets.Layout(width="340px"),
            style={"description_width": "90px"},
            readout=True,
        )
    return sliders


def make_alpha_slider(rep):
    import ipywidgets as widgets
    impl = getattr(rep, "impl", rep)
    init = float(getattr(impl, "NORM_ALPHA", 0.5))
    sl = widgets.FloatSlider(
        value=init, min=0.0, max=1.0, step=0.05,
        description="α (norm blend)", readout_format=".2f",
        continuous_update=False,
        layout=widgets.Layout(width="340px"),
        style={"description_width": "120px"},
    )
    note = widgets.HTML(
        "<span style='color:#aaa;font-size:10px'>"
        "α=1 → pure ECDF (gaps collapse) · α=0 → pure min-max (raw shape) · "
        "0&lt;α&lt;1 → ECDF order + visible gaps. Per-class ROC unchanged "
        "(monotone); pair·normalized AUC may shift slightly."
        "</span>"
    )
    return sl, note, impl


def make_temp_slider(rep):
    import ipywidgets as widgets
    has_temp = hasattr(rep, "temperature")
    sl = widgets.FloatSlider(
        value=float(getattr(rep, "temperature", 0.1)),
        min=0.01, max=2.0, step=0.01,
        description="T (temperature)", readout_format=".3f",
        continuous_update=False,
        layout=widgets.Layout(width="340px"),
        style={"description_width": "120px"},
        disabled=not has_temp,
    )
    note = widgets.HTML(
        "<span style='color:#aaa;font-size:10px'>"
        + ("temperature active — lower = sharper posterior, higher = smoother"
           if has_temp else
           f"{type(rep).__name__} has no temperature; slider is silent")
        + "</span>"
    )
    return sl, note, has_temp


def make_pair_sliders(rep, V_te, ri_pair, ci_pair, init):
    """Build pair·cosine + pair·normalized sliders. Returns
    ``(pair_cos_slider, pair_cos_scores, pair_norm_slider, pair_norm_scores_list)``.
    pair_norm_scores_list is a 1-element list so callers can mutate it in place
    on rebuilds (T or α changes recompute the underlying normalized matrix)."""
    import ipywidgets as widgets

    pair_cos_scores = rep.cosine_pairs(V_te)
    pair_cos_opts = opts_from_scores(pair_cos_scores)
    pair_cos_slider = widgets.SelectionSlider(
        options=pair_cos_opts,
        value=pick_init(pair_cos_opts, init.get("__pair_cos__")),
        description="pair·cosine",
        continuous_update=False,
        layout=widgets.Layout(width="340px"),
        style={"description_width": "100px"},
        readout=True,
    )

    raw_predict = np.asarray(rep.predict(V_te)[2], dtype=np.float64)
    P_norm = np.column_stack([rep.normalize_score(raw_predict[:, k])
                              for k in range(raw_predict.shape[1])])
    pair_norm_scores = [rank01((P_norm @ P_norm.T)[ri_pair, ci_pair])]
    pair_norm_opts = opts_from_scores(pair_norm_scores[0])
    pair_norm_slider = widgets.SelectionSlider(
        options=pair_norm_opts,
        value=pick_init(pair_norm_opts, init.get("__pair_norm__")),
        description="pair·normalized",
        continuous_update=False,
        layout=widgets.Layout(width="340px"),
        style={"description_width": "100px"},
        readout=True,
    )
    return pair_cos_slider, pair_cos_scores, pair_norm_slider, pair_norm_scores


def make_aggregator_toggle(rep, can_refit: bool):
    import ipywidgets as widgets
    current = getattr(rep, "get_aggregator", lambda: None)() or "mean"
    tog = widgets.ToggleButtons(
        options=["mean", "median"],
        value=current if current in ("mean", "median") else "mean",
        description="centroid",
        tooltips=[
            "Fréchet mean of within-class scores (sphere: mean/||mean||)",
            "Spherical geometric median (Weiszfeld under the score)",
        ],
        disabled=not can_refit,
        layout=widgets.Layout(width="340px"),
        style={"description_width": "90px"},
    )
    note = widgets.HTML(
        "<span style='color:#aaa;font-size:10px'>"
        + ("refits the rep and rebuilds ROC/dist/PCA in place"
           if can_refit else
           "pass V_tr+dev_tr to enable mean↔median refit")
        + "</span>"
    )
    return tog, note
