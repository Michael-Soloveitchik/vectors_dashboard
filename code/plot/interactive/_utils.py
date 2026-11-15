"""Pure helpers for the interactive builder — no widget state."""
from __future__ import annotations

import numpy as np


__all__ = [
    "opts_from_scores",
    "class_opts",
    "pick_init",
    "rebuild_slider_options",
    "rank01",
    "copy_traces_in_place",
]


def opts_from_scores(s: np.ndarray) -> list[tuple[str, float]]:
    """Unique sorted thresholds as ``(label, value)`` pairs, rounded to 2dp.

    Round BEFORE unique so duplicates collapse — slider readout, op-point
    label, and threshold dash all match exactly. Always include 0.0 so the
    canonical init position used by the toy example is reachable even when
    no real score rounds to exactly 0.
    """
    a = np.round(np.asarray(s, dtype=np.float64), 2)
    a = np.concatenate([a, np.array([0.0], dtype=np.float64)])
    u = np.unique(a)
    if u.size == 0:
        return [("0.00", 0.0)]
    return [(f"{v:.2f}", float(v)) for v in u]


def class_opts(P_mat: np.ndarray, lbl, class_labels_list: list,
               rep) -> list[tuple[str, float]]:
    """Per-class slider options. VALUE stays in RAW; LABEL shown is the
    NORMALIZED score via ``rep.normalize_score``. Dedup by rounded
    normalized label. Always include the (raw=0, norm=normalize_score(0))
    point so sliders can snap to the toy-init position.
    """
    ci = class_labels_list.index(lbl)
    col = np.asarray(P_mat[:, ci], dtype=np.float64)
    if col.size == 0:
        return [("0.00", 0.0)]
    raws = np.unique(np.concatenate([col, np.array([0.0])]))
    norms = rep.normalize_score(raws)
    seen: dict[str, float] = {}
    for r, n in zip(raws, norms):
        label = f"{float(n):.2f}"
        if label not in seen:
            seen[label] = float(r)
    return list(seen.items())


def pick_init(opts: list[tuple[str, float]], want: float | None) -> float:
    """Pick a slider value from ``opts``. When no preference is given, pick the
    option closest to 0.0 (toy convention). Previously fell back to the median
    of options which placed the dash and op-point at ~0.2-0.5 instead of 0.
    """
    vals = [v for _, v in opts]
    if not vals:
        return 0.0
    target = 0.0 if want is None else float(want)
    return min(vals, key=lambda v: abs(v - target))


def rebuild_slider_options(sl, opts):
    """Swap a SelectionSlider's options without dropping the live value.

    ``hold_trait_notifications`` batches value/options changes so only the
    final value-change observer fires (once) when the block exits."""
    if not opts:
        return
    cur = float(sl.value) if sl.value is not None else None
    new_val = pick_init(opts, cur)
    with sl.hold_trait_notifications():
        sl.options = opts
        sl.value = new_val


def rank01(v: np.ndarray) -> np.ndarray:
    """Return ``v`` mapped to ``[0, 1]`` by average rank.  Empty → zeros."""
    from scipy.stats import rankdata
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size <= 1:
        return np.zeros_like(a)
    return (rankdata(a, method="average") - 1.0) / (a.size - 1.0)


def copy_traces_in_place(fw, new_fig):
    """Mirror ``new_fig.data`` into ``fw`` without rebuilding the widget.

    Trace counts may differ after a structural rebuild (e.g. when a class
    appears/disappears) — fall back to wholesale replace in that case."""
    if len(fw.data) != len(new_fig.data):
        with fw.batch_update():
            fw.data = ()
            fw.add_traces(list(new_fig.data))
            fw.layout = new_fig.layout
        return
    with fw.batch_update():
        for i, new_tr in enumerate(new_fig.data):
            old_tr = fw.data[i]
            if getattr(new_tr, "x", None) is not None:
                old_tr.x = new_tr.x
            if getattr(new_tr, "y", None) is not None:
                old_tr.y = new_tr.y
            if getattr(new_tr, "text", None) is not None:
                old_tr.text = new_tr.text
            if getattr(new_tr, "customdata", None) is not None:
                old_tr.customdata = new_tr.customdata
            if getattr(new_tr, "name", None) is not None:
                old_tr.name = new_tr.name
