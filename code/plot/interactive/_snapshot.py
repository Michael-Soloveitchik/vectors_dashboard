"""Download button — serializes the dashboard's operating state to JSON.

The snapshot captures everything needed to reproduce the current
position: centroids, thresholds, ROC op-points (FPR/TPR per class),
aggregator + temperature. The download fires via a synthetic ``<a>``
click so it works under remote/SSH kernels where ``FileLink`` paths
aren't reachable.
"""
from __future__ import annotations

import numpy as np

from ..scoring import _op_point


__all__ = ["make_download_widget"]


def _build_payload(
    *, rep, method_label, model_name, sliders, per_dev_cache,
    pair_cos_slider, pair_norm_slider, y_pair, pair_cos_scores,
    pair_norm_scores,
) -> dict:
    import datetime

    class_ops = {}
    for lbl, sl in sliders.items():
        cached = per_dev_cache.get(lbl)
        if cached is None:
            continue
        y_c, raw_col = cached
        v = float(sl.value)
        fpr, tpr = _op_point(y_c, raw_col, v)
        class_ops[str(lbl)] = {
            "raw_thr": v, "fpr": fpr, "tpr": tpr,
            "n_pos": int(y_c.sum()),
            "n_neg": int((1 - y_c).sum()),
        }

    pc_v = float(pair_cos_slider.value)
    pn_v = float(pair_norm_slider.value)
    pc_fpr, pc_tpr = _op_point(np.asarray(y_pair), pair_cos_scores, pc_v)
    pn_fpr, pn_tpr = _op_point(np.asarray(y_pair), pair_norm_scores[0], pn_v)

    cents: dict[str, list] = {}
    try:
        for lbl, arr in rep.get_centroids().items():
            cents[str(lbl)] = np.asarray(arr, dtype=np.float64).tolist()
    except Exception:
        pass

    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": method_label,
        "model": model_name,
        "aggregator": (rep.get_aggregator()
                       if hasattr(rep, "get_aggregator") else None),
        "temperature": (float(rep.temperature)
                        if hasattr(rep, "temperature") else None),
        "centroids": cents,
        "class_thresholds_raw": {str(c): float(s.value)
                                 for c, s in sliders.items()},
        "class_op_points": class_ops,
        "pair_cosine": {"thr": pc_v, "fpr": pc_fpr, "tpr": pc_tpr},
        "pair_normalized": {"thr": pn_v, "fpr": pn_fpr, "tpr": pn_tpr},
    }


def make_download_widget(*, rep, method_label, model_name, sliders,
                         per_dev_cache, pair_cos_slider, pair_norm_slider,
                         y_pair, pair_cos_scores, pair_norm_scores):
    """Return ``(button, output)`` pair wired to ``_on_download``."""
    import ipywidgets as widgets

    btn = widgets.Button(
        description="💾 download (centroids + thresholds + ROC op-points)",
        layout=widgets.Layout(width="420px"),
        button_style="info",
    )
    out = widgets.Output()

    def _on_download(_btn):
        import base64, json, datetime
        out.clear_output()
        payload = _build_payload(
            rep=rep, method_label=method_label, model_name=model_name,
            sliders=sliders, per_dev_cache=per_dev_cache,
            pair_cos_slider=pair_cos_slider, pair_norm_slider=pair_norm_slider,
            y_pair=y_pair, pair_cos_scores=pair_cos_scores,
            pair_norm_scores=pair_norm_scores,
        )
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"dashboard_snapshot_{method_label}_{model_name}_{ts}.json"
        body = json.dumps(payload, indent=2)
        b64 = base64.b64encode(body.encode("utf-8")).decode("ascii")
        js = (
            "(function(){"
            "var a=document.createElement('a');"
            f"a.href='data:application/json;base64,{b64}';"
            f"a.download={json.dumps(fname)};"
            "document.body.appendChild(a);"
            "a.click();"
            "a.remove();"
            "})();"
        )
        with out:
            from IPython.display import Javascript, HTML, display
            display(Javascript(js))
            display(HTML(f"<span style='color:#4ecdc4'>⬇ {fname}"
                         f" ({len(body)} bytes)</span>"))

    btn.on_click(_on_download)
    return btn, out
