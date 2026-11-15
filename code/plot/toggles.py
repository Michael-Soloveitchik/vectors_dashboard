"""Tight-CSS ipywidgets.Button toggles that replace Plotly updatemenus.

Plotly's native updatemenu buttons have a hard-coded minimum frame size
even with deeply-negative pad values. These ipywidget buttons are styled
via injected CSS to hug their text labels without any extra padding.
"""
from __future__ import annotations

from typing import Callable


__all__ = ["build_axis_toggles", "build_dist_toggles"]


_TIGHT_CSS = """
<style>
.nbp-tight-btn button {
    font-size: 9px !important;
    padding: 0 4px !important;
    min-height: 0 !important;
    line-height: 1.3 !important;
    border-radius: 3px !important;
    background-color: #f0f0f5 !important;
    color: #141419 !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
}
.nbp-tight-btn button:hover {
    background-color: #ffffff !important;
}
</style>
"""
_CSS_INJECTED = [False]


def _ensure_css():
    import ipywidgets as widgets
    from IPython.display import display
    if not _CSS_INJECTED[0]:
        display(widgets.HTML(_TIGHT_CSS))
        _CSS_INJECTED[0] = True


def _slide(label_a: str, label_b: str,
           on_a: Callable[[], None], on_b: Callable[[], None],
           start_at: str = "a"):
    """Two-state button; description shows the NEXT action.

    ``start_at='a'`` means clicking the button will first run ``on_a``
    (so the button currently shows label_a = the pending action).
    """
    import ipywidgets as widgets
    _ensure_css()
    btn = widgets.Button(
        description=(label_a if start_at == "a" else label_b),
        layout=widgets.Layout(
            width="auto", min_width="0", height="18px",
            padding="0px", margin="0px 4px 0px 0px",
        ),
    )
    state = {"side": start_at}

    def _cb(_):
        if state["side"] == "a":
            on_a()
            state["side"] = "b"
            btn.description = label_b
        else:
            on_b()
            state["side"] = "a"
            btn.description = label_a

    btn.on_click(_cb)
    btn.add_class("nbp-tight-btn")
    return btn


def build_axis_toggles(fig):
    """HBox([x → log, y → log]) wired to fig.layout.{x,y}axis.type/range."""
    import ipywidgets as widgets

    def _log_x():
        with fig.batch_update():
            fig.layout.xaxis.type = "log"
            fig.layout.xaxis.range = [-4, 0]

    def _lin_x():
        with fig.batch_update():
            fig.layout.xaxis.type = "linear"
            fig.layout.xaxis.range = [-0.02, 1.02]

    def _log_y():
        with fig.batch_update():
            fig.layout.yaxis.type = "log"
            fig.layout.yaxis.range = [-4, 0]

    def _lin_y():
        with fig.batch_update():
            fig.layout.yaxis.type = "linear"
            fig.layout.yaxis.range = [-0.02, 1.02]

    return widgets.HBox(
        [_slide("x → log", "x → lin", _log_x, _lin_x),
         _slide("y → log", "y → lin", _log_y, _lin_y)],
        layout=widgets.Layout(margin="0px 0px 2px 0px"),
    )


def build_dist_toggles(fig, use_raw: bool):
    """HBox([domain toggle, points toggle]) for the dist figure."""
    import ipywidgets as widgets

    meta = dict(fig.layout.meta or {})
    raw_vis  = list(meta.get("raw_vis", []))
    # Accept legacy "post_vis" key so already-built figures still work.
    norm_vis = list(meta.get("norm_vis", meta.get("post_vis", [])))

    def _to_norm():
        with fig.batch_update():
            for i, v in enumerate(norm_vis):
                if i < len(fig.data):
                    fig.data[i].visible = v
            fig.layout.yaxis.title.text = "normalized score"

    def _to_raw():
        with fig.batch_update():
            for i, v in enumerate(raw_vis):
                if i < len(fig.data):
                    fig.data[i].visible = v
            fig.layout.yaxis.title.text = "raw score"

    def _pts_off():
        with fig.batch_update():
            for tr in fig.data:
                if getattr(tr, "type", None) == "box":
                    tr.boxpoints = "outliers"

    def _pts_on():
        with fig.batch_update():
            for tr in fig.data:
                if getattr(tr, "type", None) == "box":
                    tr.boxpoints = "all"

    if use_raw:
        dom = _slide("→ normalized", "→ raw", _to_norm, _to_raw, "a")
    else:
        dom = _slide("→ raw", "→ normalized", _to_raw, _to_norm, "a")
    pts = _slide("points off", "points on", _pts_off, _pts_on)
    return widgets.HBox([dom, pts],
                        layout=widgets.Layout(margin="0px 0px 2px 0px"))
