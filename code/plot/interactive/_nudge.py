"""◀/▶ nudge button row — advances a SelectionSlider by one option."""
from __future__ import annotations


__all__ = ["NUDGE_CSS_HTML", "make_nudged_row"]


_NUDGE_CSS = """<style>
.nbp-nudge button {
    background: linear-gradient(180deg, #4a505b, #2e333c) !important;
    color: rgba(238,238,242,0.95) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 4px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    font-size: 11px !important;
    line-height: 1 !important;
    padding: 0 !important;
    min-height: 0 !important;
    transition: background 120ms ease, border-color 120ms ease !important;
}
.nbp-nudge button:hover:not(:disabled) {
    background: linear-gradient(180deg, #5a606b, #383e48) !important;
    border-color: rgba(255,255,255,0.28) !important;
}
.nbp-nudge button:active:not(:disabled) {
    background: linear-gradient(180deg, #24282f, #1b1e24) !important;
}
.nbp-nudge button:disabled {
    opacity: 0.35 !important;
}
</style>"""


def NUDGE_CSS_HTML():
    """Return an ipywidgets HTML carrying the nudge button stylesheet.

    Lazy import keeps this module importable in non-Jupyter test contexts."""
    import ipywidgets as widgets
    return widgets.HTML(_NUDGE_CSS)


def _layout():
    import ipywidgets as widgets
    return widgets.Layout(width="26px", height="22px",
                          min_width="26px", padding="0px",
                          margin="0px 4px")


def _nudge(sl, sign: int):
    """Return a click callback that advances ``sl`` by one option in ``sign``."""
    import ipywidgets as widgets

    def _cb(_btn):
        if isinstance(sl, widgets.SelectionSlider):
            n = len(sl.options)
            if n == 0:
                return
            new_idx = max(0, min(n - 1, int(sl.index) + sign))
            if new_idx != sl.index:
                sl.index = new_idx
            return
        # FloatSlider (temperature): step-based nudge.
        cur = float(sl.value)
        v = cur + sign * sl.step
        v = max(float(sl.min), min(float(sl.max), v))
        if v != sl.value:
            sl.value = v

    return _cb


def make_nudged_row(sl):
    """HBox of [◀ button, slider, ▶ button] with the dark CSS class applied."""
    import ipywidgets as widgets

    btn_l = widgets.Button(description="◀", layout=_layout(),
                           tooltip="prev threshold", disabled=sl.disabled)
    btn_r = widgets.Button(description="▶", layout=_layout(),
                           tooltip="next threshold", disabled=sl.disabled)
    btn_l.add_class("nbp-nudge")
    btn_r.add_class("nbp-nudge")
    btn_l.on_click(_nudge(sl, -1))
    btn_r.on_click(_nudge(sl, +1))
    return widgets.HBox([btn_l, sl, btn_r],
                        layout=widgets.Layout(align_items="center"))
