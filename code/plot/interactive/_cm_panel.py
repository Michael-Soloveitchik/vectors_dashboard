"""Confusion-matrix figure controller — class wrapping all CM state.

Owns ``f4`` (CM FigureWidget), the four control widgets, the PCA xy
caches, and the gold-ring highlight trace appended to the PCA figure.
A click on a CM cell maps ``(row, col)`` → two PCA points and writes
them into the highlight trace; mode toggle reroutes the mapping
(samples vs centroids) and resizes the ring to hug whichever marker
class it now wraps.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..cm import build_cm_fig


__all__ = ["CmPanel", "make_cm_controls_box"]


# Ring sizes chosen so the gold circle sits one step outside the
# underlying marker (sample dots: size 6 in pca.py; centroid X: size 17).
_RING_SIZES = {"vectors": 12, "centroids": 24}
_RING_LINE_WIDTH = 2.0
_RING_COLOR = "gold"
_HIGHLIGHT_NAME = "cm·highlight"


class CmPanel:
    def __init__(self, *, f3, initial_pca_ref: dict, devices_sorted,
                 method_label: str, model_name: str):
        import ipywidgets as widgets

        self.f3 = f3
        self.initial_pca_ref = initial_pca_ref
        self.devices_sorted = devices_sorted
        self.method_label = method_label
        self.model_name = model_name
        self.xy_samples: np.ndarray | None = None
        self.xy_centroids: np.ndarray | None = None
        self.highlight_idx: int | None = None

        self.mode_toggle = widgets.ToggleButtons(
            options=["vectors", "centroids"], value="vectors",
            description="cm mode",
            layout=widgets.Layout(width="340px"),
            style={"description_width": "70px"},
            tooltips=["matrix from sample vectors (same V going into PCA)",
                      "matrix from class centroids (one row per centroid)"],
        )
        self.abs_toggle = widgets.ToggleButton(
            value=False, description="abs",
            layout=widgets.Layout(width="80px"),
            tooltip="take absolute value before power+clip",
        )
        self.power_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=8.0, step=0.1,
            description="power", continuous_update=False,
            layout=widgets.Layout(width="280px"),
            style={"description_width": "60px"},
            readout_format=".2f",
        )
        self.clip_slider = widgets.FloatRangeSlider(
            value=[-1.0, 1.0], min=-1.0, max=1.0, step=0.05,
            description="clip", continuous_update=False,
            layout=widgets.Layout(width="340px"),
            style={"description_width": "70px"},
            readout_format=".2f",
        )
        self.note = widgets.HTML(
            "<span style='color:#aaa;font-size:10px'>"
            "click a cell → tight gold ring on the matching PCA points. "
            "abs → power → clip applied in that order."
            "</span>"
        )

        self.f4 = (go.FigureWidget(self._build_fig())
                   if initial_pca_ref else None)
        if self.f4 is not None:
            self._bind_click()
        for w in (self.mode_toggle, self.abs_toggle,
                  self.power_slider, self.clip_slider):
            w.observe(self._on_control_change, names="value")
        if self.f3 is not None:
            self.populate_xy_cache()
            self.ensure_highlight_trace()

    # ── public ──────────────────────────────────────────────────────
    def populate_xy_cache(self) -> None:
        if self.f3 is None or not self.initial_pca_ref:
            return
        plot_dev = self.initial_pca_ref["plot_df"]["device_id"].to_numpy()
        xy_by_dev: dict[object, np.ndarray] = {}
        for tr in self.f3.data:
            if (tr.mode == "markers"
                    and tr.name in self.devices_sorted
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
        self.xy_samples = xy_all
        self.xy_centroids = None
        for tr in self.f3.data:
            nm = str(getattr(tr, "name", "") or "")
            if nm.endswith(" centroids"):
                self.xy_centroids = np.column_stack([
                    np.asarray(tr.x, dtype=np.float64),
                    np.asarray(tr.y, dtype=np.float64),
                ])
                break

    def ensure_highlight_trace(self) -> None:
        if self.f3 is None:
            return
        for i, tr in enumerate(self.f3.data):
            if str(getattr(tr, "name", "") or "") == _HIGHLIGHT_NAME:
                self.highlight_idx = i
                return
        self.f3.add_trace(go.Scatter(
            x=[], y=[], mode="markers",
            marker=dict(size=_RING_SIZES[self.mode_toggle.value],
                        color="rgba(0,0,0,0)",
                        line=dict(width=_RING_LINE_WIDTH, color=_RING_COLOR),
                        symbol="circle"),
            name=_HIGHLIGHT_NAME, showlegend=False,
            hoverinfo="skip",
        ))
        self.highlight_idx = len(self.f3.data) - 1

    def rebuild(self, _change=None) -> None:
        if self.f4 is None:
            return
        new = self._build_fig()
        with self.f4.batch_update():
            if len(self.f4.data) != len(new.data):
                self.f4.data = ()
                self.f4.add_traces(list(new.data))
            else:
                for i, ntr in enumerate(new.data):
                    self.f4.data[i].z = ntr.z
                    self.f4.data[i].x = ntr.x
                    self.f4.data[i].y = ntr.y
                    self.f4.data[i].zmin = ntr.zmin
                    self.f4.data[i].zmax = ntr.zmax
                    self.f4.data[i].customdata = ntr.customdata
                    self.f4.data[i].name = ntr.name
            self.f4.layout.title = new.layout.title
            self.f4.layout.meta = new.layout.meta
        self._bind_click()
        self._clear_highlight()

    # ── internal ────────────────────────────────────────────────────
    def _build_fig(self) -> go.Figure:
        ipca = self.initial_pca_ref or {}
        return build_cm_fig(
            V_emb=ipca.get("V_emb"),
            centroid_rows=ipca.get("centroid_rows", []),
            centroid_labels=ipca.get("centroid_labels", []),
            plot_df=ipca.get("plot_df"),
            method_label=self.method_label, model_name=self.model_name,
            mode=self.mode_toggle.value,
            abs_on=bool(self.abs_toggle.value),
            power=float(self.power_slider.value),
            clip_lo=float(self.clip_slider.value[0]),
            clip_hi=float(self.clip_slider.value[1]),
        )

    def _bind_click(self) -> None:
        if self.f4 is None or len(self.f4.data) == 0:
            return
        try:
            self.f4.data[0].on_click(self._on_cell_click)
        except Exception:
            pass

    def _clear_highlight(self) -> None:
        if self.f3 is None or self.highlight_idx is None:
            return
        with self.f3.batch_update():
            self.f3.data[self.highlight_idx].x = []
            self.f3.data[self.highlight_idx].y = []

    def _on_control_change(self, _change) -> None:
        self.rebuild()
        if self.f3 is not None and self.highlight_idx is not None:
            with self.f3.batch_update():
                self.f3.data[self.highlight_idx].marker.size = (
                    _RING_SIZES[self.mode_toggle.value]
                )

    def _on_cell_click(self, _trace, points, _state) -> None:
        if self.f3 is None or self.highlight_idx is None:
            return
        xs = list(getattr(points, "xs", []) or [])
        ys = list(getattr(points, "ys", []) or [])
        if not xs or not ys:
            return
        col = int(round(float(xs[0])))
        row = int(round(float(ys[0])))
        coords = (self.xy_samples
                  if self.mode_toggle.value == "vectors"
                  else self.xy_centroids)
        if coords is None or coords.shape[0] == 0:
            return
        n = coords.shape[0]
        if not (0 <= row < n and 0 <= col < n):
            return
        with self.f3.batch_update():
            self.f3.data[self.highlight_idx].x = [
                float(coords[row, 0]), float(coords[col, 0]),
            ]
            self.f3.data[self.highlight_idx].y = [
                float(coords[row, 1]), float(coords[col, 1]),
            ]
            self.f3.data[self.highlight_idx].marker.size = (
                _RING_SIZES[self.mode_toggle.value]
            )


def make_cm_controls_box(panel: CmPanel):
    import ipywidgets as widgets
    header = widgets.HTML(
        "<b>cosine matrix</b> — "
        "<span style='color:#aaa'>"
        "same vectors as PCA · click cell to ring both points on PCA"
        "</span>"
    )
    return widgets.VBox([
        header,
        widgets.HBox([panel.mode_toggle, panel.abs_toggle]),
        panel.power_slider,
        panel.clip_slider,
        panel.note,
    ])
