"""Layout: top bar · sidebar · [2×2 panel grid | stats column] over the
audio dock (WavLens, full width under every panel)."""
from __future__ import annotations

from dash import dcc, html

from ..engine import DataHub
from ..ids import IDS
from .cards import (audio_dock, dist_card, inspector, matrix_card, roc_card,
                    scatter_card)
from .sidebar import build_sidebar

PANEL_NAMES = ["projection", "matrix", "curves", "distribution", "summary",
               "inspector"]


def build_layout(hub: DataHub, defaults: dict | None = None) -> html.Div:
    d = defaults or {}
    return html.Div(className="shell", children=[
        html.Header(className="topbar", children=[
            html.Div(["vectors", html.Span("dashboard")], className="brand"),
            html.Div(id=IDS.summary, className="chips"),
            html.Span("◌ loading", id=IDS.sync, className="sync busy",
                      title="every panel shows the same fit, thresholds, "
                            "selection and focus"),
            html.Button("⌂ view", id="btn-home-all", n_clicks=0,
                        className="btn",
                        title="back to start: reset every zoom, clear the "
                              "selection, close the inspector"),
            html.Button([html.Span("☀", className="ts-ico ts-sun"),
                         html.Span("☾", className="ts-ico ts-moon"),
                         html.I(className="ts-knob")],
                        id=IDS.btn_theme, n_clicks=0, className="theme-switch",
                        title="light / dark (remembered)"),
        ]),
        html.Div(className="body", children=[
            build_sidebar(hub, d),
            html.Div(className="stage", children=[
                html.Div(className="work", children=[
                    html.Main(className="grid", children=[
                        scatter_card(d), matrix_card(), roc_card(),
                        dist_card()]),
                    inspector(),
                ]),
                audio_dock(),
            ]),
        ]),
        dcc.Store(id=IDS.store_fit, data=None),
        dcc.Store(id=IDS.store_thr, data=None),
        dcc.Store(id=IDS.store_sel, data=None),
        dcc.Store(id=IDS.store_focus, data=None),
        # None = follow the saved choice, else the OS colour scheme
        dcc.Store(id=IDS.store_theme, data=d.get("theme")),
        dcc.Store(id=IDS.store_client, data=None),
        dcc.Store(id=IDS.store_want, data=None),
        dcc.Store(id=IDS.store_kick, data=0),
        dcc.Store(id=IDS.store_late, data=None),
        dcc.Store(id=IDS.lasso_echo, data=0),
        *[dcc.Store(id=i, data=None) for i in (
            IDS.int_lasso, IDS.int_classes, IDS.int_click, IDS.int_insp,
            IDS.int_reset, IDS.store_mods, IDS.thr_pick, IDS.store_labels,
            IDS.op_pick)],
        dcc.Store(id=IDS.drawn_scatter, data=0),
        dcc.Store(id=IDS.store_ptsne, data=None),
        dcc.Interval(id=IDS.tick, interval=1500),
        *[dcc.Store(id=f"stamp-{n}", data=None) for n in PANEL_NAMES],
        dcc.Store(id=IDS.store_per_class_n, data={}),
        dcc.Store(id=IDS.sig_scatter), dcc.Store(id=IDS.sig_roc),
        dcc.Store(id=IDS.sig_dist), dcc.Store(id=IDS.sig_matrix),
    ])


__all__ = ["build_layout"]
