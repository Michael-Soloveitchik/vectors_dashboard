from __future__ import annotations
import pandas as pd
from dash import dcc, html
from ..ids import IDS
from ._controls import DARK, _build_controls

def _graph(_id: str, height: str = "700px") -> dcc.Graph:
    return dcc.Graph(
        id=_id,
        style={"height": height, "width": "100%"},
        config={"displaylogo": False, "scrollZoom": True,
                "responsive": True,
                "displayModeBar": "hover",
                "modeBarButtonsToRemove": ["zoomIn2d", "zoomOut2d"],
                "modeBarButtonsToAdd": ["resetScale2d"],
                "doubleClick": "reset"},
    )


def build_layout(df: pd.DataFrame):
    return html.Div(
        id=IDS.shell,
        style={
            "background": DARK["bg"], "color": DARK["text"],
            "minHeight": "100vh", "padding": "12px 16px",
            "fontFamily": "Inter, system-ui, -apple-system, sans-serif",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                id=IDS.header,
                style={"display": "flex", "alignItems": "center",
                       "justifyContent": "space-between",
                       "marginBottom": "12px"},
                children=[
                    html.Div("vectors_dashboard",
                             style={"fontSize": "16px", "fontWeight": "700"}),
                    html.Button("☀", id=IDS.btn_theme, n_clicks=0,
                                style={"background": "transparent",
                                       "border": f"1px solid {DARK['border']}",
                                       "color": DARK["text"],
                                       "borderRadius": "4px",
                                       "padding": "4px 10px",
                                       "cursor": "pointer"}),
                ],
            ),
            html.Div(
                style={"display": "grid",
                       "gridTemplateColumns": "320px minmax(0, 1fr) minmax(0, 1fr)",
                       "gridTemplateRows": "auto auto auto",
                       "gap": "12px",
                       "width": "100%"},
                children=[
                    html.Div(
                        id=IDS.side_col,
                        style={"gridRow": "1 / span 3",
                               "background": DARK["panel"],
                               "border": f"1px solid {DARK['border']}",
                               "borderRadius": "8px",
                               "maxHeight": "calc(100vh - 60px)",
                               "overflowY": "auto"},
                        children=_build_controls(df),
                    ),
                    html.Div(
                        style={"background": DARK["panel"],
                               "border": f"1px solid {DARK['border']}",
                               "borderRadius": "8px", "padding": "8px"},
                        children=[_graph(IDS.fig_pca, height="720px")],
                    ),
                    html.Div(
                        style={"background": DARK["panel"],
                               "border": f"1px solid {DARK['border']}",
                               "borderRadius": "8px", "padding": "8px"},
                        children=[
                            html.Div(
                                style={"display": "flex",
                                       "alignItems": "center",
                                       "gap": "8px",
                                       "marginBottom": "6px"},
                                children=[
                                    html.Button(
                                        "centroids / vectors",
                                        id=IDS.btn_view_mode_matrix,
                                        n_clicks=0,
                                        style={"background": DARK["btn_bg"],
                                               "color": DARK["btn_fg"],
                                               "border": "none",
                                               "borderRadius": "4px",
                                               "padding": "4px 10px",
                                               "fontSize": "11px",
                                               "cursor": "pointer"}),
                                    dcc.Dropdown(
                                        id=IDS.ctrl_matrix_sort,
                                        options=[],
                                        value=[], multi=True,
                                        placeholder="sort by…",
                                        style={"flex": 1,
                                               "fontSize": "11px",
                                               "color": "#000"},
                                    ),
                                ],
                            ),
                            _graph(IDS.fig_cm, height="680px"),
                        ],
                    ),
                    # Row 2: ROC (col 2) + per-label dist (col 3) side-by-side.
                    html.Div(
                        style={"gridColumn": "2",
                               "background": DARK["panel"],
                               "border": f"1px solid {DARK['border']}",
                               "borderRadius": "8px", "padding": "8px"},
                        children=[_graph(IDS.fig_roc, height="600px")],
                    ),
                    html.Div(
                        style={"gridColumn": "3",
                               "background": DARK["panel"],
                               "border": f"1px solid {DARK['border']}",
                               "borderRadius": "8px", "padding": "8px",
                               "overflow": "auto",
                               "maxHeight": "calc(100vh - 60px)"},
                        children=[_graph(IDS.fig_dist, height="600px")],
                    ),
                ],
            ),
            dcc.Store(id=IDS.store_centroids, data=None),
            dcc.Store(id=IDS.store_thresholds, data={}),
            dcc.Store(id=IDS.store_view_mode, data="vectors"),
            dcc.Store(id=IDS.store_theme, data="dark"),
            dcc.Store(id=IDS.store_aggregated, data=[]),
            dcc.Store(id=IDS.store_cm_click_pair, data=None),
            dcc.Store(id=IDS.store_per_class_n, data={}),
            dcc.Store(id=IDS.store_filtered_df, data=[]),  # list of kept session_ids
        ],
    )
