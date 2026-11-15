"""HTML layout — controls panel + 4-subplot grid + theme stores.

Grid:
    +------------+----------------+----------------+
    | controls   | PCA scatter    | confusion mat. |
    | (sticky)   |                |                |
    |            +----------------+----------------+
    |            | distribution (full-width, scrolls)
    +------------+--------------------------------+
    | ROC (full-width)
    +-----------------------------------------------+
"""
from __future__ import annotations

import pandas as pd
from dash import dcc, html

from ..ids import IDS

DARK = {
    "bg": "#0f1115", "panel": "#181b22", "border": "#2a2e38",
    "text": "#e6e8ee", "muted": "#8a8f99", "accent": "#4ecdc4",
    "btn_bg": "#2a2e38", "btn_fg": "#e6e8ee",
}


def _section(title: str, *children) -> html.Div:
    return html.Div(
        style={
            "background": DARK["panel"],
            "border": f"1px solid {DARK['border']}",
            "borderRadius": "6px",
            "padding": "10px 12px", "marginBottom": "10px",
        },
        children=[
            html.Div(title, style={
                "fontSize": "11px", "color": DARK["muted"],
                "fontWeight": "600", "marginBottom": "6px",
                "textTransform": "uppercase", "letterSpacing": "0.5px",
            }),
            *children,
        ],
    )


def _dropdown(_id: str, options: list[str], value, multi: bool = False,
              clearable: bool = False) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=_id,
        options=[{"label": o, "value": o} for o in options],
        value=value, multi=multi, clearable=clearable,
        style={"fontSize": "12px", "color": "#000"},
    )


def _switch(_id: str, label: str, value: bool) -> dcc.Checklist:
    return dcc.Checklist(
        id=_id,
        options=[{"label": " " + label, "value": "on"}],
        value=["on"] if value else [],
        labelStyle={"fontSize": "12px", "color": DARK["text"]},
        inputStyle={"marginRight": "4px"},
    )


def _uniq(df: pd.DataFrame, col: str) -> list[str]:
    """Sorted unique non-null string values for a column."""
    if col not in df.columns:
        return []
    return sorted(df[col].dropna().astype(str).unique().tolist())


def _build_controls(df: pd.DataFrame) -> html.Div:
    from backend.representations import METHOD_REGISTRY
    # Try to import available embedding models; fall back to model_name col
    try:
        from data.real_data import available_models
        emb_models = available_models()
    except Exception:
        emb_models = []

    models = sorted(df["model_name"].unique().tolist())
    methodologies = sorted(METHOD_REGISTRY.keys())
    aggregators = ["mean", "median", "min", "max",
                   "percentile:90", "percentile:75", "percentile:25",
                   "trimmed_mean:10", "trimmed_mean:25"]
    label_keys = [c for c in df.columns if c != "centroid"]

    # Build filter option lists from df
    languages = _uniq(df, "language")
    file_languages = _uniq(df, "file_language")
    lang_opts = sorted(set(languages + file_languages))
    dialects = _uniq(df, "dialect")
    genders = _uniq(df, "gender")
    rec_types = _uniq(df, "recording_type")
    noise_levels = _uniq(df, "noise_level")
    age_buckets = _uniq(df, "age_bucket")
    dist_buckets = _uniq(df, "distance_bucket")
    vocal_efforts = _uniq(df, "vocal_effort")

    color_by_opts = [c for c in ["language", "dialect", "gender",
                                  "recording_type", "file_language"]
                     if c in df.columns]

    return html.Div(
        style={"display": "flex", "flexDirection": "column",
               "padding": "10px"},
        children=[
            # ── embedding model switcher ─────────────────────────────────
            *([_section(
                "embedding model",
                _dropdown(IDS.ctrl_emb_model, emb_models,
                          value=emb_models[0] if emb_models else None,
                          clearable=False),
                html.Div(
                    id=IDS.store_emb_model,
                    style={"fontSize": "10px", "color": DARK["muted"],
                           "marginTop": "4px"},
                    children=(
                        f"{emb_models[0]}" if emb_models
                        else "No embeddings found. Run extract_embeddings.py"
                    ),
                ),
            )] if emb_models else [
                _section(
                    "embedding model",
                    html.Div(
                        "No .parquet embeddings found in "
                        "vectors_dashboard/data/embeddings/. "
                        "Run extract_embeddings.py to populate, or set "
                        "VECTORS_DASHBOARD_EMB_DIR env var.",
                        style={"fontSize": "10px", "color": "#f0a060",
                               "lineHeight": "1.5"},
                    ),
                )
            ]),
            # stats panel hidden — too verbose for sidebar.
            html.Div(style={"display": "none"},
                     children=[html.Div(id=IDS.stats_panel)]),
            # All metadata filters dropped per dedup spec — `labels (multi)`
            # already drives grouping; `skip classes` already drives
            # exclusion. Hidden stubs keep callback Inputs valid.
            html.Div(style={"display": "none"}, children=[
                _dropdown(IDS.ctrl_lang_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_dialect_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_gender_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_recording_type_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_noise_level_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_age_bucket_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_distance_bucket_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_vocal_effort_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_is_simulated_filter, [], value=[],
                          multi=True, clearable=True),
                _dropdown(IDS.ctrl_color_by, color_by_opts,
                          value=color_by_opts[0] if color_by_opts else None,
                          clearable=False),
                html.Div(id=IDS.wav_player_panel),
                html.Div(id=IDS.store_clicked_session),
            ]),
            # `model_name` column = embedding-model identifier inside the
            # currently-loaded df. Notebook decides which model df ships
            # in (USE_REAL flag → real_data.load_real picks one parquet).
            # Keep ONE selector — labelled "embedding model" per spec.
            _section("embedding model",
                     _dropdown(IDS.ctrl_model, models,
                               value=models[0] if models else None)),
            _section(
                "methodology · aggregator · multi-centroid",
                # 30/30/30 row.  Same _dropdown helper → identical styling/colors.
                html.Div(style={"display": "flex",
                                "gap": "8px"}, children=[
                    html.Div(style={"flex": 1, "minWidth": 0}, children=[
                        html.Span("methodology", style={
                            "fontSize": "10px", "color": DARK["muted"]}),
                        _dropdown(IDS.ctrl_methodology, methodologies,
                                  value="NCM"),
                    ]),
                    html.Div(style={"flex": 1, "minWidth": 0}, children=[
                        html.Span("aggregator", style={
                            "fontSize": "10px", "color": DARK["muted"]}),
                        _dropdown(IDS.ctrl_aggregator, aggregators,
                                  value="mean"),
                    ]),
                    html.Div(style={"flex": 1, "minWidth": 0}, children=[
                        html.Span("split approach", style={
                            "fontSize": "10px", "color": DARK["muted"]}),
                        _dropdown(IDS.ctrl_centroid_approach,
                                  ["auto", "kmeans-sphere", "kmedoids",
                                   "vmf-em", "agglomerative-cos"],
                                  value="auto"),
                    ]),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center",
                                "gap": "8px", "marginTop": "8px"}, children=[
                    html.Span("global n centroids", style={
                        "fontSize": "11px", "color": DARK["muted"]}),
                    dcc.Input(
                        id=IDS.ctrl_n_centroids, type="number",
                        value=1, min=1, max=12, step=1,
                        debounce=True,
                        style={"width": "60px", "fontSize": "11px"},
                    ),
                    html.Span("per-class →", style={
                        "fontSize": "10px", "color": DARK["muted"]}),
                ]),
                html.Div(id=IDS.per_class_n_panel,
                         style={"display": "flex",
                                "flexDirection": "column",
                                "gap": "2px",
                                "marginTop": "4px"},
                         children=[]),
            ),
            _section("labels (multi)",
                     _dropdown(IDS.ctrl_labels, label_keys,
                               value=["device_id"] if "device_id" in label_keys
                                     else label_keys[:1],
                               multi=True, clearable=True)),
            _section("skip classes",
                     _dropdown(IDS.ctrl_skip_classes, [], value=[],
                               multi=True, clearable=True)),
            # view-mode buttons LIVE on the CM panel (btn-view-mode-matrix).
            # Hidden stubs keep callback Inputs valid — sidebar duplicates
            # dropped per dedup spec.
            html.Div(style={"display": "none"}, children=[
                html.Button(id=IDS.ctrl_view_mode, n_clicks=0),
                html.Button(id=IDS.btn_view_centroids, n_clicks=0),
            ]),
            # raw/normalized score-domain radio dropped — scores are
            # always normalized; the α slider in the ROC section drives
            # the normalization blend.
            html.Div(style={"display": "none"}, children=[
                dcc.RadioItems(id=IDS.ctrl_score_domain,
                               options=[{"label": "norm", "value": "normalized"}],
                               value="normalized")]),
            # box ↔ hist lives on the dist plot's updatemenus button.
            # Hidden stub keeps the callback Input/Value contract alive.
            html.Div(style={"display": "none"}, children=[
                dcc.RadioItems(
                    id=IDS.ctrl_dist_view,
                    options=[{"label": "box", "value": "box"},
                             {"label": "hist", "value": "hist"}],
                    value="box"),
            ]),
            _section(
                "2D reduction",
                _dropdown(IDS.ctrl_reduction, ["PCA", "t-SNE", "UMAP"],
                          value="PCA"),
                # tSNE-only param wrapper — visibility callback toggles it.
                html.Div(id="tsne-param-wrap",
                         style={"display": "none",
                                "marginTop": "6px",
                                "alignItems": "center", "gap": "8px"},
                         children=[
                    html.Span("perplexity", style={
                        "fontSize": "10px", "color": DARK["muted"]}),
                    dcc.Input(id=IDS.ctrl_perplexity, type="number",
                              value=30, min=5, max=100, step=5,
                              debounce=True,
                              style={"width": "55px", "fontSize": "11px"}),
                ]),
                # UMAP-only param wrapper.
                html.Div(id="umap-param-wrap",
                         style={"display": "none",
                                "marginTop": "6px",
                                "alignItems": "center", "gap": "8px"},
                         children=[
                    html.Span("n_nbrs", style={
                        "fontSize": "10px", "color": DARK["muted"]}),
                    dcc.Input(id=IDS.ctrl_n_neighbors, type="number",
                              value=15, min=2, max=200, step=1,
                              debounce=True,
                              style={"width": "55px", "fontSize": "11px"}),
                    html.Span("min_dist", style={
                        "fontSize": "10px", "color": DARK["muted"]}),
                    dcc.Input(id=IDS.ctrl_min_dist, type="number",
                              value=0.1, min=0.0, max=1.0, step=0.05,
                              debounce=True,
                              style={"width": "55px", "fontSize": "11px"}),
                ])),
            _section("PCA",
                     _switch(IDS.ctrl_show_ellipses, "show class ellipses", True),
                     _switch(IDS.ctrl_density_contour, "density contour", False),
                     html.Div(style={"marginTop": "6px"}, children=[
                         html.Span("σ", style={"fontSize": "11px",
                                               "color": DARK["muted"],
                                               "marginRight": "6px"}),
                         dcc.Input(
                             id=IDS.ctrl_ellipse_n_sigma, type="number",
                             value=2.0, min=0.5, max=5.0, step=0.5,
                             style={"width": "60px", "fontSize": "11px"},
                         ),
                     ])),
            # log_x / log_y live on the ROC plot's updatemenus toggles.
            # Hidden stubs keep the callback Input/Value contract alive.
            html.Div(style={"display": "none"}, children=[
                dcc.Checklist(id=IDS.ctrl_roc_log_x,
                              options=[{"label": "x", "value": "on"}],
                              value=[]),
                dcc.Checklist(id=IDS.ctrl_roc_log_y,
                              options=[{"label": "y", "value": "on"}],
                              value=[]),
            ]),
            _section("ROC",
                     html.Div(style={"marginTop": "6px"}, children=[
                         dcc.RadioItems(
                             id=IDS.ctrl_roc_mode,
                             options=[{"label": " all", "value": "all"},
                                      {"label": " per-class", "value": "per-class"},
                                      {"label": " pair", "value": "pair"}],
                             value="all",
                             labelStyle={"fontSize": "12px",
                                         "color": DARK["text"],
                                         "marginRight": "10px"},
                             inputStyle={"marginRight": "4px"},
                         )]),
                     html.Div(style={"marginTop": "4px"}, children=[
                         dcc.RadioItems(
                             id=IDS.ctrl_curve_mode,
                             options=[{"label": " ROC", "value": "ROC"},
                                      {"label": " DET", "value": "DET"}],
                             value="ROC",
                             labelStyle={"fontSize": "11px",
                                         "color": DARK["muted"],
                                         "marginRight": "10px"},
                             inputStyle={"marginRight": "4px"},
                         )]),
                     html.Div(style={"marginTop": "8px"}, children=[
                         html.Span("selection thr (sel-vs-rest)",
                                   style={"fontSize": "10px",
                                          "color": DARK["muted"]}),
                         dcc.Slider(id=IDS.ctrl_pair_cos_thr,
                                    min=0.0, max=1.0, step=0.005, value=0.0,
                                    marks=None, updatemode="drag",
                                    tooltip={"placement": "bottom",
                                             "always_visible": False}),
                         html.Span("pair·norm thr",
                                   style={"fontSize": "10px",
                                          "color": DARK["muted"],
                                          "marginTop": "4px",
                                          "display": "block"}),
                         dcc.Slider(id=IDS.ctrl_pair_norm_thr,
                                    min=0.0, max=1.0, step=0.005, value=0.0,
                                    marks=None, updatemode="drag",
                                    tooltip={"placement": "bottom",
                                             "always_visible": False}),
                         html.Span("α (norm blend)",
                                   style={"fontSize": "10px",
                                          "color": DARK["muted"],
                                          "marginTop": "4px",
                                          "display": "block"}),
                         dcc.Slider(id=IDS.ctrl_alpha,
                                    min=0.0, max=1.0, step=0.05, value=1.0,
                                    marks={0: "0 (minmax)", 0.5: "0.5",
                                           1: "1 (ECDF)"},
                                    tooltip={"placement": "bottom",
                                             "always_visible": False}),
                     ])),
            _section("matrix color",
                     _switch(IDS.ctrl_color_abs, "abs", False),
                     html.Div(style={"marginTop": "6px"}, children=[
                         html.Span("power", style={
                             "fontSize": "11px", "color": DARK["muted"],
                             "marginRight": "6px"}),
                         dcc.Input(
                             id=IDS.ctrl_color_power, type="number",
                             value=1.0, min=0.1, max=8.0, step=0.1,
                             style={"width": "60px", "fontSize": "11px"},
                         ),
                     ]),
                     html.Div(style={"marginTop": "8px"}, children=[
                         html.Span("range", style={
                             "fontSize": "11px", "color": DARK["muted"]}),
                         dcc.RangeSlider(
                             id=IDS.ctrl_color_range,
                             min=-1.0, max=1.0, step=0.05, value=[-1.0, 1.0],
                             marks={-1: "-1", 0: "0", 1: "1"},
                             allowCross=False,
                             tooltip={"placement": "bottom", "always_visible": False},
                         ),
                     ])),
            _section("aggregation",
                     html.Div(style={
                         "display": "flex", "alignItems": "center",
                         "gap": "8px"},
                         children=[
                             html.Button(
                                 "reset", id=IDS.btn_reset_aggregation,
                                 n_clicks=0,
                                 style={"background": DARK["btn_bg"],
                                        "color": DARK["btn_fg"],
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "4px 10px",
                                        "fontSize": "11px",
                                        "cursor": "pointer"},
                             ),
                             html.Button(
                                 "reset labels", id=IDS.btn_reset_labels,
                                 n_clicks=0,
                                 style={"background": DARK["btn_bg"],
                                        "color": DARK["btn_fg"],
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "4px 10px",
                                        "fontSize": "11px",
                                        "cursor": "pointer"},
                             ),
                             html.Span(id=IDS.badge_selection,
                                       children="0/0",
                                       style={"fontSize": "11px",
                                              "color": DARK["muted"],
                                              "marginLeft": "auto"}),
                         ]),
                     html.Div(style={"marginTop": "6px",
                                     "maxHeight": "120px",
                                     "overflowY": "auto"},
                              children=[
                                  dcc.Checklist(
                                      id=IDS.ctrl_label_allow,
                                      options=[], value=[],
                                      labelStyle={"fontSize": "11px",
                                                  "display": "block",
                                                  "color": DARK["text"]},
                                      inputStyle={"marginRight": "4px"},
                                  )])),
            _section("thresholds",
                     html.Div(id=IDS.threshold_panel, children=[])),
            _section("dump",
                     html.Div(style={"display": "flex", "gap": "6px",
                                     "flexWrap": "nowrap",
                                     "justifyContent": "space-between"},
                              children=[
                                  html.Button(
                                      "vectors.npz",
                                      id=IDS.btn_dump_npz, n_clicks=0,
                                      style={"background": DARK["btn_bg"],
                                             "color": DARK["btn_fg"],
                                             "border": "none",
                                             "borderRadius": "4px",
                                             "padding": "6px 8px",
                                             "fontSize": "11px",
                                             "cursor": "pointer",
                                             "flex": 1, "minWidth": 0,
                                             "whiteSpace": "nowrap"}),
                                  html.Button(
                                      "predictions.csv",
                                      id=IDS.btn_dump_csv, n_clicks=0,
                                      style={"background": DARK["btn_bg"],
                                             "color": DARK["btn_fg"],
                                             "border": "none",
                                             "borderRadius": "4px",
                                             "padding": "6px 8px",
                                             "fontSize": "11px",
                                             "cursor": "pointer",
                                             "flex": 1, "minWidth": 0,
                                             "whiteSpace": "nowrap"}),
                                  html.Button(
                                      "centroids.json",
                                      id=IDS.btn_dump_centroids_json,
                                      n_clicks=0,
                                      style={"background": DARK["btn_bg"],
                                             "color": DARK["btn_fg"],
                                             "border": "none",
                                             "borderRadius": "4px",
                                             "padding": "6px 8px",
                                             "fontSize": "11px",
                                             "cursor": "pointer",
                                             "flex": 1, "minWidth": 0,
                                             "whiteSpace": "nowrap"}),
                                  dcc.Download(id=IDS.download_npz),
                                  dcc.Download(id=IDS.download_csv),
                                  dcc.Download(id=IDS.download_centroids_json),
                              ])),
        ],
    )


