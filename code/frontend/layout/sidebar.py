"""Sidebar — everything that changes WHAT is computed. Per-panel view
toggles live in each card header (see cards.py)."""
from __future__ import annotations

from dash import dcc, html

from ..engine import DataHub
from ..ids import IDS
from .widgets import (button, dropdown, field, number, section, segmented,
                      toggles)

AGGREGATORS = ["mean", "median", "min", "max", "percentile:90",
               "percentile:75", "percentile:25", "trimmed_mean:10",
               "trimmed_mean:25"]


def _slider(_id, value, **kw):
    return dcc.Slider(id=_id, min=0.0, max=1.0, value=value, marks=None,
                      updatemode="drag", className="slider",
                      tooltip={"placement": "bottom", "always_visible": False},
                      **kw)


def build_sidebar(hub: DataHub, d: dict | None = None) -> html.Aside:
    from backend.representations import METHOD_REGISTRY
    d = d or {}
    model = d.get("model") if d.get("model") in hub.models else hub.models[0]
    cols = hub.label_columns(model)
    labels = [c for c in (d.get("labels") or []) if c in cols] \
        or hub.default_labels(model)
    method = d.get("methodology") if d.get("methodology") in METHOD_REGISTRY \
        else "NCM"
    aggregator = d.get("aggregator") if d.get("aggregator") in AGGREGATORS \
        else "mean"
    return html.Aside(className="sidebar", children=[
        section(
            "Data",
            field("embedding model", dropdown(IDS.ctrl_model, hub.models, model)),
            field("class labels", dropdown(
                IDS.ctrl_labels, cols, labels, multi=True, clearable=True,
                placeholder="label column(s)…"),
                hint="several columns → joint classes"),
            field("skip from fit", dropdown(
                IDS.ctrl_skip, [], [], multi=True, clearable=True,
                placeholder="classes / values to exclude…")),
            field("max points drawn", number(
                IDS.ctrl_max_points, int(d.get("max_points") or 4000), 200,
                50000, 100, width=84),
                hint="metrics use all"),
        ),
        section(
            "Method",
            field("methodology", dropdown(
                IDS.ctrl_methodology, sorted(METHOD_REGISTRY), method)),
            html.Div(className="row", children=[
                field("aggregator", dropdown(
                    IDS.ctrl_aggregator, AGGREGATORS, aggregator),
                    style={"flex": "1 1 0", "minWidth": 0}),
                field("centroids / class", number(
                    IDS.ctrl_n_centroids, 1, 1, 12, 1, width=64)),
            ]),
            html.Details(className="sub-details", children=[
                html.Summary("per-class centroid count"),
                html.Div(id=IDS.per_class_n_panel, className="per-class-n"),
            ]),
            html.Div(id=IDS.fit_status, className="fit-status"),
        ),
        section(
            "Thresholds",
            *[
                html.Div(className="thr-all", children=[
                    html.Span("all classes", className="thr-name"),
                    html.Div(_slider(IDS.ctrl_thr_all, 0.0, step=0.005),
                             className="thr-slider"),
                ]),
                html.Div(className="row gap", children=[
                    button("EER", IDS.btn_thr_eer,
                           title="each class → its equal-error-rate threshold"),
                    button("min DCF", IDS.btn_thr_dcf,
                           title="each class → the threshold minimizing the "
                                 "detection cost (P_target 0.01, C_miss = "
                                 "C_fa = 1)"),
                    button("reset", IDS.btn_thr_reset,
                           title="accept everything"),
                ]),
                html.Div(id=IDS.threshold_panel, className="thr-panel"),
                field("score normalization  α", _slider(
                    IDS.ctrl_alpha, 1.0, step=0.05),
                    hint="0 = min-max · 1 = ECDF rank"),
            ],
            hint="normalized score",
        ),
        section(
            "Selection",
            html.Div(className="row gap", children=[
                html.Span("0 selected", id=IDS.badge_selection,
                          className="badge"),
                button("clear", IDS.btn_sel_clear),
            ]),
            field("combine selections", segmented(
                IDS.ctrl_sel_mode, [("replace", "single"), ("add", "union"),
                                    ("intersect", "intersect")], "single"),
                hint="lasso · classes · picks"),
            field("select whole classes", dropdown(
                IDS.ctrl_sel_classes, [], [], multi=True, clearable=True,
                placeholder="or lasso / box on the projection…")),
            html.Div(id=IDS.sel_summary, className="sel-summary"),
            html.Div(id=IDS.sel_list, className="sel-list"),
            field("my labels", html.Div(className="row gap", children=[
                dcc.Input(id=IDS.ctrl_label_name, type="text", debounce=False,
                          placeholder="name…  (empty = unlabel)", maxLength=60,
                          className="num", style={"flex": "1 1 0",
                                                  "minWidth": 0}),
                button("label selection", IDS.btn_label_apply,
                       title="give the selected vectors this label"),
                button("✕", IDS.btn_label_clear, kind="icon",
                       title="remove ALL my labels of this model"),
            ]), hint="→ class labels · colour · matrix sort"),
            html.Div(id=IDS.label_status, className="note"),
        ),
        section(
            "Matrix colour",
            toggles(IDS.ctrl_color_auto,
                    [("auto range", "auto"), ("abs", "abs")], ["auto"]),
            html.Div(className="row gap", children=[
                field("power", number(IDS.ctrl_color_power, 1.0, 0.1, 8.0, 0.1)),
                field("manual range", dcc.RangeSlider(
                    id=IDS.ctrl_color_range, min=-1.0, max=1.0, step=0.05,
                    value=[-1.0, 1.0], marks=None, allowCross=False,
                    className="slider",
                    tooltip={"placement": "bottom", "always_visible": False}),
                    style={"flex": "1 1 0", "minWidth": 0}),
            ]),
            open_=False,
        ),
        section(
            "Export",
            html.Div(className="row gap wrap", children=[
                button("vectors.npz", IDS.btn_dump_npz),
                button("predictions.csv", IDS.btn_dump_csv),
                button("centroids.json", IDS.btn_dump_json),
            ]),
            dcc.Download(id=IDS.download),
            html.Div("also written to ./dumps/", className="note"),
            open_=False,
        ),
    ])
