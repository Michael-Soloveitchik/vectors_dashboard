"""The four panels + the inspector drawer."""
from __future__ import annotations

from dash import html

from ..ids import IDS
from .widgets import button, card, dropdown, graph, number, segmented, toggles


def _mark(sym: str, text: str):
    return html.Span([html.Span(sym, className="mk"), text], className="lg")


def _roles(*items: str):
    """What each mouse gesture does, right where it is used."""
    return html.Span([html.Span(i, className="role") for i in items],
                     className="roles")


def scatter_card(d: dict | None = None):
    red = (d or {}).get("reduction") or "PCA"
    return card(
        "Projection", IDS.sub_scatter,
        tools=[
            segmented(IDS.ctrl_reduction,
                      [("PCA", "PCA"), ("t-SNE", "t-SNE"), ("UMAP", "UMAP")],
                      red),
            html.Span(id=IDS.wrap_tsne, className="tool-param hidden", children=[
                html.Label("perplexity"),
                number(IDS.ctrl_perplexity, 30, 5, 100, 5, width=52)]),
            html.Span(id=IDS.wrap_umap, className="tool-param hidden", children=[
                html.Label("nbrs"),
                number(IDS.ctrl_n_neighbors, 15, 2, 200, 1, width=52),
                html.Label("min d"),
                number(IDS.ctrl_min_dist, 0.1, 0.0, 1.0, 0.05, width=52)]),
            toggles(IDS.ctrl_scatter_opts,
                    [("envelope", "hull"), ("density", "density")], ["hull"]),
            html.Span(className="tool-param colour-by", children=[
                html.Label("colour", title="colour the points by class or by "
                                           "any other column"),
                dropdown(IDS.ctrl_color_by, ["class"], "class")]),
        ],
        legend=[_mark("●", "accepted"), _mark("○", "below own threshold"),
                _mark("✕", "another class wins"), _mark("◇", "skipped"),
                _mark("✚", "centroid"), _mark("╌", "accepted envelope")],
        roles=_roles("click: inspect", "Ctrl+click: add / remove",
                     "drag: select", "Shift+drag: add",
                     "double-click: clear", "right-click: reset zoom"),
        body=graph(IDS.fig_scatter), cls="card-scatter")


def matrix_card():
    return card(
        "Matrix", IDS.sub_matrix,
        tools=[
            segmented(IDS.ctrl_matrix_mode,
                      [("pair scores", "vectors"), ("centroids", "classes"),
                       ("confusion", "confusion"), ("tree", "tree")],
                      "vectors"),
            html.Span(id=IDS.wrap_matrix_sort, className="tool-param grow",
                      children=[dropdown(IDS.ctrl_matrix_sort, [], [],
                                         multi=True, clearable=True,
                                         placeholder="sort rows by…")]),
        ],
        roles=_roles("click a cell: inspect the pair",
                     "Ctrl+click: add / remove both vectors", "drag: zoom",
                     "right-click: reset zoom",
                     "tree: click a junction → select its classes"),
        body=graph(IDS.fig_matrix), cls="card-matrix")


def roc_card():
    return card(
        "Detection curves", IDS.sub_roc,
        tools=[
            segmented(IDS.ctrl_curve, [("ROC", "roc"), ("DET", "det")], "roc"),
            segmented(IDS.ctrl_roc_show,
                      [("all", "all"), ("per class", "class"),
                       ("pairs", "pair")], "all"),
            toggles(IDS.ctrl_roc_axes, [("log x", "x"), ("log y", "y")], []),
        ],
        body=graph(IDS.fig_roc), cls="card-roc")


def dist_card():
    return card(
        "Score distribution", IDS.sub_dist,
        tools=[segmented(IDS.ctrl_dist_view,
                         [("histogram", "hist"), ("samples", "strip")],
                         "hist")],
        body=graph(IDS.fig_dist), cls="card-dist")


def inspector():
    return html.Div(id=IDS.inspector, className="drawer", children=[
        html.Div(className="drawer-head", children=[
            html.Span("Inspector", id=IDS.inspector_title,
                      className="card-title"),
            button("✕", IDS.btn_inspector_close, kind="icon", title="close"),
        ]),
        html.Div(className="drawer-body", children=[
            html.Div(id=IDS.inspector_audio, className="insp-audio"),
            html.Div(id=IDS.inspector_body, className="insp-details"),
        ]),
    ])
