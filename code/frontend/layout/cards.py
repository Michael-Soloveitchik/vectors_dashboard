"""The four panels, the stats column and the audio dock."""
from __future__ import annotations

from dash import html

from ..ids import IDS
from .widgets import button, card, dropdown, graph, number, segmented, toggles


def _mark(sym: str, text: str):
    return html.Span([html.Span(sym, className="mk"), text], className="lg")


def _roles(*items: str):
    """What each mouse gesture does here: one ⓘ in the card header."""
    return html.Span(["i", html.Span([html.Div(i) for i in items],
                                     className="tip-body")],
                     className="tip", tabIndex=0)


def scatter_card(d: dict | None = None):
    red = (d or {}).get("reduction") or "PCA"
    return card(
        "Projection", IDS.sub_scatter,
        tools=[
            segmented(IDS.ctrl_reduction,
                      [("PCA", "PCA"), ("t-SNE", "t-SNE"),
                       ("p-t-SNE", "p-t-SNE"), ("UMAP", "UMAP")], red),
            html.Span(id=IDS.wrap_tsne, className="tool-param hidden", children=[
                html.Label("perplexity"),
                number(IDS.ctrl_perplexity, 30, 5, 100, 5, width=52)]),
            html.Span(id=IDS.wrap_umap, className="tool-param hidden", children=[
                html.Label("nbrs"),
                number(IDS.ctrl_n_neighbors, 15, 2, 200, 1, width=52),
                html.Label("min d"),
                number(IDS.ctrl_min_dist, 0.1, 0.0, 1.0, 0.05, width=52)]),
            html.Span(id=IDS.wrap_ptsne, className="tool-param hidden", children=[
                button("▶ train", IDS.btn_ptsne_train,
                       title="parametric t-SNE: train 300 more steps "
                             "(press again to refine)"),
                html.Span("not trained", id=IDS.ptsne_status,
                          className="tool-label")]),
            toggles(IDS.ctrl_scatter_opts,
                    [("envelope", "hull"), ("ellipses", "ellipses"),
                     ("density", "density")], ["hull"]),
            html.Span(id=IDS.wrap_sigma, className="tool-param hidden", children=[
                html.Label("σ"),
                number(IDS.ctrl_ellipse_sigma, 2.0, 0.5, 5.0, 0.5, width=50)]),
            html.Span(className="tool-param", children=[
                html.Label("click", title="what a plain click on a point "
                                          "does; Ctrl+click does the other"),
                segmented(IDS.ctrl_click_mode,
                          [("inspect", "inspect"), ("select", "select")],
                          "inspect")]),
            html.Span(className="tool-param colour-by", children=[
                html.Label("colour", title="colour the points by class or by "
                                           "any other column"),
                dropdown(IDS.ctrl_color_by, ["class"], "class")]),
        ],
        legend=[_mark("●", "accepted"), _mark("○", "below own threshold"),
                _mark("✕", "another class wins"), _mark("◇", "skipped"),
                _mark("✚", "centroid"), _mark("╌", "accepted envelope")],
        roles=_roles("click a point: inspect it — or select it (the "
                     "“click” switch)", "Ctrl+click: the other of the two",
                     "drag: select", "Shift+drag: add to the selection",
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
                      children=[html.Label("sort by"),
                                dropdown(IDS.ctrl_matrix_sort, [], None,
                                         multi=True, clearable=True,
                                         placeholder="1st key, 2nd key…")]),
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
        roles=_roles("drag a ● spot along its curve: that class's "
                     "threshold follows", "click a curve: threshold jumps "
                     "to that point", "click a ● spot: inspect the class",
                     "drag elsewhere: zoom", "right-click: reset zoom"),
        body=graph(IDS.fig_roc), cls="card-roc")


def dist_card():
    return card(
        "Score distribution", IDS.sub_dist,
        tools=[segmented(IDS.ctrl_score_domain,
                         [("normalized", "norm"), ("raw", "raw")], "norm"),
               segmented(IDS.ctrl_dist_view,
                         [("histogram", "hist"), ("box", "box"),
                          ("samples", "strip")], "hist")],
        roles=_roles("click in a lane: that class's threshold moves there",
                     "normalized: the scale thresholds are set on",
                     "raw: the methodology's own scores",
                     "drag: zoom", "right-click: reset zoom"),
        body=graph(IDS.fig_dist), cls="card-dist")


def inspector():
    """Stats of what was clicked: a narrow column right of the panels."""
    return html.Aside(id=IDS.inspector, className="drawer", children=[
        html.Div(className="drawer-head", children=[
            html.Span("Inspector", id=IDS.inspector_title,
                      className="card-title"),
            button("✕", IDS.btn_inspector_close, kind="icon",
                   title="close  (Esc)"),
        ]),
        html.Div(className="drawer-body", children=[
            html.Div(id=IDS.inspector_body, className="insp-details"),
        ]),
    ])


def audio_dock():
    """WavLens for the inspected vector(s): full width, under every panel.
    Absent while there is nothing to play; ▾ folds it to its title bar."""
    return html.Section(id="audio-dock", className="dock", children=[
        html.Div(className="dock-head", children=[
            html.Span("Audio", className="card-title"),
            html.Span("WavLens · wheel = zoom · click = seek · space = play "
                      "· FFT = spectrogram · ☾ = its own dark mode",
                      className="card-sub"),
            html.Button("▾", className="btn btn-icon dock-toggle",
                        title="fold / unfold the audio panel"),
        ]),
        html.Div(id=IDS.inspector_audio, className="insp-audio"),
    ])
