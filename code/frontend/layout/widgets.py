"""Small layout atoms. All styling lives in assets/style.css."""
from __future__ import annotations

from dash import dcc, html

GRAPH_CONFIG = {
    "displaylogo": False, "scrollZoom": True, "responsive": True,
    "displayModeBar": "hover", "doubleClick": "reset",
    "modeBarButtonsToRemove": ["zoomIn2d", "zoomOut2d", "autoScale2d",
                               "toggleSpikelines", "hoverClosestCartesian",
                               "hoverCompareCartesian"],
    "toImageButtonOptions": {"scale": 2},
}


def section(title: str, *children, open_: bool = True, hint: str | None = None):
    head = [html.Span(title, className="sec-title")]
    if hint:
        head.append(html.Span(hint, className="sec-hint"))
    return html.Details(open=open_, className="section", children=[
        html.Summary(head), html.Div(children, className="sec-body")])


def field(label: str, control, hint: str | None = None, **kw):
    kids = [html.Label(label, className="field-label")]
    if hint:
        kids.append(html.Span(hint, className="field-hint"))
    return html.Div([html.Div(kids, className="field-head"), control],
                    className="field", **kw)


def dropdown(_id, options, value=None, multi=False, clearable=False,
             placeholder=None, **kw):
    opts = [o if isinstance(o, dict) else {"label": o, "value": o}
            for o in options]
    return dcc.Dropdown(id=_id, options=opts, value=value, multi=multi,
                        clearable=clearable, placeholder=placeholder,
                        className="dd", **kw)


def segmented(_id, options: list[tuple[str, str]], value: str):
    """Pill group (exclusive). ``options`` = [(label, value), ...]."""
    return dcc.RadioItems(
        id=_id, value=value, inline=True, className="seg",
        options=[{"label": lbl, "value": v} for lbl, v in options],
        inputClassName="seg-input", labelClassName="seg-label")


def toggles(_id, options: list[tuple[str, str]], value: list[str]):
    """Pill group (independent on/off)."""
    return dcc.Checklist(
        id=_id, value=value, inline=True, className="seg seg-multi",
        options=[{"label": lbl, "value": v} for lbl, v in options],
        inputClassName="seg-input", labelClassName="seg-label")


def number(_id, value, min_, max_, step, width: int = 58, **kw):
    return dcc.Input(id=_id, type="number", value=value, min=min_, max=max_,
                     step=step, debounce=True, className="num",
                     style={"width": f"{width}px"}, **kw)


def button(label, _id, kind: str = "ghost", **kw):
    return html.Button(label, id=_id, n_clicks=0, className=f"btn btn-{kind}",
                       **kw)


def graph(_id: str) -> dcc.Loading:
    return dcc.Loading(
        dcc.Graph(id=_id, config=GRAPH_CONFIG, className="graph",
                  style={"height": "100%", "width": "100%"}),
        delay_show=350, type="dot", color="var(--accent)",
        overlay_style={"visibility": "visible", "opacity": 0.45},
        parent_className="graph-wrap")


def card(title: str, sub_id: str, tools: list, body, *, cls: str = "",
         legend: list | None = None, roles=None):
    head = [html.Div([html.Span(title, className="card-title"),
                      html.Span(id=sub_id, className="card-sub")],
                     className="card-heading"),
            html.Div([*tools, html.Button(
                "⌂", className="btn btn-icon home-btn",
                title="reset zoom  (also: right-click the plot)")],
                className="card-tools")]
    kids = [html.Div(head, className="card-head")]
    if legend:
        kids.append(html.Div(legend, className="card-legend"))
    if roles is not None:          # what each mouse gesture does here
        kids.append(html.Div(roles, className="card-legend"))
    kids.append(html.Div(body, className="card-body"))
    return html.Section(kids, className=f"card {cls}".strip())
