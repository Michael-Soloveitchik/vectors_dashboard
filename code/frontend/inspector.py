"""Inspector drawer content: one vector, a pair of vectors, or classes."""
from __future__ import annotations

from urllib.parse import quote

import numpy as np
from dash import html

from .engine import Analysis, DataHub, state
from .engine import curves as cv
from .engine import matrix as mx
from .theme import class_colors

SKIP_META = {"session_id", "model_name"}
TOP_VOTES = 8


def _tag(text: str, kind: str = "") -> html.Span:
    return html.Span(text, className=f"tag {kind}".strip())


def note(text: str) -> list:
    return [html.Div(text, className="note")]


def _act(label: str, act: str, title: str) -> html.Button:
    return html.Button(label, id={"type": "insp-act", "act": act}, n_clicks=0,
                       className="btn", title=title)


def _actions(an: Analysis, row: int) -> html.Div:
    """Turn what is inspected into a selection (all panels follow)."""
    kids = [_act("select its class", "class", "every vector of this class"),
            _act("select neighbours", "nbrs", "this vector + its nearest")]
    df = an.frame.df
    if "speaker_id" in df.columns:
        n = int((df["speaker_id"].astype(str) == str(df["speaker_id"].iloc[row])).sum())
        if n > 1:
            kids.append(_act(f"select speaker ({n})", "speaker",
                             "every vector of the same speaker_id"))
    return html.Div(kids, className="row gap wrap")


def composition(hub: DataHub, an: Analysis, rows: list[int]) -> list:
    """What a selection is made of: top values of every label column."""
    if not rows:
        return []
    df = an.frame.df.iloc[rows]
    out = []
    for col in hub.label_columns(an.frame.name)[:7]:
        vc = df[col].astype(str).value_counts(normalize=True)
        txt = " · ".join(f"{k} {v:.0%}" for k, v in vc.head(3).items())
        if len(vc) > 3:
            txt += f" · +{len(vc) - 3}"
        out += [html.Span(col, className="k"),
                html.Span(txt, className="v", title=txt)]
    return out


def _neighbours(an: Analysis, row: int, cmap) -> html.Div:
    kids = []
    for r, cos in state.neighbours(an, row):
        lbl = str(an.y[r])
        kids += [html.Span([html.I(style={"background": cmap.get(lbl, "#888")}),
                            lbl], className="thr-name", title=lbl),
                 html.Span(an.frame.sids[r], className="insp-id"),
                 html.Span(f"{cos:.3f}", className="num-r")]
    return html.Div(className="insp-item", children=[
        html.Div("nearest neighbours  ·  cosine  ·  ringed on the projection",
                 className="insp-h", style={"marginTop": 0}),
        html.Div(kids, className="nbrs")])


def _votes(an: Analysis, row: int, P: np.ndarray, tv: np.ndarray, cmap) -> html.Div:
    order = np.argsort(-P[row])[:TOP_VOTES]
    kids = []
    for k in order:
        lbl = an.labels[k]
        kids += [
            html.Span(lbl, title=lbl),
            html.Div(className="vbar", children=[
                html.I(style={"width": f"{P[row, k] * 100:.1f}%",
                              "background": cmap.get(lbl, "#888")}),
                html.B(style={"left": f"{tv[k] * 100:.1f}%"},
                       title=f"threshold {tv[k]:.3f}")]),
            html.Span(f"{P[row, k]:.3f}", className="num-r"),
        ]
    return html.Div(kids, className="votes")


def _row_item(an: Analysis, row: int, d, P, tv, cmap, role: str = "") -> html.Div:
    sid, lbl = an.frame.sids[row], str(an.y[row])
    pred = an.labels[d.pred[row]] if d.pred[row] >= 0 else None
    tags = [_tag(lbl)]
    if an.true_idx[row] < 0:
        tags.append(_tag("class skipped in fit"))
    elif pred is None:
        tags.append(_tag("rejected by every class", "bad"))
    elif d.correct[row]:
        tags.append(_tag(f"predicted {pred}", "ok"))
    else:
        tags.append(_tag(f"predicted {pred}", "bad"))
    if an.true_idx[row] >= 0 and not d.own_ok[row]:
        tags.append(_tag("below own-class threshold", "bad"))
    if not an.fit_mask[row]:
        tags.append(_tag("excluded from fit"))
    meta = an.frame.df.iloc[row]
    kv = []
    for col, val in meta.items():
        if col in SKIP_META or val is None or str(val) in ("nan", "unknown", ""):
            continue
        txt = f"{val:.3g}" if isinstance(val, float) else str(val)
        kv += [html.Dt(col), html.Dd(txt if len(txt) <= 44 else txt[:41] + "…",
                                     title=txt)]
    return html.Div(className="insp-item", children=[
        html.Div([html.Span(role, className="field-label") if role else None,
                  html.Span(sid, className="insp-id"), *tags],
                 className="insp-head"),
        html.Dl(kv[:24], className="kv"),
        html.Div("normalized score per class  ·  gold = threshold",
                 className="insp-h"),
        _votes(an, row, P, tv, cmap),
    ])


def audio_view(hub: DataHub, an: Analysis, rows: list[int]) -> list:
    if not hub.has_audio:
        return note("no audio — pass wav_dir=… to Gui to listen here")
    have = [r for r in rows if hub.wav_path(an.frame.sids[r]) is not None]
    if not have:
        return note("no wav file found for this vector")
    names = "|".join(f"{an.y[r]} · {an.frame.sids[r][:8]}" for r in have)
    # relative URL: the page may be served under /proxy/<port>/
    src = (f"_wavlens?model={quote(an.frame.name)}"
           f"&rows={','.join(map(str, have))}&names={quote(names)}")
    return [html.Div("audio  ·  WavLens — wheel = zoom · click = seek · "
                     "space = play · FFT = spectrogram", className="insp-h"),
            html.Iframe(src=src, className="wavlens",
                        style={"height": f"{122 + 88 * (len(have) + (len(have) > 1))}px"})]


def rows_view(an: Analysis, rows: list[int], thr, alpha) -> list:
    d, P, tv = an.decide(thr, alpha), an.norm(alpha), an.thr_vector(thr)
    cmap = class_colors(sorted(set(an.y.tolist())))
    out = []
    if len(rows) == 2:
        a, b = rows
        V = an.frame.V
        score = float(an.score_xy(V[[a]], V[[b]])[0, 0])
        cos = float(an.frame.Vn[a] @ an.frame.Vn[b])
        same = an.y[a] == an.y[b]
        out.append(html.Div(className="insp-item", children=[html.Div([
            _tag("same class", "ok") if same else _tag("different classes"),
            html.Span(f"{an.spec.methodology} score  "),
            html.B(f"{score:.4f}"),
            html.Span(f"   cosine  {cos:.4f}", className="field-label"),
        ], className="insp-head")]))
    roles = ["row", "col"] if len(rows) == 2 else [""]
    out += [_row_item(an, r, d, P, tv, cmap, role)
            for r, role in zip(rows, roles)]
    if len(rows) == 1:
        out.insert(0, _actions(an, rows[0]))
        out.append(_neighbours(an, rows[0], cmap))
    return out


def labels_view(an: Analysis, labels: list[str], thr, alpha) -> list:
    d = an.decide(thr, alpha)
    ops = cv.operating_points(an, thr, alpha)
    curves = {c.key: c for c in cv.class_curves(an, alpha)}
    counts, _, _ = mx.confusion(an, d)
    out = []
    for lbl in labels:
        if lbl not in an.labels:
            continue
        k = an.labels.index(lbl)
        c, op = curves.get(lbl), ops.get(lbl)
        n = int((an.true_idx == k).sum())
        kv = [html.Dt("members"), html.Dd(f"{n:,}"),
              html.Dt("prototypes"),
              html.Dd(str(an.centroids.get(lbl, np.zeros((0, 0))).shape[0]))]
        if c is not None:
            kv += [html.Dt("AUC"), html.Dd(f"{c.auc:.4f}"),
                   html.Dt("EER"), html.Dd(f"{c.eer:.2%}  at thr {c.eer_thr:.3f}"),
                   html.Dt("min DCF"),
                   html.Dd(f"{c.dcf:.3f}  at thr {c.dcf_thr:.3f}")]
        if op is not None:
            kv += [html.Dt("threshold"), html.Dd(f"{op[2]:.3f}"),
                   html.Dt("TPR / FPR"), html.Dd(f"{op[1]:.2%} / {op[0]:.2%}"),
                   html.Dt("DCF here"),
                   html.Dd(f"{float(cv.dcf(op[0], 1 - op[1])):.3f}")]
        row = counts[k]
        conf = [(an.labels[j], int(v)) for j, v in enumerate(row[:-1])
                if j != k and v > 0]
        conf.sort(key=lambda p: -p[1])
        kv += [html.Dt("correct"), html.Dd(f"{int(row[k]):,}"),
               html.Dt("rejected"), html.Dd(f"{int(row[-1]):,}"),
               html.Dt("lost to"),
               html.Dd(", ".join(f"{a} ({v})" for a, v in conf[:6]) or "—")]
        out.append(html.Div(className="insp-item", children=[
            html.Div([_tag(lbl, "ok")], className="insp-head"),
            html.Dl(kv, className="kv")]))
    if len(labels) == 2 and all(lb in an.labels for lb in labels):
        a, b = (an.labels.index(lb) for lb in labels)
        out.append(html.Div(className="insp-item", children=[html.Dl([
            html.Dt(f"{labels[0]} → {labels[1]}"), html.Dd(str(int(counts[a, b]))),
            html.Dt(f"{labels[1]} → {labels[0]}"), html.Dd(str(int(counts[b, a]))),
        ], className="kv")]))
    return out or [html.Div("nothing to show", className="note")]
