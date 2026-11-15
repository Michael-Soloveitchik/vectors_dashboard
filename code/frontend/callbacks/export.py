"""Exports — written to ./dumps/ and streamed to the browser.

Thresholds are stored AND applied in normalized space (the old CSV
compared normalized thresholds against raw scores).
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from dash import Input, Output, State, ctx, dcc
from dash.exceptions import PreventUpdate

from ..debug_log import trace
from ..engine import Engine, state
from ..engine import curves as cv
from ..ids import IDS

DUMPS = Path(__file__).resolve().parents[3] / "dumps"


def _unit(C: np.ndarray) -> np.ndarray:
    return C / np.maximum(np.linalg.norm(C, axis=-1, keepdims=True), 1e-12)


def _npz(an, thr, alpha):
    flat = {"labels": np.asarray(an.labels, dtype=object),
            "methodology": np.asarray(an.spec.methodology, dtype=object),
            "aggregator": np.asarray(an.spec.aggregator, dtype=object),
            "norm_alpha": np.asarray(float(alpha if alpha is not None else 1.0))}
    for lbl, t in zip(an.labels, an.thr_vector(thr)):
        if lbl in an.centroids:
            flat[f"centroids/{lbl}"] = _unit(an.centroids[lbl]).astype(np.float32)
        flat[f"thresholds/{lbl}"] = np.asarray(float(t))
    buf = io.BytesIO()
    np.savez(buf, **flat)
    (DUMPS / "vectors.npz").write_bytes(buf.getvalue())
    return dict(content=base64.b64encode(buf.getvalue()).decode("ascii"),
                filename="vectors.npz", base64=True)


def _csv(an, thr, alpha):
    d, P = an.decide(thr, alpha), an.norm(alpha)
    tv = an.thr_vector(thr)
    out = an.frame.df.copy()
    out["true_class"] = an.y
    out["pred_class"] = [an.labels[p] if p >= 0 else None for p in d.pred]
    out["pred_score_raw"] = [an.raw[i, p] if p >= 0 else np.nan
                             for i, p in enumerate(d.pred)]
    out["pred_score_norm"] = [P[i, p] if p >= 0 else np.nan
                              for i, p in enumerate(d.pred)]
    out["pred_threshold_norm"] = [tv[p] if p >= 0 else np.nan for p in d.pred]
    out["accepted_by_own_class"] = d.own_ok
    out["correct"] = d.correct
    out["excluded_from_fit"] = ~an.fit_mask
    out["votes_norm"] = [json.dumps({lbl: round(float(v), 5)
                                     for lbl, v in zip(an.labels, row)})
                         for row in P]
    out["methodology"] = an.spec.methodology
    out["aggregator"] = an.spec.aggregator
    out["label_keys"] = " | ".join(an.spec.label_keys)
    out["norm_alpha"] = float(alpha if alpha is not None else 1.0)
    out.to_csv(DUMPS / "predictions.csv", index=False)
    return dcc.send_data_frame(out.to_csv, "predictions.csv", index=False)


def _json(an, thr, alpha):
    classes = []
    ops = cv.operating_points(an, thr, alpha)
    curves = {c.key: c for c in cv.class_curves(an, alpha)}
    for lbl, t in zip(an.labels, an.thr_vector(thr)):
        k = an.labels.index(lbl)
        members = np.flatnonzero(an.true_idx == k)
        classes.append({
            "label": lbl, "n_members": int(members.size),
            "centroids": _unit(an.centroids[lbl]).tolist()
            if lbl in an.centroids else [],
            "threshold_norm": float(t),
            "operating_point": ({"fpr": ops[lbl][0], "tpr": ops[lbl][1]}
                                if lbl in ops else None),
            "auc": curves[lbl].auc if lbl in curves else None,
            "eer": curves[lbl].eer if lbl in curves else None,
            "eer_threshold_norm": curves[lbl].eer_thr if lbl in curves else None,
            "min_dcf": curves[lbl].dcf if lbl in curves else None,
            "min_dcf_threshold_norm": curves[lbl].dcf_thr
            if lbl in curves else None,
            "member_session_ids": [an.frame.sids[i] for i in members]})
    doc = {"meta": {**an.spec.to_dict(),
                    "norm_alpha": float(alpha if alpha is not None else 1.0)},
           "classes": classes}
    text = json.dumps(doc, indent=2)
    (DUMPS / "centroids.json").write_text(text)
    return dict(content=text, filename="centroids.json")


def register(app, engine: Engine) -> None:
    @app.callback(
        Output(IDS.download, "data"),
        Input(IDS.btn_dump_npz, "n_clicks"), Input(IDS.btn_dump_csv, "n_clicks"),
        Input(IDS.btn_dump_json, "n_clicks"),
        State(IDS.store_fit, "data"), State(IDS.store_thr, "data"),
        State(IDS.ctrl_alpha, "value"), prevent_initial_call=True)
    @trace
    def _dump(_a, _b, _c, fit, thr, alpha):
        if not fit or fit.get("error") or not ctx.triggered[0]["value"]:
            raise PreventUpdate
        if not state.thr_synced(fit, thr):
            raise PreventUpdate        # never export against stale thresholds
        an = engine.analysis(fit["spec"])
        thr = state.thr_values(thr)
        DUMPS.mkdir(exist_ok=True)
        fn = {IDS.btn_dump_npz: _npz, IDS.btn_dump_csv: _csv,
              IDS.btn_dump_json: _json}[ctx.triggered_id]
        return fn(an, thr, alpha)
