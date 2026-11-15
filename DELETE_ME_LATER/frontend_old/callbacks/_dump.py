"""Dump callbacks — vectors.npz + predictions.csv.

Both write to ``./dumps/`` under the project root AND stream the file
back to the browser via ``dcc.Download``.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Input, Output, State, no_update, dcc

from ..debug_log import trace
from ..ids import IDS
from ._fit import build_rep_from_state, _stack

DUMPS = Path(__file__).resolve().parents[3] / "dumps"


def register(app, df: pd.DataFrame) -> None:
    @app.callback(
        Output(IDS.download_npz, "data"),
        Input(IDS.btn_dump_npz, "n_clicks"),
        State(IDS.store_centroids, "data"),
        State(IDS.store_thresholds, "data"),
        prevent_initial_call=True,
    )
    @trace
    def _dump_npz(n, state, thresholds):
        if not n or not state:
            return no_update
        DUMPS.mkdir(exist_ok=True)
        labels = sorted(state.get("centroids", {}).keys())
        flat = {
            "labels":       np.asarray(labels, dtype=object),
            "methodology":  np.asarray(state.get("methodology", "?"),
                                       dtype=object),
            "aggregator":   np.asarray(state.get("aggregator", "?"),
                                       dtype=object),
        }
        thr = thresholds or {}
        for lbl in labels:
            arr = np.asarray(state["centroids"][lbl], dtype=np.float32)
            n_arr = np.linalg.norm(arr, axis=-1, keepdims=True)
            arr = arr / np.maximum(n_arr, 1e-12)
            flat[f"centroids/{lbl}"] = arr.astype(np.float32)
            flat[f"thresholds/{lbl}"] = np.asarray(
                float(thr.get(lbl, 0.0)), dtype=np.float64)
        disk = DUMPS / "vectors.npz"
        np.savez(disk, **flat)
        buf = io.BytesIO()
        np.savez(buf, **flat)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        print(f"[dump] wrote {disk}", flush=True)
        return dict(content=b64, filename="vectors.npz", base64=True)

    @app.callback(
        Output(IDS.download_csv, "data"),
        Input(IDS.btn_dump_csv, "n_clicks"),
        State(IDS.store_centroids, "data"),
        State(IDS.store_thresholds, "data"),
        State(IDS.ctrl_model, "value"),
        State(IDS.ctrl_labels, "value"),
        prevent_initial_call=True,
    )
    @trace
    def _dump_csv(n, state, thresholds, model, label_keys):
        if not n or not state or not model:
            return no_update
        DUMPS.mkdir(exist_ok=True)
        df_m = df[df["model_name"] == model].reset_index(drop=True)
        if df_m.empty:
            return no_update
        V, sids = _stack(df_m)
        rep = build_rep_from_state(state)
        labels = sorted(state.get("centroids", {}).keys())
        full_thr = {lbl: float((thresholds or {}).get(lbl, 0.0))
                    for lbl in labels}
        cls_arr, score_arr, votes = rep.predict(V, thresholds=full_thr)
        ordered = list(rep.class_order_fn())
        excl = set(state.get("excluded_sids") or [])
        preds_df = pd.DataFrame({
            "session_id":  [str(s) for s in sids],
            "pred_cls":    list(cls_arr),
            "pred_score":  list(score_arr),
            "pred_votes":  [
                json.dumps({lbl: float(votes[j, k])
                            for k, lbl in enumerate(ordered)})
                for j in range(len(sids))
            ],
            "threshold":   [
                float(full_thr.get(str(cls_arr[j]), 0.0))
                if cls_arr[j] is not None else 0.0
                for j in range(len(sids))
            ],
            "excluded_from_fit": [str(s) in excl for s in sids],
            "methodology": state.get("methodology", "?"),
            "aggregator":  state.get("aggregator", "?"),
            "label_keys":  " | ".join(label_keys or []),
            "model":       model,
        })
        out_df = df_m.copy()
        out_df["session_id"] = out_df["session_id"].astype(str)
        out_df = out_df.merge(preds_df, on="session_id", how="left",
                              suffixes=("", "_pred"))
        disk = DUMPS / "predictions.csv"
        out_df.to_csv(disk, index=False)
        print(f"[dump] wrote {disk} ({len(out_df)} rows)", flush=True)
        return dcc.send_data_frame(out_df.to_csv, "predictions.csv",
                                   index=False)

    @app.callback(
        Output(IDS.download_centroids_json, "data"),
        Input(IDS.btn_dump_centroids_json, "n_clicks"),
        State(IDS.store_centroids, "data"),
        State(IDS.store_thresholds, "data"),
        State(IDS.ctrl_model, "value"),
        State(IDS.ctrl_labels, "value"),
        prevent_initial_call=True,
    )
    @trace
    def _dump_centroids_json(n, state, thresholds, model, label_keys):
        if not n or not state or not model or not label_keys:
            return no_update
        DUMPS.mkdir(exist_ok=True)
        df_m = df[df["model_name"] == model].reset_index(drop=True)
        if df_m.empty:
            return no_update
        V, sids = _stack(df_m)
        y = (df_m[label_keys].astype(str)
             .agg(" | ".join, axis=1).to_numpy())
        labels = sorted(state.get("centroids", {}).keys())
        thr = thresholds or {}
        out = {"meta": {
            "model": model,
            "label_keys": list(label_keys or []),
            "methodology": state.get("methodology", "?"),
            "aggregator": state.get("aggregator", "?")},
            "classes": []}
        for lbl in labels:
            mask = (y == lbl)
            members = [str(s) for s, m in zip(sids, mask) if m]
            V_lbl = V[mask]
            cov = (np.cov(V_lbl, rowvar=False).tolist()
                   if V_lbl.shape[0] > 1 else None)
            cents = np.asarray(state["centroids"][lbl],
                               dtype=np.float64).tolist()
            out["classes"].append({
                "label": str(lbl),
                "n_members": int(mask.sum()),
                "centroids": cents,
                "covariance": cov,
                "threshold": float(thr.get(lbl, 0.0)),
                "member_session_ids": members,
            })
        disk = DUMPS / "centroids.json"
        disk.write_text(json.dumps(out, indent=2))
        print(f"[dump] wrote {disk}", flush=True)
        return dict(content=json.dumps(out, indent=2),
                    filename="centroids.json")


__all__ = ["register"]
