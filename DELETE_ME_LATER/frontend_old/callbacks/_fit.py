"""Auto-fit callback + methodology rebuild helpers."""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from dash import Input, Output, no_update

from ..debug_log import trace
from ..ids import IDS


def _stack(df_m: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    V = np.stack([np.asarray(v).reshape(-1)
                  for v in df_m["centroid"].to_numpy()])
    sids = [str(s) for s in df_m["session_id"].astype(str).tolist()]
    return V.astype(np.float64), sids


# Whitelist of (methodology class name → __init__ kwarg) for true
# multi-centroid splits. Note: KNN.k / ArcCosineKNN.k are *neighbor*
# counts, not centroids — excluded here so the per-class panel doesn't
# mislead.
_MULTI_CENTROID_REGISTRY = {
    "NCM":              "n_clusters",   # spherical k-means / median-init
    "VMFMixture":       "n_components", # vMF EM mixture per class
    "MultiProxyAnchor": "n_proxies",    # spherical k-means proxies
}


def _multi_arg(cls):
    """Return the constructor kwarg controlling multi-centroid count, or
    None if the methodology doesn't support per-class multi-prototype
    splits."""
    return _MULTI_CENTROID_REGISTRY.get(cls.__name__)


def _fit_methodology(V: np.ndarray, y: np.ndarray, methodology: str,
                     aggregator: str, n_global: int = 1,
                     per_class_n: dict | None = None) -> tuple[object, dict]:
    """Global fit + return (rep, centroid_state).

    `n_global`   — global multi-centroid count (e.g. NCM.n_clusters).
    `per_class_n`— {label: int} forwarded as `hints` so individual
                   classes override the global count.
    """
    from backend.representations import Methodology, METHOD_REGISTRY
    cls = METHOD_REGISTRY[methodology]
    arg_name = _multi_arg(cls)
    method_args = {arg_name: int(n_global)} if arg_name else {}
    hints = {str(k): int(v) for k, v in (per_class_n or {}).items()
             if v} if arg_name else None
    rep = Methodology(normalize_data_on_sphere=True)
    rep.fit(V, y, method=methodology, method_args=method_args,
            aggregator=aggregator, hints=hints)
    cs = rep.get_centroids() or {}
    if not cs:
        labels = sorted(set(str(v) for v in y))
        cs = {}
        for lbl in labels:
            mask = (y == lbl)
            if int(mask.sum()) < 1:
                continue
            cs[lbl] = np.atleast_2d(rep._centroid_under_score(
                V[mask], aggregator=aggregator))
    state = {
        "centroids": {lbl: np.asarray(c, dtype=np.float64).tolist()
                      for lbl, c in cs.items()},
        "methodology": methodology,
        "aggregator": aggregator,
        "method_args": method_args,
    }
    return rep, state


def build_rep_from_state(state: dict):
    """Re-instantiate fitted Methodology from stored centroids."""
    from backend.representations import Methodology, METHOD_REGISTRY
    cls = METHOD_REGISTRY[state.get("methodology", "NCM")]
    args = dict(state.get("method_args") or {})
    try:
        impl = cls(**args)
    except TypeError:
        impl = cls()
    labels = sorted(state.get("centroids", {}).keys())
    means_list, owner_list = [], []
    for idx, lbl in enumerate(labels):
        c = np.asarray(state["centroids"][lbl], dtype=np.float64)
        means_list.append(c)
        owner_list.append(np.full(c.shape[0], idx))
    impl._centroids = np.concatenate(means_list, axis=0)
    impl._owner = np.concatenate(owner_list, axis=0)
    impl._labels = np.asarray(labels)
    impl._aggregator = state.get("aggregator", "mean")
    impl._fitted = True
    impl._labels.setflags(write=False)
    rep = Methodology(normalize_data_on_sphere=True)
    rep._impl = impl
    rep._method_name = state.get("methodology", "NCM")
    rep._method_args = args
    return rep


def _seed_thresholds(rep, V: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Init each class threshold to 0.0 in NORMALIZED-score space.

    Convention: thresholds live in the alpha-blended normalized score
    space [0, 1] used by the dist axis and ROC pair-norm.  0.0 means
    "no separation" (loose), 1.0 means "perfect separation" (tight).
    All 4 plots + the per-class slider read from the same store and
    must display the same value.
    """
    if not labels:
        return {}
    _, _, _ = rep.predict(V)
    return {lbl: 0.0 for lbl in labels}


def register(app, df: pd.DataFrame) -> None:
    @app.callback(
        Output(IDS.store_centroids, "data"),
        Output(IDS.store_thresholds, "data"),
        Input(IDS.ctrl_model, "value"),
        Input(IDS.ctrl_labels, "value"),
        Input(IDS.ctrl_methodology, "value"),
        Input(IDS.ctrl_aggregator, "value"),
        Input(IDS.ctrl_skip_classes, "value"),
        Input(IDS.ctrl_n_centroids, "value"),
        Input(IDS.store_per_class_n, "data"),
    )
    @trace
    def _autofit(model, label_keys, methodology, aggregator, skip,
                 n_global, per_class_n):
        if not (model and methodology and aggregator):
            return no_update, no_update
        df_full = df[df["model_name"] == model].reset_index(drop=True)
        if df_full.empty or not label_keys:
            return None, {}
        y_full = (df_full[label_keys].astype(str)
                  .agg(" | ".join, axis=1).to_numpy())
        # Skip semantics: a point is excluded from fit iff ANY of its
        # identifier-cell values matches the skip set. Check every
        # label_key column AND session_id (skip values are arbitrary
        # strings — class label, device id, or specific session).
        # Excluded points still receive predictions (in dumps + faded
        # on PCA) so we never lose them entirely.
        skip_set = set(str(s) for s in (skip or []))
        check_cols = list(label_keys) + (["session_id"]
                         if "session_id" in df_full.columns else [])
        if skip_set:
            keep_mask = np.ones(len(df_full), dtype=bool)
            for col in check_cols:
                col_vals = df_full[col].astype(str).to_numpy()
                keep_mask &= np.array([v not in skip_set for v in col_vals])
        else:
            keep_mask = np.ones(len(df_full), dtype=bool)
        excluded_sids = [str(s) for s, k in zip(
            df_full["session_id"].astype(str).to_numpy(), keep_mask) if not k]
        df_fit = df_full.iloc[keep_mask].reset_index(drop=True)
        y = y_full[keep_mask]
        if len(set(y)) < 2:
            print(f"[autofit] not enough classes after skip "
                  f"keep={int(keep_mask.sum())} excluded={len(excluded_sids)}",
                  flush=True)
            return None, {}
        V, _ = _stack(df_fit)
        try:
            rep, state = _fit_methodology(
                V, y, methodology, aggregator,
                n_global=int(n_global or 1),
                per_class_n=per_class_n or {},
            )
        except Exception as exc:
            print(f"[autofit] failed: {exc}", flush=True)
            return None, {}
        state["excluded_sids"] = excluded_sids
        state["per_class_n"] = dict(per_class_n or {})
        labels = sorted(state["centroids"].keys())
        thr = _seed_thresholds(rep, V, labels)
        # Cache fitted rep so non-centroid methodologies (KNN/SRC/kernel)
        # don't lose their training state on next render.
        try:
            V_fp = V.tobytes()[:64] + V.tobytes()[-64:]
        except Exception:
            V_fp = b""
        cache_key = _fit_cache_key(model, methodology, aggregator,
                                   int(n_global or 1),
                                   per_class_n or {},
                                   tuple(label_keys or ()),
                                   V_fp)
        _FITTED_REP_CACHE[cache_key] = rep
        if len(_FITTED_REP_CACHE) > 32:
            _FITTED_REP_CACHE.pop(next(iter(_FITTED_REP_CACHE)))
        print(f"[autofit] {methodology}/{aggregator} n={n_global} "
              f"hints={len(per_class_n or {})} → {len(labels)} labels  "
              f"excluded={len(excluded_sids)}  cached_reps={len(_FITTED_REP_CACHE)}",
              flush=True)
        return state, thr

    @app.callback(
        Output(IDS.ctrl_skip_classes, "options"),
        Input(IDS.ctrl_model, "value"),
        Input(IDS.ctrl_labels, "value"),
    )
    @trace
    def _populate_skip_options(model, label_keys):
        if not model or not label_keys:
            return []
        df_m = df[df["model_name"] == model]
        if df_m.empty:
            return []
        # Skip values are arbitrary strings — collect every distinct
        # cell value across label_keys columns AND session_id so the
        # user can target either a class label, a single device, or a
        # specific session by name.
        cols = list(label_keys) + (["session_id"]
                    if "session_id" in df_m.columns else [])
        vals: set[str] = set()
        for col in cols:
            vals.update(df_m[col].astype(str).unique().tolist())
        return [{"label": v, "value": v} for v in sorted(vals)]


# Live fitted-rep cache. Keyed on a fingerprint of everything that
# changes the score function: methodology + aggregator + multi-centroid
# args + per-class hints + V identity + alpha.  Each key holds the
# actual fitted `Methodology` instance so non-centroid methodologies
# (KNN, SRC, kernel) keep their internal training data — letting CM /
# PCA / dist / ROC compute scores without re-fitting per render.
_FITTED_REP_CACHE: dict[tuple, object] = {}


def _fit_cache_key(model: str | None, methodology: str, aggregator: str,
                   n_global: int, per_class_n: dict | None,
                   label_keys: tuple, V_fp: bytes) -> tuple:
    pcn = tuple(sorted((per_class_n or {}).items()))
    return (str(model), str(methodology), str(aggregator),
            int(n_global), pcn, tuple(label_keys or ()), V_fp)


def get_cached_rep(model: str, state: dict, V: np.ndarray,
                   label_keys: list | None,
                   df_m: "pd.DataFrame | None" = None) -> object | None:
    """Lookup fitted rep instance keyed on autofit inputs.  Returns the
    instance directly (not a copy) — callers may mutate `impl.NORM_ALPHA`
    safely as that is the only mutable knob the renderers touch.

    On cache miss, when `df_m` is provided, refit inline. This is the
    safe path for methodologies whose impl state is not just centroids
    (ACPAN proxies, KNN training set, SRC dictionary, LabelPropagation
    graph, BinghamGeodesic concentration). `build_rep_from_state` only
    rehydrates `_centroids/_owner/_labels` which is incomplete for
    those, leading to AxisError when impl methods read None attrs.
    """
    if not state or not model:
        return None
    method = state.get("methodology", "?")
    aggregator = state.get("aggregator", "?")
    method_args = state.get("method_args") or {}
    arg_name = next((k for k in ("n_clusters", "n_components",
                                  "n_proxies", "k")
                     if k in method_args), None)
    n_global = int(method_args.get(arg_name, 1)) if arg_name else 1
    pcn = state.get("per_class_n") or {}
    V_fp = (V.tobytes()[:64] + V.tobytes()[-64:]) if V.size else b""
    key = _fit_cache_key(model, method, aggregator, n_global,
                         pcn, tuple(label_keys or ()), V_fp)
    hit = _FITTED_REP_CACHE.get(key)
    if hit is not None or df_m is None:
        return hit
    try:
        if not label_keys:
            return None
        y = (df_m[list(label_keys)].astype(str)
             .agg(" | ".join, axis=1).to_numpy())
        rep, _ = _fit_methodology(V, y, method, aggregator,
                                  n_global=n_global, per_class_n=pcn)
        _FITTED_REP_CACHE[key] = rep
        if len(_FITTED_REP_CACHE) > 32:
            _FITTED_REP_CACHE.pop(next(iter(_FITTED_REP_CACHE)))
        return rep
    except Exception as exc:
        print(f"[get_cached_rep] inline refit failed "
              f"({method}/{aggregator}): {exc}", flush=True)
        return None


_CM_SCORE_CACHE: dict[tuple, "np.ndarray"] = {}


def cached_score_xy_matrix(rep, V: np.ndarray, model: str,
                           methodology: str, alpha: float | None) -> "np.ndarray":
    """Full N×N pairwise score matrix via `rep.score_XY(V, V)` —
    methodology's pair-primitive (NOT raw V@V.T cosine). Cached
    per (model, methodology, alpha, V fingerprint). Slicing on
    selection happens downstream."""
    V_fp = (V.tobytes()[:64] + V.tobytes()[-64:]) if V.size else b""
    key = (str(model), str(methodology),
           float(alpha) if alpha is not None else None, V_fp)
    hit = _CM_SCORE_CACHE.get(key)
    if hit is not None:
        return hit
    if alpha is not None:
        try:
            rep.impl.NORM_ALPHA = float(alpha)
        except Exception:
            pass
    M = np.asarray(rep.score_XY(V, V), dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != V.shape[0]:
        raise ValueError(f"score_XY returned shape {M.shape}")
    _CM_SCORE_CACHE[key] = M
    if len(_CM_SCORE_CACHE) > 16:
        _CM_SCORE_CACHE.pop(next(iter(_CM_SCORE_CACHE)))
    return M


_PREDICT_CACHE: dict[tuple, tuple] = {}


def cached_predict(rep, V: np.ndarray, methodology: str,
                   alpha: float | None = None,
                   thresholds_key: tuple = ()) -> tuple:
    """Cache `rep.predict(V)` keyed on (V hash, methodology, alpha, thr).

    Side effect: writes/reads from `_PREDICT_CACHE`. Bound by 16 entries
    so a long session of model/methodology toggles doesn't leak."""
    key = (V.tobytes()[:64] + V.tobytes()[-64:],     # boundary fingerprint
           V.shape, V.dtype.str, methodology,
           float(alpha) if alpha is not None else None,
           thresholds_key)
    hit = _PREDICT_CACHE.get(key)
    if hit is not None:
        return hit
    if alpha is not None:
        try:
            rep.impl.NORM_ALPHA = float(alpha)
        except Exception:
            pass
    out = rep.predict(V)
    if len(_PREDICT_CACHE) > 16:
        _PREDICT_CACHE.pop(next(iter(_PREDICT_CACHE)))
    _PREDICT_CACHE[key] = out
    return out


__all__ = ["register", "build_rep_from_state", "_stack",
           "cached_predict", "get_cached_rep"]
