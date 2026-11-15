"""Fit one ``FitSpec`` on one frame → ``Analysis`` (errors become values,
so a failing methodology shows its reason in the UI and is cached like
any other result instead of being refitted by every panel)."""
from __future__ import annotations

import warnings

import numpy as np

from .concurrency import freeze
from .data import ModelFrame


def _skip_mask(frame: ModelFrame, label_keys, y, skip) -> np.ndarray:
    """True = keep for fit. A row is skipped if its class label, any of its
    label-column values or its session id is in ``skip``, or if it matches
    a ``column=value`` entry (any column: gender=female, language=fa…)."""
    keep = np.ones(frame.n, dtype=bool)
    if not skip:
        return keep
    s = set(skip)
    keep &= ~np.isin(y.astype(str), list(s))
    for col in list(label_keys) + ["session_id"]:
        if col in frame.df.columns:
            keep &= ~frame.df[col].astype(str).isin(s).to_numpy()
    for entry in s:
        col, eq, val = entry.partition("=")
        if eq and col in frame.df.columns:
            keep &= ~(frame.df[col].astype(str) == val).to_numpy()
    return keep


def _centroids(rep, frame, y, keep, labels, aggregator) -> dict[str, np.ndarray]:
    try:
        cs = {str(k): np.atleast_2d(np.asarray(v, dtype=np.float64))
              for k, v in (rep.get_centroids() or {}).items()}
    except Exception:
        cs = {}
    if cs:
        return cs
    out = {}
    for lbl in labels:   # non-prototype reps: score-medoid under aggregator
        m = keep & (y == lbl)
        if not m.any():
            continue
        try:
            out[lbl] = np.atleast_2d(np.asarray(
                rep._centroid_under_score(frame.V[m], aggregator=aggregator),
                dtype=np.float64))
        except Exception:
            out[lbl] = np.atleast_2d(frame.Vn[m].mean(axis=0))
    return out


def fit_analysis(spec, frame: ModelFrame):
    from backend.representations import METHOD_REGISTRY, Methodology
    from .analysis import MAX_CLASSES, MULTI_CENTROID_ARG, Analysis
    missing = [k for k in spec.label_keys if k not in frame.df.columns]
    y = frame.labels(spec.label_keys)
    keep = _skip_mask(frame, spec.label_keys, y, spec.skip)
    an = Analysis(spec=spec, frame=frame, y=y, fit_mask=keep)
    classes = sorted(set(y[keep].tolist()))
    cls = METHOD_REGISTRY.get(spec.methodology)
    if not spec.label_keys:
        an.error = "pick at least one label column"
    elif missing:
        an.error = f"label column(s) not in this model: {', '.join(missing)}"
    elif len(classes) < 2:
        an.error = "need at least 2 classes after skipping"
    elif len(classes) > MAX_CLASSES:
        an.error = (f"{len(classes)} classes — pick a coarser label "
                    f"(max {MAX_CLASSES})")
    elif cls is None:
        an.error = f"unknown methodology {spec.methodology!r}"
    if an.error:
        return an
    arg = MULTI_CENTROID_ARG.get(spec.methodology)
    method_args = {arg: spec.n_global} if arg else {}
    hints = dict(spec.per_class_n) if arg and spec.per_class_n else None
    rep = Methodology(normalize_data_on_sphere=True)
    try:
        with warnings.catch_warnings():      # solver chatter (SRC lasso …)
            warnings.simplefilter("ignore")  # would flood the notebook cell
            rep.fit(frame.V[keep], y[keep], method=spec.methodology,
                    method_args=method_args, aggregator=spec.aggregator,
                    hints=hints)
            _, _, raw = rep.predict(frame.V)
        raw = np.asarray(raw, dtype=np.float64)
        if np.isnan(raw).any():
            raise FloatingPointError("NaN scores")
    except Exception as exc:  # surface the reason in the UI, keep serving
        an.error = f"{spec.methodology} failed: {type(exc).__name__}: {exc}"
        return an
    an.rep = rep
    an.raw = freeze(raw)
    an.labels = [str(c) for c in rep.class_order_fn()]
    pos = {lbl: i for i, lbl in enumerate(an.labels)}
    an.true_idx = freeze(np.array([pos.get(str(v), -1) for v in y], dtype=int))
    an.centroids = _centroids(rep, frame, y, keep, an.labels, spec.aggregator)
    return an
