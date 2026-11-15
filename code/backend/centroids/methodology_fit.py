"""Methodology-driven centroid extraction.

New algorithmic path: replaces the sklearn-GMM `fit_centroids` route. A
single `Methodology()` instance is fit on (X, y) across all selected
labels, then `rep.get_centroids()` yields per-class prototypes via the
solver chain (A: closed-form mean, B: Weiszfeld median, F: discrete medoid).

The result is wrapped in a `CentroidSet` so the rest of the dashboard
(ROC, projection, ellipses) keeps using the same data structure. Sample
covariance is computed per class on the training subset for ellipse display.

The user toggle `n_per_label` is currently ignored — every methodology
runs with `n_clusters=1`. This will be revisited later.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..common.data_access import SESSION_COL, VECTOR_COL
from .gmm import CentroidSet


def _stack_label_group(group: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    raw_vecs = group[VECTOR_COL].to_numpy()
    raw_sids = group[SESSION_COL].to_numpy()
    rows: list[np.ndarray] = []
    sids: list[str] = []
    for sid, v in zip(raw_sids, raw_vecs):
        arr = np.asarray(v).reshape(-1)
        if arr.size == 0:
            continue
        rows.append(arr)
        sids.append(str(sid))
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(rows, axis=0).astype(np.float32, copy=False), sids


def _holdout_split(
    n_total: int, holdout_fraction: float, rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    if holdout_fraction <= 0 or n_total < 2:
        return np.arange(n_total), np.empty(0, dtype=int)
    n_eval = max(1, int(n_total * holdout_fraction))
    n_fit = max(1, n_total - n_eval)
    n_eval = n_total - n_fit
    perm = rng.permutation(n_total)
    return perm[:n_fit], perm[n_fit:]


def _sample_covariance(vecs: np.ndarray) -> np.ndarray:
    """Per-class sample covariance with a tiny ridge for ellipse display."""
    n, d = vecs.shape
    if n < 2:
        return np.eye(d, dtype=np.float32) * 1e-4
    cov = np.cov(vecs, rowvar=False, bias=False)
    cov = np.atleast_2d(cov).astype(np.float32, copy=False)
    cov.flat[:: d + 1] += 1e-4
    return cov


def fit_centroids_via_methodology(
    df: pd.DataFrame,
    label_series: pd.Series,
    methodology: str,
    aggregator: str | Callable,
    method_args: dict | None = None,
    n_per_label: dict[str, int] | int = 1,
    random_state: int = 42,
    progress_callback=None,
    max_seconds: float | None = None,
    holdout_fraction: float = 0.0,
) -> CentroidSet:
    """Fit one global `Methodology()` over all labels — toy-aligned.

    Single fit on stacked (X, y), then `rep.get_centroids()` returns
    `{label: (k, D)}` for every prototype-building rep. Discriminative
    reps (CosFace, KNN, ArcCosine*) require all classes together; the
    fallback for those is `_centroid_under_score` per class, executed
    on the same fitted rep.

    `method_args` is forwarded to the underlying Representation. When
    omitted, defaults to `{"n_clusters": int(n_per_label)}` for reps
    that accept it (NCM, MultiProxyAnchor, ACPAN, ...).

    `holdout_fraction` defaults to 0 to match the toy / run.ipynb
    pattern (full data fit, no validation split). Pass >0 to enable a
    held-out eval set per label.
    """
    import inspect
    import time
    from backend.representations import Methodology, METHOD_REGISTRY

    if methodology not in METHOD_REGISTRY:
        raise ValueError(
            f"unknown methodology={methodology!r}. "
            f"Valid: {sorted(METHOD_REGISTRY)}"
        )

    t0 = time.perf_counter()
    groups = list(df.groupby(label_series, sort=True))
    valid_groups = [(lbl, g) for lbl, g in groups
                    if lbl not in (None, "") and len(g) >= 2]
    if not valid_groups:
        return CentroidSet(
            centroids={}, covariances={}, weights={},
            member_session_ids={}, n_per_label={}, eval_session_ids={},
        )

    rng = np.random.RandomState(random_state)
    fit_X_parts: list[np.ndarray] = []
    fit_y_parts: list[np.ndarray] = []
    members: dict[str, list[str]] = {}
    eval_members: dict[str, list[str]] = {}
    fit_vecs_per_label: dict[str, np.ndarray] = {}

    for label, group in valid_groups:
        all_vecs, all_sids = _stack_label_group(group)
        if all_vecs.shape[0] < 2:
            continue
        fit_idx, eval_idx = _holdout_split(
            all_vecs.shape[0], holdout_fraction, rng,
        )
        vecs = all_vecs[fit_idx]
        fit_X_parts.append(vecs)
        fit_y_parts.append(np.full(vecs.shape[0], str(label), dtype=object))
        members[str(label)] = all_sids
        eval_members[str(label)] = [all_sids[i] for i in eval_idx]
        fit_vecs_per_label[str(label)] = vecs

    if not fit_X_parts:
        return CentroidSet(
            centroids={}, covariances={}, weights={},
            member_session_ids={}, n_per_label={}, eval_session_ids={},
        )

    X = np.concatenate(fit_X_parts, axis=0)
    y = np.concatenate(fit_y_parts, axis=0)
    # n_clusters = max requested across labels (rep stores per-class).
    if isinstance(n_per_label, dict):
        k_global = max(int(n_per_label.get(lbl, 1)) for lbl in members) or 1
    else:
        k_global = max(1, int(n_per_label))

    m_args = dict(method_args or {})
    cls_obj = METHOD_REGISTRY[methodology]
    try:
        sig = inspect.signature(cls_obj.__init__)
        accepted = set(sig.parameters.keys())
    except (TypeError, ValueError):
        accepted = set()
    if "n_clusters" in accepted:
        m_args.setdefault("n_clusters", k_global)
    m_args = {kk: vv for kk, vv in m_args.items()
              if not accepted or kk in accepted}

    rep = Methodology(normalize_data_on_sphere=True)
    if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
        return CentroidSet(
            centroids={}, covariances={}, weights={},
            member_session_ids=members, n_per_label={},
            eval_session_ids=eval_members,
        )
    rep.fit(X, y, method=methodology, method_args=m_args,
            aggregator=aggregator)

    rep_centroids = rep.get_centroids() or {}
    centroids: dict[str, np.ndarray] = {}
    covariances: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    n_used: dict[str, int] = {}

    for idx, lbl in enumerate(sorted(fit_vecs_per_label.keys())):
        vecs = fit_vecs_per_label[lbl]
        proto = rep_centroids.get(lbl)
        if proto is None or (hasattr(proto, "__len__") and len(proto) == 0):
            try:
                solo = rep._centroid_under_score(vecs, aggregator=aggregator)
                proto = np.atleast_2d(np.asarray(solo, dtype=np.float32))
            except Exception as exc:
                print(f"[methodology-fit] {lbl!r} no centroid path: {exc}",
                      flush=True)
                continue
        proto = np.asarray(proto, dtype=np.float32)
        if proto.ndim == 1:
            proto = proto[None, :]
        centroids[lbl] = proto
        covariances[lbl] = _sample_covariance(vecs)[None, :, :].repeat(
            proto.shape[0], axis=0)
        weights[lbl] = np.full(proto.shape[0],
                               1.0 / proto.shape[0], dtype=np.float32)
        n_used[lbl] = int(proto.shape[0])
        if progress_callback is not None:
            progress_callback(lbl, idx)

    return CentroidSet(
        centroids=centroids,
        covariances=covariances,
        weights=weights,
        member_session_ids=members,
        n_per_label=n_used,
        eval_session_ids=eval_members,
    )


__all__ = ["fit_centroids_via_methodology"]
