"""Shared helpers for figures_{pca,cm,dist,roc}.py adapter layers.

Split from figures.py per CLAUDE.md auto-split rule (270 line limit).
Holds the palette + state-extraction + reducer-factory utilities used by
every panel. Avoids circular imports: panel modules depend on this
module; this module depends on nothing local.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# numpy 2.x removed `trapz` → `trapezoid`. Local shim picks whichever.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


DEFAULT_PALETTE = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#FF9F43", "#4ECDC4", "#7DCEA0", "#BB8FCE", "#F1948A",
    "#85C1E2", "#F7DC6F", "#82E0AA", "#F5B7B1", "#AED6F1",
]


def palette_for(values: list[str]) -> dict[str, str]:
    return {v: DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)]
            for i, v in enumerate(sorted(values))}


def _restrict(df_m: pd.DataFrame, V: np.ndarray, sids: list[str],
              agg: Iterable[str] | None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not agg:
        return df_m, V, sids
    keep = set(str(s) for s in agg)
    mask = np.array([str(s) in keep for s in sids], dtype=bool)
    if not mask.any():
        return (df_m.iloc[0:0].copy(),
                np.empty((0, V.shape[1] if V.ndim == 2 else 0)),
                [])
    df_view = (df_m[df_m["session_id"].astype(str).isin(keep)]
               .reset_index(drop=True))
    return df_view, V[mask], [s for s, m in zip(sids, mask) if m]


def _centroids(state: dict | None) -> tuple[list[np.ndarray], list[str]]:
    if not state:
        return [], []
    raw = state.get("centroids", {}) or {}
    labels = sorted(raw.keys())
    rows = [np.asarray(raw[lbl], dtype=np.float64) for lbl in labels]
    return rows, labels


def _y_from_label_keys(df_m: pd.DataFrame,
                       label_keys: list[str] | None) -> np.ndarray:
    if not label_keys:
        return df_m["device_id"].astype(str).to_numpy()
    return (df_m[label_keys].astype(str)
            .agg(" | ".join, axis=1).to_numpy())


def _make_reducer(method: str, params: dict):
    """Return a callable `(vectors, _ignored_params) -> (N, 2)` that
    dispatches into backend.reduction_2d. The toy `build_pca_fig` calls
    `run_pca(joint, {"random_state": 42})` so we wrap to discard its
    args and inject our own."""
    from backend.reduction_2d import reduce_2d
    method = (method or "PCA").strip()

    def _runner(joint, _ignored_params):
        return reduce_2d(method, joint, params or {}).proj
    return _runner


def _run_pca(joint, _params):
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=42).fit_transform(joint)


def _norm_to_raw_thresholds(thr_norm_dict: dict, P_te,
                             rep, ordered_labels) -> dict:
    """DEPRECATED — kept as identity stub for any lingering caller.
    Slider thresholds now live in NORMALIZED [0, 1] end-to-end (see
    threshold_sync_plan.md P0). Direct passthrough; norm→raw round-trip
    removed because it degenerated on clipped methodologies."""
    return dict(thr_norm_dict or {})
