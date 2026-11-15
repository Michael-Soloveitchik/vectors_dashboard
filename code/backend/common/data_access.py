"""Session-id-stable accessors for the dataframe.

EVERYTHING upstream addresses points by `session_id` because:
  * the `aggregated_points` set churns on every click,
  * sort orders rearrange rows in the confusion matrix,
  * a fresh reduction may reshuffle the order entirely.

So the only stable identity for a row is its session_id. Numeric
indices in `df` are NEVER trusted across two different views.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

VECTOR_COL = "centroid"
SESSION_COL = "session_id"


def get_label_keys(df: pd.DataFrame, exclude: Sequence[str] = ()) -> list[str]:
    """All columns of primitive (non-vector) type, in df column order.

    Anything that ends up as a numpy array, list, dict, or tuple is
    treated as a vector and skipped — see the `vectors are not
    considered!` rule from the dashboard spec.
    """
    out: list[str] = []
    excluded = set(exclude) | {VECTOR_COL}
    for col in df.columns:
        if col in excluded:
            continue
        sample = df[col].iloc[0] if len(df) else None
        if isinstance(sample, (np.ndarray, list, tuple, dict)):
            continue
        out.append(col)
    return out


def session_to_row_index(df: pd.DataFrame) -> dict[str, int]:
    """Map session_id -> integer row index in the *current* df ordering.

    The mapping is rebuilt every call because callers pass filtered /
    re-sorted dataframes; cache it on the call site, not here.
    """
    return {sid: i for i, sid in enumerate(df[SESSION_COL].to_numpy())}


def get_vectors_for_sessions(
    df: pd.DataFrame,
    session_ids: Iterable[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Stack the centroid vectors for the given session_ids in caller order.

    Passing `session_ids=None` returns every row in df order, which the
    aggregation logic uses as the implicit `aggregated_points == [] ->
    everything` semantics. The order of the returned array follows the
    order of `session_ids`, NOT the row order of df.

    Returns (vectors of shape (N, D), list of session_ids actually used).
    Rows whose centroid is missing or has zero size are silently dropped.
    """
    if session_ids is None:
        sids = df[SESSION_COL].tolist()
    else:
        sids = list(session_ids)
    if not sids:
        return np.empty((0, 0), dtype=np.float32), []

    idx_map = session_to_row_index(df)
    rows: list[np.ndarray] = []
    used: list[str] = []
    for sid in sids:
        i = idx_map.get(sid)
        if i is None:
            continue
        v = df.iloc[i][VECTOR_COL]
        arr = np.asarray(v).reshape(-1)
        if arr.size == 0:
            continue
        rows.append(arr)
        used.append(sid)
    if not rows:
        return np.empty((0, 0), dtype=np.float32), []

    dim = rows[0].size
    if any(r.size != dim for r in rows):
        raise ValueError(
            "vectors_for_sessions: mixed dims encountered. "
            "Filter to a single model_name first."
        )
    return np.stack(rows, axis=0).astype(np.float32, copy=False), used


def build_multi_label_series(
    df: pd.DataFrame,
    label_keys: Sequence[str],
    join: str = " | ",
) -> pd.Series:
    """Join several label columns into one per-row composite label.

    Empty `label_keys` -> a Series of empty strings of length len(df).
    The empty string is the sentinel for "no labels chosen", which the
    front-end interprets as a single grey color group.
    """
    if not label_keys:
        return pd.Series([""] * len(df), index=df.index, dtype=object)
    parts = [df[k].astype(str) for k in label_keys]
    out = parts[0].copy()
    for p in parts[1:]:
        out = out.str.cat(p, sep=join)
    return out
