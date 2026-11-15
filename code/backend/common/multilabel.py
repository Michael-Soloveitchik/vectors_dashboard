"""Unified multilabel computation.

Spec: "maintain multilabel_set of all multilabels that have been
defined by controls or by hierarchical labeling".

A session's multilabel comes from one OR both of:
  * the user-selected key columns (e.g. ['voice_id', 'device_id']),
  * the hierarchical click-driven labels stored in HierarchyLabels.

When both are present we concatenate them with `join` so the user can
see the joint group identity. When only one is present we return its
labels directly. When neither is present every session gets the empty
sentinel "" (interpreted by the front-end as "no label, gray color").
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .data_access import SESSION_COL, build_multi_label_series

HIER_PREFIX = "h:"


def compute_multilabel_series(
    df: pd.DataFrame,
    label_keys: Sequence[str],
    hier_label_dict: dict[str, int | str] | None = None,
    join: str = " | ",
) -> pd.Series:
    """Return a per-row multilabel string for every row in df.

    `hier_label_dict` is a `session_id -> label_id` map produced by
    HierarchyLabels.labels(). Empty-string label values are skipped.
    """
    base = build_multi_label_series(df, label_keys, join=join)
    if not hier_label_dict:
        return base

    hier_strs: list[str] = []
    for sid in df[SESSION_COL].astype(str).tolist():
        v = hier_label_dict.get(sid, "")
        if v == "" or v is None:
            hier_strs.append("")
        else:
            hier_strs.append(f"{HIER_PREFIX}{v}")
    hier_series = pd.Series(hier_strs, index=df.index, dtype=object)

    if not label_keys:
        return hier_series
    merged = []
    for a, b in zip(base.tolist(), hier_series.tolist()):
        if a and b:
            merged.append(f"{a}{join}{b}")
        elif a:
            merged.append(a)
        else:
            merged.append(b)
    return pd.Series(merged, index=df.index, dtype=object)


def unique_multilabels(series: pd.Series) -> list[str]:
    """All distinct non-empty multilabel values, sorted for stability."""
    vals = sorted({s for s in series.tolist() if s})
    return vals


def restrict_to_session_ids(
    series: pd.Series,
    df: pd.DataFrame,
    session_ids: Sequence[str],
) -> pd.Series:
    """Reindex `series` to follow the explicit session_id order."""
    sid_to_idx = {sid: i for i, sid in enumerate(df[SESSION_COL].astype(str).tolist())}
    out = []
    for sid in session_ids:
        i = sid_to_idx.get(str(sid))
        out.append(series.iloc[i] if i is not None else "")
    return pd.Series(out, index=range(len(session_ids)), dtype=object)
