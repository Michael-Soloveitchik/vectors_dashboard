"""Shared-state contracts between callbacks — the synchronization rules.

Every store that refers to data carries what it belongs to, and every
consumer checks it, so a slow or out-of-order callback can never paint
one model's selection / thresholds onto another model's figures:

    store_fit    {"spec", "labels", "error", "token"}   token = hash(spec)
    store_thr    {"token", "values": {label: thr}}      valid iff token == fit token
    store_sel    {"model", "clauses": {src: …},        valid iff model == fit model
                  "mode", "seq"}
    store_focus  {"model", "rows" | "labels"}           valid iff model == fit model

Rows are indices into the model's frame (immutable for the process
life); they are bounds-checked on every read.
"""
from __future__ import annotations

import hashlib
import json

from .analysis import Analysis


def token(spec: dict) -> str:
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:12]


def thr_synced(fit: dict | None, thr: dict | None) -> bool:
    """True when the threshold store was written for this exact fit."""
    return bool(fit and thr and thr.get("token") == fit.get("token"))


def thr_values(thr: dict | None) -> dict:
    return dict((thr or {}).get("values") or {})


def _rows(n: int, rows) -> list[int]:
    return [int(r) for r in (rows or [])
            if isinstance(r, (int, float)) and 0 <= int(r) < n]


def resolve(frame, sel: dict | None, y=None, label_keys=None) -> list[int]:
    """Rows a selection stands for on ``frame``: each clause → a row set,
    combined by the selection's strategy (newest only / union / intersect).
    A class clause is stored as names (not 46k indices) and only applies to
    the label columns it was made with. No clauses = no selection."""
    import numpy as np
    if not sel or sel.get("model") != frame.name:
        return []
    sets = []
    for src, c in (sel.get("clauses") or {}).items():
        if src == "classes":
            keys = tuple(c.get("label_keys") or ())
            if label_keys is not None and keys != tuple(label_keys):
                continue
            yy = y if y is not None else frame.labels(keys)
            sets.append(set(np.flatnonzero(
                np.isin(yy, list(c.get("classes") or []))).tolist()))
        else:
            sets.append(set(_rows(frame.n, c.get("rows"))))
    if not sets:
        return []
    out = set.intersection(*sets) if sel.get("mode") == "intersect" \
        else set.union(*sets)
    return sorted(out)


def sel_rows(an: Analysis, sel: dict | None) -> list[int]:
    return resolve(an.frame, sel, y=an.y, label_keys=an.spec.label_keys)


def sel_key(sel: dict | None):
    """What the selection means, without its bookkeeping (``seq``)."""
    if not sel:
        return None
    return [sel.get("model"), sel.get("mode"),
            {s: {k: v for k, v in c.items() if k != "seq"}
             for s, c in (sel.get("clauses") or {}).items()}]


def focus_parts(an: Analysis, focus: dict | None) -> tuple[list[int], list[str]]:
    """→ (rows, labels) of a focus that belongs to this analysis."""
    if not focus or focus.get("model") != an.frame.name:
        return [], []
    return (_rows(an.frame.n, focus.get("rows"))[:2],
            [str(lb) for lb in (focus.get("labels") or []) if lb in an.labels])


def neighbours(an_or_frame, row: int, k: int = 8) -> list[tuple[int, float]]:
    """Top-k cosine neighbours of one row: [(row, cosine), ...]."""
    import numpy as np
    frame = getattr(an_or_frame, "frame", an_or_frame)
    sims = frame.Vn @ frame.Vn[row]
    sims[row] = -2.0
    k = min(k, sims.size - 1)
    if k <= 0:
        return []
    top = np.argpartition(-sims, k - 1)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]
