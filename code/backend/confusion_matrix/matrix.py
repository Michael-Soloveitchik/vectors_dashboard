"""Build the V @ V.T cosine-similarity matrix over an aggregated session set.

Empty `session_ids` (or None) is the canonical "everything" sentinel
per spec point 22 — the matrix then covers ALL rows in df. The
`session_ids` field on the returned ConfusionMatrix is ALWAYS the
list that maps row i / col i of the matrix back to the dataframe.

The `df` we receive is already filtered to one model (so all rows
share a vector dimensionality) — the multi-model split happens at
the front-end model selector.

**Always cosine, never raw dot product.** The demo data is L2-normalized
at generation time (`data/generate.py:enrich_with_model_and_centroids`),
so historically `V @ V.T` already produced cosine similarity. The moment
the user starts COMBINING models (concat / reinforce — see
`data/speaker_model.py`) the resulting vectors are not unit-norm and the
raw dot product would silently shift scale. We L2-renormalize each row
of `V` here so the matrix is always cosine similarity regardless of
how the vectors arrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..common.data_access import get_vectors_for_sessions, SESSION_COL

L2_EPS = 1e-12


@dataclass
class ConfusionMatrix:
    """Per-cell cosine-similarity matrix + the session_ids that index it.

    `matrix[i, j]` is the cosine similarity between the vectors of
    `session_ids[i]` and `session_ids[j]` — always, even when the
    upstream vectors weren't unit-norm. The session_ids list is
    truth — never index `df.iloc[i]` for cell labeling.
    """

    matrix: np.ndarray
    session_ids: list[str]
    is_full: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.session_ids)


def _l2_normalize_rows(v: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize. Zero-rows stay zero (no nan)."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, L2_EPS)


def build_matrix(
    df: pd.DataFrame,
    session_ids: list[str] | None,
    mask_non_selected: bool = False,
) -> ConfusionMatrix:
    """Compute the cosine-similarity matrix for the given session subset.

    The matrix is always cosine — the row L2-normalize step makes the
    result correct even when the upstream vectors are NOT unit-norm
    (e.g., concat / reinforce fusions out of `data/speaker_model.py`).

    ``mask_non_selected`` keeps the full-shape matrix but sets every
    non-selected vector to NaN before the matmul. NaN propagates, so
    any cell that references a non-selected vector ends up NaN — the
    resulting matrix has the same dimensions as the unaggregated one
    but only the selected subgrid carries real values. Ignored when
    ``session_ids`` is None/empty (the "everything" sentinel).
    """
    if session_ids is None or len(session_ids) == 0:
        sids = df[SESSION_COL].tolist()
        is_full = True
        mask_non_selected = False
    elif mask_non_selected:
        sids = df[SESSION_COL].tolist()
        is_full = True
    else:
        sids = list(session_ids)
        is_full = False

    vecs, used = get_vectors_for_sessions(df, sids)
    if vecs.size == 0:
        return ConfusionMatrix(
            np.zeros((0, 0), dtype=np.float32),
            [],
            is_full=is_full,
        )
    vecs = vecs.astype(np.float32, copy=True)
    if mask_non_selected:
        sel = {str(s) for s in session_ids}
        mask_idx = np.asarray(
            [i for i, s in enumerate(used) if str(s) not in sel],
            dtype=np.int64,
        )
        if mask_idx.size:
            vecs[mask_idx] = np.nan
    vecs_unit = _l2_normalize_rows(vecs)
    cm = (vecs_unit @ vecs_unit.T).astype(np.float32, copy=False)
    return ConfusionMatrix(cm, used, is_full=is_full)
