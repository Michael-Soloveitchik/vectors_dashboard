"""Multi-key lexicographic sort for the confusion matrix.

Spec point 23 — three sort levels, each picks a key from the dataframe
or `random`. Levels with `random` add a uniform shuffle inside the
remaining ties. Once a key is picked at level k, it MUST disappear
from the option list of levels k+1.. so we expose
`available_sort_keys()` for the front-end dropdowns.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .matrix import ConfusionMatrix
from ..common.data_access import SESSION_COL, get_label_keys

RANDOM_TOKEN = "random"


def available_sort_keys(
    df: pd.DataFrame,
    chosen_so_far: Sequence[str],
    exclude: Sequence[str] = (),
) -> list[str]:
    """Keys still pickable for the next sort level + the random token."""
    used = set(chosen_so_far) - {RANDOM_TOKEN}
    keys = [k for k in get_label_keys(df, exclude=exclude) if k not in used]
    return keys + [RANDOM_TOKEN]


def sort_matrix(
    cm: ConfusionMatrix,
    df: pd.DataFrame,
    sort_keys: Sequence[str],
    rng: np.random.Generator | None = None,
) -> ConfusionMatrix:
    """Reorder rows+cols of `cm` by lexicographic key order."""
    if cm.n == 0 or not sort_keys:
        return cm

    rng = rng or np.random.default_rng(0)
    sub = df[df[SESSION_COL].isin(cm.session_ids)].copy()
    # drop=False keeps session_id as a column too, so it's still a
    # legal sort key alongside being the row index used for ordering.
    sub = sub.set_index(SESSION_COL, drop=False).loc[cm.session_ids]

    rank_cols: list[np.ndarray] = []
    for key in sort_keys:
        if key == RANDOM_TOKEN:
            rank_cols.append(rng.permutation(cm.n).astype(np.int64))
        else:
            if key not in sub.columns:
                raise KeyError(f"sort key {key!r} not in df columns")
            codes = pd.Categorical(sub[key].astype(str)).codes.astype(np.int64)
            rank_cols.append(codes)

    # np.lexsort sorts by the LAST key first, so reverse to make
    # rank_cols[0] the primary key.
    order = np.lexsort(rank_cols[::-1])

    new_sids = [cm.session_ids[i] for i in order]
    new_mat = cm.matrix[np.ix_(order, order)]
    return ConfusionMatrix(
        matrix=new_mat,
        session_ids=new_sids,
        is_full=cm.is_full,
        meta={**cm.meta, "sort_keys": list(sort_keys)},
    )
