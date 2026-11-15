"""Common helpers shared across backend modules.

The single source of truth for moving data between the dataframe (rows
indexed implicitly) and the rest of the system (which addresses
everything by `session_id`). All vector / label / hash plumbing lives
here so the algorithm modules stay tight and unaware of dataframe
mechanics.
"""

from .data_access import (
    get_label_keys,
    get_vectors_for_sessions,
    build_multi_label_series,
    session_to_row_index,
)
from .hashing import stable_hash, hash_session_subset
from .multilabel import (
    HIER_PREFIX,
    compute_multilabel_series,
    restrict_to_session_ids,
    unique_multilabels,
)

__all__ = [
    "get_label_keys",
    "get_vectors_for_sessions",
    "build_multi_label_series",
    "session_to_row_index",
    "stable_hash",
    "hash_session_subset",
    "HIER_PREFIX",
    "compute_multilabel_series",
    "restrict_to_session_ids",
    "unique_multilabels",
]
