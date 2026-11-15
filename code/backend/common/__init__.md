# __init__.py

## Purpose
Package initializer for `backend.common`. Re-exports all public symbols from the three submodules so consumers can do `from backend.common import get_label_keys` without knowing which subfile a function lives in.

## Re-exports

**From `data_access`:**
- `get_label_keys`
- `get_vectors_for_sessions`
- `build_multi_label_series`
- `session_to_row_index`

**From `hashing`:**
- `stable_hash`
- `hash_session_subset`

**From `multilabel`:**
- `HIER_PREFIX`
- `compute_multilabel_series`
- `restrict_to_session_ids`
- `unique_multilabels`

All symbols are listed in `__all__`.
