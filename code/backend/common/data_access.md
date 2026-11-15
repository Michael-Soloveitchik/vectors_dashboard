# data_access.py

## Purpose
Session-id-stable accessors for extracting vectors, labels, and row mappings from a pandas DataFrame. This module is the single source of truth for translating between implicit DataFrame row indices and `session_id`-based addressing used by the rest of the system. It exists because aggregated point sets churn on every click, sort orders rearrange rows, and fresh reductions may reshuffle order entirely -- so the only stable row identity is `session_id`.

## Motivation
The dashboard's dataframe rows have no stable numeric index: filtering, sorting, and re-reduction all change row order. Every upstream module needs to address rows by `session_id`, but the raw DataFrame uses implicit integer indexing. This module centralizes that translation so algorithm modules (reduction, confusion matrix, centroids) stay unaware of DataFrame mechanics.

## Key Classes / Functions

### `get_label_keys`
- **What it does:** Returns all column names that contain primitive (non-vector) data, suitable for use as label/category keys.
- **Args:**
  - `df` (pd.DataFrame): the source dataframe
  - `exclude` (Sequence[str], default `()`): column names to skip
- **Returns:** `list[str]` -- column names in df column order, excluding `VECTOR_COL` ("centroid") and any column whose first-row value is an ndarray, list, tuple, or dict.
- **Example call:**
  ```python
  keys = get_label_keys(df, exclude=["session_id"])
  # e.g. ["voice_id", "device_id", "language"]
  ```

### `session_to_row_index`
- **What it does:** Builds a mapping from session_id strings to integer row positions in the current DataFrame ordering.
- **Args:**
  - `df` (pd.DataFrame): the dataframe to index
- **Returns:** `dict[str, int]` -- `{session_id: row_index}` for every row. Rebuilt on every call; callers should cache if needed.
- **Example call:**
  ```python
  idx_map = session_to_row_index(df)
  row = idx_map["sess_0042"]
  ```

### `get_vectors_for_sessions`
- **What it does:** Stacks centroid vectors for requested session_ids into a contiguous float32 array, preserving caller order.
- **Args:**
  - `df` (pd.DataFrame): source dataframe with a "centroid" column
  - `session_ids` (Iterable[str] | None): ordered list of session_ids to retrieve. `None` means "all rows in df order".
- **Returns:** `tuple[np.ndarray, list[str]]` -- `(vectors of shape (N, D) float32, list of session_ids actually used)`. Rows with missing or zero-size centroids are silently dropped.
- **Raises:** `ValueError` if the collected vectors have mixed dimensionality (e.g., mixing embeddings from different models).
- **Example call:**
  ```python
  vecs, used_sids = get_vectors_for_sessions(df, ["s1", "s2", "s3"])
  # vecs.shape == (3, 256), used_sids == ["s1", "s2", "s3"]
  ```

### `build_multi_label_series`
- **What it does:** Joins multiple label columns into a single composite string per row, separated by `join`.
- **Args:**
  - `df` (pd.DataFrame): source dataframe
  - `label_keys` (Sequence[str]): column names to concatenate
  - `join` (str, default `" | "`): separator between label parts
- **Returns:** `pd.Series` of dtype `object` with one composite string per row. Empty `label_keys` returns a series of empty strings (the sentinel for "no labels chosen").
- **Example call:**
  ```python
  labels = build_multi_label_series(df, ["voice_id", "device_id"])
  # e.g. "spk_01 | mic_A"
  ```

## Inputs & Outputs
- **In:** A pandas DataFrame where each row represents a data point with a `session_id` column (str) and a `centroid` column (numpy array of shape `(D,)` -- the embedding vector). Additional columns are primitive label/metadata fields.
- **Out:** Numpy arrays of shape `(N, D)` for vectors, `list[str]` for session_id lists, `pd.Series` for label strings, and `dict[str, int]` for index lookups.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| `session_ids=None` | Returns all rows in df order | Used as "select everything" semantics |
| Empty `session_ids` list | `(np.empty((0,0)), [])` | No crash, returns empty array |
| Session id not found in df | Silently skipped | Dropped from output, not an error |
| Centroid is empty/zero-size | Silently skipped | Row excluded from returned vectors |
| Mixed vector dimensions | `ValueError` raised | Must filter to single model_name first |
| Empty `label_keys` | Series of empty strings | Front-end interprets as "grey, no group" |
| DataFrame with zero rows | `get_label_keys` returns `[]`; vectors return `(0,0)` array | Sample-based type check uses `iloc[0]` only if `len(df) > 0` |

## Where This Is Used
- `backend/common/multilabel.py` -- imports `SESSION_COL`, `build_multi_label_series`
- `backend/common/__init__.py` -- re-exports all four functions
- `backend/centroids/gmm.py:17` -- imports `SESSION_COL`, `get_vectors_for_sessions`
- `backend/confusion_matrix/matrix.py:29` -- imports `get_vectors_for_sessions`, `SESSION_COL`
- `backend/confusion_matrix/sort.py:18` -- imports `SESSION_COL`, `get_label_keys`
- `frontend/layout/controls_sections.py:14` -- imports `get_label_keys`
- `frontend/layout/controls_sections_cm.py:12` -- imports `get_label_keys`
- `frontend/callbacks/render_helpers.py:12` -- imports `build_multi_label_series`, `get_vectors_for_sessions`
- `frontend/callbacks/nearest_neighbors.py:14` -- imports `get_vectors_for_sessions`
- `TESTS/tests/test_data_access.py` -- unit tests for all functions
- `TESTS/tests/test_ellipses.py` -- uses `build_multi_label_series`

## Known Gotchas & Non-Obvious Behavior
- `session_to_row_index` is rebuilt on every call intentionally -- the caller's df may be filtered or re-sorted. Caching should happen at the call site, not here.
- `get_vectors_for_sessions` returns vectors in **caller order** (order of `session_ids`), NOT in df row order. This is critical for maintaining alignment with external data structures.
- The type-detection in `get_label_keys` checks only `iloc[0]` -- if column types are mixed across rows, the first-row sample determines whether the column is treated as a label or a vector.
- `VECTOR_COL` and `SESSION_COL` are module-level constants (`"centroid"` and `"session_id"`) imported widely across the codebase as the canonical column names.

## Dependencies
- `numpy`
- `pandas`
- No external packages beyond the standard scientific stack.
