# matrix.py

## Purpose
Builds a cosine-similarity matrix (`V @ V.T`) over a subset (or all) of the session embeddings in the dataframe. This is the core data structure behind the confusion matrix heatmap in the dashboard (spec point 22). The matrix is always cosine similarity, enforced by explicit L2 row-normalization before the dot product.

## Motivation
The dashboard originally assumed all vectors were L2-normalized at generation time, so `V @ V.T` was implicitly cosine similarity. When multi-model fusion was introduced (concat/reinforce in `data/speaker_model.py`), the resulting vectors were no longer unit-norm, and the raw dot product silently shifted scale. This module explicitly L2-normalizes each row before computing the matrix, making the result correct regardless of how vectors were produced upstream.

## Key Classes / Functions

### `ConfusionMatrix`
- **What it does:** Dataclass wrapping the (N, N) cosine-similarity matrix along with the session IDs that index its rows and columns.
- **Fields:**
  - `matrix` (np.ndarray): shape (N, N), float32, cosine similarity values in [-1, 1].
  - `session_ids` (list[str]): length N, maps row/column index i to a session_id. This is the authoritative index -- never use `df.iloc[i]` for cell labeling.
  - `is_full` (bool, default=False): True when the matrix covers all rows in the dataframe (i.e., no subset was requested).
  - `meta` (dict, default={}): metadata dict accumulated by downstream transforms (sort keys, color transform params, etc.).
- **Properties:**
  - `n -> int`: number of sessions (length of `session_ids`).

### `_l2_normalize_rows`
- **What it does:** Row-wise L2 normalization. Zero-norm rows remain zero (no NaN).
- **Args:**
  - `v` (np.ndarray): shape (N, D).
- **Returns:** (N, D) array with each row having unit L2 norm (or zero if the input row was zero).

### `build_matrix`
- **What it does:** Extracts vectors for the given session IDs, L2-normalizes, and computes the cosine-similarity matrix.
- **Args:**
  - `df` (pd.DataFrame): the main dataframe with `session_id` and `centroid` columns.
  - `session_ids` (list[str] | None): which sessions to include. None or empty list means "all sessions" (the canonical "everything" sentinel per spec point 22).
  - `mask_non_selected` (bool, default=False): when True and `session_ids` is a non-empty subset, the returned matrix keeps the FULL shape (all rows in `df`) but rows whose session_id is not in `session_ids` are set to NaN before the matmul. NaN propagates, so every cell that touches a non-selected vector ends up NaN — the selected subgrid carries real values, everything else is NaN. Ignored when `session_ids` is None/empty.
- **Returns:** `ConfusionMatrix` with the cosine-similarity matrix and the list of session IDs actually used (which may differ from the input if some IDs were not found).
- **Raises:** No exceptions; missing session IDs are silently filtered by `get_vectors_for_sessions`. An empty result returns a 0x0 matrix.
- **Example call:**
  ```python
  cm = build_matrix(df, session_ids=["sess_001", "sess_002", "sess_003"])
  cm_all = build_matrix(df, session_ids=None)  # everything
  cm_masked = build_matrix(df, session_ids=["A", "C"], mask_non_selected=True)
  # cm_masked has full df shape; only A/C rows+cols are real, rest NaN
  ```

## Inputs & Outputs

**IN:**
- `df`: DataFrame with `session_id` (str) column and `centroid` column containing D-dimensional vectors.
- `session_ids`: optional list of session_id strings to subset.

**OUT:**
- `ConfusionMatrix` with:
  - `matrix`: (N, N) float32 cosine similarity.
  - `session_ids`: length-N list of strings.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal subset | (N, N) cosine matrix for N sessions | Standard path |
| `session_ids=None` or empty list | Matrix covers all rows, `is_full=True` | Canonical "everything" sentinel |
| Some session IDs not found in df | Silently uses only found IDs | `get_vectors_for_sessions` filters |
| Zero-norm vector row | Row and column for that session are all zeros | `_l2_normalize_rows` avoids NaN |
| No valid sessions | 0x0 matrix, empty session_ids list | Returns successfully |
| Single session | 1x1 matrix with value 1.0 (self-similarity) | Cosine of a vector with itself |
| `mask_non_selected=True` + subset | Full n×n matrix, NaN on non-selected rows/cols | Selected subgrid real, rest NaN |

## Where This Is Used

- `backend/confusion_matrix/__init__.py` -- re-exported as public API.
- `backend/confusion_matrix/sort.py` -- `sort_matrix` takes and returns `ConfusionMatrix`.
- `backend/confusion_matrix/transforms.py` -- `apply_color_transform` takes and returns `ConfusionMatrix`.
- `backend/hierarchical/clustering.py:21` -- imports `ConfusionMatrix` for hierarchical clustering input.
- Multiple test files in `code/TESTS/tests/`.

## Known Gotchas & Non-Obvious Behavior

- `L2_EPS = 1e-12` is the epsilon used to prevent division by zero during normalization. This is a module-level constant.
- The matrix is always float32 regardless of input dtype. The `astype(np.float32, copy=False)` calls avoid unnecessary copies when input is already float32.
- `session_ids` on the returned `ConfusionMatrix` may differ from the input list (reordered or filtered). Always use the returned `session_ids` as the authoritative index.
- The `meta` dict is empty on construction but gets populated by downstream transforms (`sort_matrix`, `apply_color_transform`). It is accumulated via `{**cm.meta, ...}` spread so earlier metadata is preserved.

## Dependencies

- `numpy`, `pandas` -- core computation.
- `backend.common.data_access` -- `get_vectors_for_sessions`, `SESSION_COL`.
