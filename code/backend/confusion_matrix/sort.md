# sort.py

## Purpose
Implements multi-key lexicographic sorting of the confusion matrix rows and columns. Supports up to three sort levels where each level can be a dataframe column or the special `"random"` token for a uniform shuffle within remaining ties (spec point 23). Also exposes a helper to compute which sort keys are still available for subsequent levels, preventing the same key from being used twice.

## Motivation
Sorting the confusion matrix by metadata columns (e.g., speaker, language, session type) reveals block-diagonal structure that indicates cluster quality. The three-level sort with a random fallback gives users enough flexibility to explore hierarchical groupings without needing a custom sort UI. The key-exclusion logic (`available_sort_keys`) keeps the front-end dropdowns consistent by removing already-chosen keys from later levels.

## Key Classes / Functions

### `available_sort_keys`
- **What it does:** Returns the list of sort keys still available for the next sort level, plus the `"random"` token. Keys already chosen at earlier levels are excluded (except `"random"`, which can appear at multiple levels).
- **Args:**
  - `df` (pd.DataFrame): the main dataframe.
  - `chosen_so_far` (Sequence[str]): keys already selected at earlier sort levels.
  - `exclude` (Sequence[str], default=()): additional keys to exclude (e.g., vector columns).
- **Returns:** list[str] of available keys, ending with `"random"`.
- **Example call:**
  ```python
  keys = available_sort_keys(df, chosen_so_far=["speaker"])
  # e.g. ["language", "gender", "random"]
  ```

### `sort_matrix`
- **What it does:** Reorders the rows and columns of a `ConfusionMatrix` according to lexicographic sorting by the given keys.
- **Args:**
  - `cm` (ConfusionMatrix): the matrix to sort.
  - `df` (pd.DataFrame): the main dataframe (must contain the sort key columns and `session_id`).
  - `sort_keys` (Sequence[str]): ordered list of sort keys. First key is primary, last is least significant. `"random"` at any level inserts a random permutation as that level's rank.
  - `rng` (np.random.Generator | None, default=None): random number generator. Defaults to `default_rng(0)` if None.
- **Returns:** A new `ConfusionMatrix` with rows/columns reordered. The `meta` dict includes `"sort_keys"`.
- **Raises:** `KeyError` if a non-random sort key is not found in the dataframe columns.
- **Example call:**
  ```python
  sorted_cm = sort_matrix(cm, df, sort_keys=["speaker", "language", "random"])
  ```

## Inputs & Outputs

**IN:**
- `cm`: `ConfusionMatrix` with an (N, N) matrix and N session_ids.
- `df`: DataFrame with `session_id` column and any metadata columns used as sort keys.
- `sort_keys`: list of 1-3 key names or `"random"`.

**OUT:**
- New `ConfusionMatrix` with:
  - `matrix`: (N, N) with rows and columns reordered by the sort.
  - `session_ids`: reordered to match.
  - `meta`: includes `"sort_keys": [...]`.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal multi-key sort | Matrix reordered lexicographically | Primary key = sort_keys[0] |
| `sort_keys` is empty | Returns cm unchanged | Early return |
| `cm.n == 0` | Returns cm unchanged | Early return |
| `"random"` at one level | Uniform shuffle within ties of higher-priority keys | Deterministic given `rng` seed |
| Unknown key in sort_keys | `KeyError` raised | Only non-random keys checked |
| `"random"` at multiple levels | Each level gets an independent permutation | Each call to `rng.permutation` advances state |

## Where This Is Used

- `backend/confusion_matrix/__init__.py` -- re-exported as `sort_matrix` and `available_sort_keys`.
- Front-end callbacks that handle sort-level dropdown changes.

## Known Gotchas & Non-Obvious Behavior

- `np.lexsort` sorts by the **last** array in the list first. The code reverses `rank_cols` with `[::-1]` to make `rank_cols[0]` the primary sort key. This is correct but non-obvious.
- The `"random"` token is defined as the module constant `RANDOM_TOKEN = "random"`. It is excluded from the deduplication logic in `available_sort_keys`, meaning users can select `"random"` at multiple sort levels.
- The sub-dataframe is re-indexed by `session_id` with `drop=False`, keeping `session_id` as both the index and a column. This allows `session_id` itself to be a valid sort key.
- String columns are converted to `pd.Categorical` codes for sorting. This means sort order is lexicographic on string values, not on original dataframe row order.
- The returned `ConfusionMatrix` preserves all existing `meta` keys via spread (`{**cm.meta, ...}`) and adds/overwrites `"sort_keys"`.

## Dependencies

- `numpy`, `pandas` -- sorting and categorical encoding.
- `.matrix` -- `ConfusionMatrix`.
- `backend.common.data_access` -- `SESSION_COL`, `get_label_keys`.
