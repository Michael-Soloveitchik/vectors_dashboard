# multilabel.py

## Purpose
Computes unified multilabel strings for each data point by combining user-selected column labels with hierarchical click-driven labels. A session's multilabel can come from control-panel columns (e.g., "voice_id", "device_id"), from hierarchy labels assigned by interactive drill-down, or both. This module merges those two sources into a single per-row string used for coloring, grouping, and filtering throughout the dashboard.

## Motivation
The dashboard spec requires maintaining a `multilabel_set` of all multilabels defined by controls or hierarchical labeling. Labels from the two sources need to be concatenated when both are present, passed through when only one exists, and default to the empty string sentinel (displayed as grey/ungrouped) when neither is present. Centralizing this logic prevents inconsistent label assembly across the rendering, dump, and centroid callbacks.

## Key Classes / Functions

### `compute_multilabel_series`
- **What it does:** Returns a per-row multilabel string for every row in the DataFrame by merging column-based labels with hierarchical labels.
- **Args:**
  - `df` (pd.DataFrame): source dataframe
  - `label_keys` (Sequence[str]): column names selected by the user in controls
  - `hier_label_dict` (dict[str, int | str] | None, default `None`): a `session_id -> label_id` map from HierarchyLabels. Empty-string values are skipped.
  - `join` (str, default `" | "`): separator between label parts
- **Returns:** `pd.Series` of dtype `object` -- one composite label string per row. Empty string means "no label".
- **Example call:**
  ```python
  series = compute_multilabel_series(df, ["voice_id"], hier_label_dict={"s1": 2, "s3": 1})
  # Row s1: "spk_01 | h:2"
  # Row s2: "spk_02"         (no hier label)
  # Row s3: "spk_03 | h:1"
  ```

### `unique_multilabels`
- **What it does:** Returns all distinct non-empty multilabel values from a Series, sorted alphabetically for deterministic ordering.
- **Args:**
  - `series` (pd.Series): the multilabel series (e.g., from `compute_multilabel_series`)
- **Returns:** `list[str]` -- sorted unique non-empty labels.
- **Example call:**
  ```python
  labels = unique_multilabels(series)
  # ["h:1", "spk_01 | h:2", "spk_02", "spk_03 | h:1"]
  ```

### `restrict_to_session_ids`
- **What it does:** Reindexes a multilabel Series to follow an explicit session_id ordering, producing a new Series aligned to the given ids.
- **Args:**
  - `series` (pd.Series): the full multilabel series
  - `df` (pd.DataFrame): the dataframe used to map session_ids to row positions
  - `session_ids` (Sequence[str]): desired output order
- **Returns:** `pd.Series` -- length `len(session_ids)`, with empty string for any session_id not found.
- **Example call:**
  ```python
  restricted = restrict_to_session_ids(full_series, df, ["s3", "s1"])
  ```

### `HIER_PREFIX`
- **What it is:** Module-level constant string `"h:"`, prefixed to hierarchical label values to visually distinguish them from column-based labels.

## Inputs & Outputs
- **In:** A DataFrame with `session_id` column, a list of label column names, and optionally a hierarchical label dict mapping session_ids to label integers/strings.
- **Out:** `pd.Series` of composite label strings (dtype `object`).

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| No label_keys, no hier_label_dict | Series of empty strings | "No labels" sentinel |
| label_keys only | `build_multi_label_series` output directly | No hierarchy prefix |
| hier_label_dict only | Series with `"h:{value}"` per labeled row | Empty string for unlabeled rows |
| Both present | Concatenated: `"col_label | h:hier_value"` | Join separator configurable |
| hier_label_dict has empty-string or None value | That row gets no hierarchy part | Skipped, not `"h:"` |
| session_id not in hier_label_dict | Gets empty string for hierarchy part | Only column labels shown |
| `restrict_to_session_ids` with unknown session_id | Returns empty string for that position | Does not raise |

## Where This Is Used
- `backend/common/__init__.py` -- re-exports `HIER_PREFIX`, `compute_multilabel_series`, `unique_multilabels`, `restrict_to_session_ids`
- `frontend/callbacks/centroids_dynamic.py:27` -- imports `compute_multilabel_series`, `unique_multilabels` for centroid label computation
- `frontend/callbacks/dump.py:24` -- imports `compute_multilabel_series` for data export labeling
- `frontend/callbacks/render.py:25` -- imports `compute_multilabel_series` for scatter plot coloring

## Known Gotchas & Non-Obvious Behavior
- Hierarchical labels are prefixed with `"h:"` (`HIER_PREFIX`) to distinguish them visually from column-based labels. This prefix is part of the label string and will appear in legend/color mappings.
- The merge logic in `compute_multilabel_series` handles three cases (both, column-only, hier-only) via a zip loop. The empty string `""` acts as a sentinel throughout -- it is never displayed and the front-end interprets it as "grey / ungrouped".
- `restrict_to_session_ids` converts session_ids to `str` before lookup, so numeric session_ids in the dict will match string session_ids and vice versa.

## Dependencies
- `numpy`
- `pandas`
- Internal: `backend.common.data_access` (`SESSION_COL`, `build_multi_label_series`)
