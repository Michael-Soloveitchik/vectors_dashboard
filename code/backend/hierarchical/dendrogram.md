# dendrogram.py

## Purpose
Extracts dendrogram coordinates from a scipy linkage result and repacks them into Plotly-friendly trace data. Also computes the `(x, y)` position of every internal cluster node (merge junction) so the front-end can render invisible scatter markers on them -- those markers are the click targets that let the user select tree nodes for hierarchical labeling.

## Motivation
scipy's `dendrogram(no_plot=True)` returns segment coordinates (`icoord`/`dcoord`) without requiring matplotlib rendering, but the raw format needs reshaping for Plotly's scatter/line traces. Additionally, the front-end needs per-node positions to place interactive click targets on the dendrogram junctions, which scipy does not provide directly. This module bridges the gap between scipy's output format and the Plotly/Dash rendering layer.

## Key Classes / Functions

### `dendrogram_traces(result, line_color, line_width)`
- **What it does:** Returns a dict of Plotly-ready line segments for the dendrogram plus per-node click-target positions.
- **Args:**
  - `result` (HierarchyResult): output of `cluster()`.
  - `line_color` (str): CSS color for dendrogram lines. Default `"#444444"`.
  - `line_width` (float): line width in pixels. Default `1.4`.
- **Returns:** dict with keys:
  - `lines_x`, `lines_y` (list[float | None]): Plotly line coordinates with `None` breaks between segments.
  - `node_ids` (list[int]): internal cluster IDs (n + k for k-th merge).
  - `node_x`, `node_y` (list[float]): center-top position of each internal node's U-shape.
  - `color` (str), `width` (float): style echoed back.
- **Example call:**
  ```python
  from backend.hierarchical.dendrogram import dendrogram_traces
  traces = dendrogram_traces(result, line_color="#222", line_width=2.0)
  ```

### `_internal_node_positions(result, dd)`
- **What it does:** Computes the `(x, y)` center-top position of every internal node's U-shape in the dendrogram.
- **Args:**
  - `result` (HierarchyResult): clustering result with `n` leaves.
  - `dd` (dict): raw scipy dendrogram output with `icoord`/`dcoord` keys.
- **Returns:** tuple of `(ids, xs, ys)` -- three parallel lists.

### `node_xy_lookup(traces)`
- **What it does:** Builds a reverse lookup dict from cluster ID to its `(x, y)` junction position.
- **Args:**
  - `traces` (dict): output of `dendrogram_traces()`.
- **Returns:** `dict[int, tuple[float, float]]` mapping cluster_id to (x, y).
- **Example call:**
  ```python
  lookup = node_xy_lookup(traces)
  x, y = lookup[cluster_id]
  ```

## Inputs & Outputs
- **In:** `HierarchyResult` (from `clustering.py`) containing `Z` shape `(n-1, 4)` and `n`.
- **Out:** Dict with Plotly line coordinates (lists of float/None) and per-node position arrays. For `n < 2`, all lists are empty.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| `n >= 2` | Full line and node position data | Normal path |
| `n < 2` | All lists empty, style keys still present | Graceful degenerate case |
| Large `n` (hundreds of sessions) | Works but Plotly rendering may be slow | No internal limit |

## Where This Is Used
- `backend/hierarchical/__init__.py` -- re-exported as part of the public surface.
- `frontend/callbacks/render_helpers.py` -- calls `dendrogram_traces()` to build the Plotly figure.
- `TESTS/tests/test_hierarchical.py` -- tests both `dendrogram_traces` and `node_xy_lookup`.
- `TESTS/tests/test_figures.py` -- integration test with figure building.

## Known Gotchas & Non-Obvious Behavior
- The k-th row of `Z` corresponds to internal cluster ID `n + k`. scipy emits `icoord`/`dcoord` entries in the same row order as `Z`, so zipping them yields the correct cluster-to-position mapping.
- The top of the U-shape is at `(mean(icoord[1:3]), dcoord[1])` -- midpoint of the upper horizontal bar, not the geometric center of the entire U.
- `color_threshold=-1` is passed to scipy's `dendrogram()` to suppress automatic color assignment (all segments use a single color, set by the caller).
- `None` values in `lines_x`/`lines_y` are Plotly's line-break convention for drawing disconnected segments in a single trace.

## Dependencies
- `numpy`
- `scipy.cluster.hierarchy.dendrogram`
- `.clustering.HierarchyResult` (internal)
