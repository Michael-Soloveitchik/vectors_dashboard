# labels.py

## Purpose
Manages click-driven hierarchical label assignment on a scipy linkage tree. When a user clicks an internal node in the dendrogram, all leaves under that node receive its label. Clicking child nodes splits the parent label; clicking them again un-splits. The class maintains an `active_nodes` set and resolves each leaf's label as the lowest (deepest) active ancestor walking up the tree.

## Motivation
The dashboard's hierarchical mode (spec point 41) requires interactive labeling driven by dendrogram clicks rather than manual per-session assignment. The logic for toggling nodes, splitting/merging labels, and resolving the "lowest active ancestor" is non-trivial and needs to be separated from UI callbacks. This class encapsulates the tree traversal and state management so the front-end only needs to call `toggle(node_id)` and `labels()`.

## Key Classes / Functions

### `HierarchyLabels`
- **What it does:** Stateful label manager built on a scipy linkage matrix. Tracks which internal nodes are "active" and resolves each leaf's label as its lowest active ancestor.

#### `__init__(self, Z, leaf_session_ids)`
- **Args:**
  - `Z` (np.ndarray): scipy linkage matrix, shape `(n-1, 4)`.
  - `leaf_session_ids` (list[str]): session IDs in leaf order (length `n`).

#### `toggle(self, node_id)`
- **What it does:** Adds `node_id` to the active set if absent, removes it if present.
- **Args:**
  - `node_id` (int): internal cluster ID (>= n for internal nodes, < n for leaves).

#### `reset(self)`
- **What it does:** Clears all active nodes, leaving the dataset fully unlabeled.

#### `labels(self)`
- **What it does:** Returns a dict mapping each session_id to its label (the int ID of the lowest active ancestor). Unlabeled leaves get the empty string `""`.
- **Returns:** `dict[str, int | str]` -- session_id to label ID or `""`.
- **Example call:**
  ```python
  hl = HierarchyLabels(result.Z, result.ordered_session_ids)
  hl.toggle(cluster_id)
  label_map = hl.labels()
  ```

#### `to_state(self)`
- **What it does:** Serializes the active set to a JSON-friendly dict for Dash Store persistence.
- **Returns:** `dict` with key `"active_nodes"` containing a sorted list of ints.

#### `from_state(cls, state, Z, leaf_session_ids)`
- **What it does:** Reconstructs a `HierarchyLabels` from a serialized state dict.
- **Args:**
  - `state` (dict): output of `to_state()`.
  - `Z` (np.ndarray): linkage matrix.
  - `leaf_session_ids` (list[str]): session IDs.
- **Returns:** `HierarchyLabels` instance with the active set restored.

### `_build_parent_map(self)`
- **What it does:** Builds a child-to-parent lookup dict from the linkage matrix.
- **Returns:** `dict[int, int]` mapping each non-root node to its parent.

### `_leaves_under(self, node_id)`
- **What it does:** Returns all leaf indices under a given node via iterative DFS.
- **Args:**
  - `node_id` (int): any node ID (leaf or internal).
- **Returns:** `list[int]` of leaf indices.

## Inputs & Outputs
- **In:** scipy linkage matrix `Z` shape `(n-1, 4)`, leaf session IDs list of length `n`, user click events (node IDs).
- **Out:** `dict[str, int | str]` mapping every session to either an integer cluster label or `""` (unlabeled sentinel).

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| No active nodes | All leaves map to `""` | Default unlabeled state |
| Single root node active | All leaves get root's ID | Entire dataset one label |
| Parent + child both active | Leaves under child get child's ID, rest get parent's | "Lowest active ancestor" rule |
| Both children of N active | N has no leaves left | N effectively disappears |
| Toggle same node twice | Returns to previous state | Toggle is idempotent pair |
| Leaf node toggled | That single leaf gets its own label | Leaves are valid node IDs |
| `from_state` with empty state | All unlabeled | Graceful recovery |

## Where This Is Used
- `backend/hierarchical/__init__.py` -- re-exported.
- `frontend/callbacks/render_helpers.py` -- instantiated and used to resolve labels after dendrogram clicks.
- `TESTS/tests/test_hier_labels.py` -- dedicated test suite.

## Known Gotchas & Non-Obvious Behavior
- Label resolution walks **up** from each leaf to the root, stopping at the **first** active ancestor. This means a deeper active node always overrides a shallower one -- the user gets fine-grained control by clicking deeper into the tree.
- The `active_nodes` set can contain both internal nodes (>= n) and leaf nodes (< n). Leaf nodes as active nodes is valid and labels just that single session.
- `to_state()` / `from_state()` are designed for Dash Store round-tripping. The active set is serialized as a sorted list of ints, not a set, because JSON does not support sets.
- The parent map does not include the root node (it has no parent), so `_parent.get(root)` returns `None`, which terminates the ancestor walk.

## Dependencies
- `numpy`
