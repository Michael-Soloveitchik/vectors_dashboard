"""Click-driven hierarchical_labels manager.

Spec point 41:
* Clicking a tree node N "creates a label" — every leaf under N gets
  label N.
* Clicking a child C of an active node N splits the label: leaves
  under C now belong to label C, the rest of N stays in label N.
* If both children C1 and C2 of N are clicked, N has nothing left,
  so it disappears from the label set.
* Clicking C again (un-toggle) brings N back as the label for those
  leaves.
* Reset clears the entire active set, leaving the dataset unlabeled.

Implementation: keep an `active_nodes` set. The label of a leaf is the
*lowest* active ancestor (the deepest node walking up the tree that
the user has activated). If no ancestor is active, the leaf is
unlabeled (sentinel: empty string).
"""

from __future__ import annotations

import numpy as np


class HierarchyLabels:
    """Label assignment driven by clicks on a scipy linkage tree.

    Constructed from the same `HierarchyResult.Z` and
    `ordered_session_ids` produced by `cluster()`. The active set is
    plain Python so it serializes cleanly into a Dash Store.
    """

    def __init__(self, Z: np.ndarray, leaf_session_ids: list[str]) -> None:
        self.Z = Z
        self.leaf_sids = list(leaf_session_ids)
        self.n = len(self.leaf_sids)
        self.active_nodes: set[int] = set()
        self._parent = self._build_parent_map()

    def _build_parent_map(self) -> dict[int, int]:
        """child_id -> parent_id for every non-root node."""
        parent: dict[int, int] = {}
        for k in range(self.Z.shape[0]):
            c1, c2 = int(self.Z[k, 0]), int(self.Z[k, 1])
            parent[c1] = self.n + k
            parent[c2] = self.n + k
        return parent

    def _leaves_under(self, node_id: int) -> list[int]:
        if node_id < self.n:
            return [node_id]
        out: list[int] = []
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur < self.n:
                out.append(cur)
            else:
                k = cur - self.n
                stack.append(int(self.Z[k, 0]))
                stack.append(int(self.Z[k, 1]))
        return out

    def toggle(self, node_id: int) -> None:
        if node_id in self.active_nodes:
            self.active_nodes.remove(node_id)
        else:
            self.active_nodes.add(node_id)

    def reset(self) -> None:
        self.active_nodes.clear()

    def labels(self) -> dict[str, int | str]:
        """session_id -> label id (the lowest active ancestor's int id).

        Unlabeled leaves get the empty string sentinel so the front-end
        can apply the no-label gray color to them.
        """
        out: dict[str, int | str] = {}
        for leaf in range(self.n):
            cur: int | None = leaf
            chosen: int | str = ""
            while cur is not None:
                if cur in self.active_nodes:
                    chosen = cur
                    break
                cur = self._parent.get(cur)
            out[self.leaf_sids[leaf]] = chosen
        return out

    def to_state(self) -> dict:
        return {"active_nodes": sorted(self.active_nodes)}

    @classmethod
    def from_state(
        cls,
        state: dict,
        Z: np.ndarray,
        leaf_session_ids: list[str],
    ) -> "HierarchyLabels":
        obj = cls(Z, leaf_session_ids)
        obj.active_nodes = set(int(n) for n in state.get("active_nodes", []))
        return obj
