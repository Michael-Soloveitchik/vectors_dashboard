# knn.py

## Purpose
Implements k-NN label propagation from a small set of user-labeled anchor sessions to the full dataset. Given anchor sessions with known labels, this module fits a k-NN index on L2-normalized vectors (cosine similarity via euclidean distance on the unit sphere) and assigns each unlabeled session the majority label among its k nearest neighbors that are anchors. This enables the Atlas/TensorBoard-style semi-automated labeling workflow: the user labels 2-3 sessions, and the dashboard auto-extends those labels across the embedding space.

## Motivation
Manual labeling of every session is impractical for large datasets. k-NN propagation lets users bootstrap labels from a handful of anchor points, leveraging the assumption that sessions close in embedding space share semantic similarity. sklearn's `NearestNeighbors` handles the index building and querying efficiently, while this module adds the voting logic, confidence scoring, and anchor preservation on top.

## Key Classes / Functions

### `PropagationResult`
- **What it does:** Dataclass holding the output of label propagation, aligned to the full session list.
- **Fields:**
  - `propagated_labels` (pd.Series): label strings for all sessions. `""` for sessions that could not be labeled.
  - `confidence` (pd.Series): float in `[0, 1]`, fraction of k neighbors that agreed on the assigned label. `1.0` for anchor sessions.
  - `is_anchor` (pd.Series): boolean, True for sessions that were in the original `anchor_labels`.

### `propagate_labels(vectors, session_ids, anchor_labels, k)`
- **What it does:** Propagates labels from anchor sessions to all sessions via k-NN majority vote.
- **Args:**
  - `vectors` (np.ndarray): shape `(N, D)`, embedding vectors for all sessions.
  - `session_ids` (list[str]): session IDs aligned to `vectors`, length `N`.
  - `anchor_labels` (dict[str, str]): maps session_id to label for known anchor sessions. Only non-empty string labels are used.
  - `k` (int): number of neighbors to consider. Default `5`.
- **Returns:** `PropagationResult` with Series of length `N`.
- **Example call:**
  ```python
  from backend.propagation.knn import propagate_labels
  result = propagate_labels(
      vectors, session_ids,
      anchor_labels={"sess_01": "cat", "sess_05": "dog"},
      k=5,
  )
  print(result.propagated_labels.value_counts())
  ```

## Inputs & Outputs
- **In:** Embedding matrix `vectors` shape `(N, D)`, `session_ids` list of length `N`, `anchor_labels` dict mapping session_id to label string, and `k` int.
- **Out:** `PropagationResult` with three pandas Series, all of length `N`:
  - `propagated_labels`: str (label or `""`)
  - `confidence`: float `[0, 1]`
  - `is_anchor`: bool

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal: some anchors, k < N | All sessions get labels if they have anchor neighbors | Standard path |
| Anchor session in input | Keeps original label, confidence=1.0, is_anchor=True | Anchors are never overridden |
| No anchor labels (empty dict) | All labels `""`, confidence 0.0 | Nothing to propagate from |
| N == 0 | Empty Series for all three outputs | Graceful empty case |
| k > N | `k_eff = min(k, N)` | Clamped automatically |
| Session with no anchor neighbors | Label `""`, confidence 0.0 | No votes means no label |
| Tie between labels | `max()` picks one (implementation-defined) | Python `max` on dict is deterministic but arbitrary for ties |
| Anchor session_id not in session_ids | Silently skipped | `sid_to_idx.get(sid)` returns None |
| Anchor with empty string label | Skipped (filtered by `if idx is not None and lbl`) | Only non-empty labels count |

## Where This Is Used
- `backend/propagation/__init__.py` -- re-exported.
- `frontend/callbacks/render.py:39` -- calls `propagate_labels` during label propagation workflow.
- `TESTS/tests/test_knn_propagation.py` -- dedicated test suite.

## Known Gotchas & Non-Obvious Behavior
- `NearestNeighbors` is imported lazily inside the function to avoid a top-level sklearn dependency.
- The k-NN index is fitted on **all** points (not just anchors), so every session's neighbors include both anchors and non-anchors. Only anchor neighbors contribute votes. This means a session surrounded entirely by non-anchors will remain unlabeled even if k is large.
- Confidence is `votes[best_label] / k_eff`, where `k_eff` is the actual number of neighbors queried (may be less than k if N is small). This means confidence reflects the fraction of **all** neighbors, not just anchor neighbors, which can result in low confidence even for unambiguous labels if most neighbors are not anchors.
- L2-normalization is applied before fitting the index, so euclidean distance on the unit sphere is equivalent to cosine distance. This is the standard sklearn pattern for cosine k-NN.

## Dependencies
- `numpy`
- `pandas`
- `sklearn.neighbors.NearestNeighbors` -- lazy import
