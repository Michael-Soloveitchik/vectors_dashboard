# lookup.py

## Purpose
Performs top-K nearest neighbor lookup by cosine similarity for a single query session. When a user clicks a point in the scatter plot, this module returns the K most similar sessions with their cosine distances and similarities. The implementation is self-contained: it L2-normalizes the vectors, computes cosine similarity via dot product, and uses `np.argpartition` for efficient top-K selection without fitting a full index.

## Motivation
The dashboard's per-point inspection feature ("click a point, see its neighbors") needs a fast nearest-neighbor query that runs on every click. A full sklearn `NearestNeighbors` fit would be wasteful for a single query against a dataset that fits in memory. Instead, this module computes the full cosine similarity vector for the query and uses `argpartition` for O(N) top-K selection, which is fast enough for interactive use on datasets up to tens of thousands of sessions.

## Key Classes / Functions

### `NearestResult`
- **What it does:** Dataclass holding the result of a single-point nearest-neighbor query.
- **Fields:**
  - `query_sid` (str): session_id of the query point.
  - `neighbor_sids` (list[str]): K nearest session IDs, sorted by descending similarity.
  - `distances` (np.ndarray): cosine distances `(1 - similarity)`, shape `(K,)`.
  - `similarities` (np.ndarray): cosine similarities, shape `(K,)`.

### `find_nearest(vectors, session_ids, query_sid, k)`
- **What it does:** Finds the K nearest neighbors of a query session by cosine similarity.
- **Args:**
  - `vectors` (np.ndarray): shape `(N, D)`, embedding vectors for all sessions.
  - `session_ids` (list[str]): session IDs aligned to `vectors`, length `N`.
  - `query_sid` (str): session_id to query neighbors for.
  - `k` (int): number of neighbors to return. Default `5`.
- **Returns:** `NearestResult` or `None` if `query_sid` is not found or `N < 2`.
- **Example call:**
  ```python
  from backend.neighbors.lookup import find_nearest
  result = find_nearest(vectors, session_ids, "session_042", k=10)
  if result:
      for sid, sim in zip(result.neighbor_sids, result.similarities):
          print(f"{sid}: {sim:.4f}")
  ```

## Inputs & Outputs
- **In:** Embedding matrix `vectors` shape `(N, D)`, `session_ids` list of length `N`, a `query_sid` string, and `k` int.
- **Out:** `NearestResult` with `neighbor_sids` (list of K strings), `distances` and `similarities` (both shape `(K,)`), sorted by descending similarity. Returns `None` for invalid queries.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal query, K < N | K neighbors sorted by similarity | Standard path |
| K >= N-1 | Returns all sessions except query | `k_eff = min(k, N-1)` |
| `query_sid` not in `session_ids` | `None` | Graceful miss |
| N < 2 | `None` | Cannot have neighbors with 0 or 1 points |
| Zero-norm query vector | Still works, all similarities near 0 | `max(norm, 1e-12)` prevents div-by-zero |
| Duplicate session IDs | First occurrence wins in `sid_to_idx` | Not explicitly guarded |

## Where This Is Used
- `backend/neighbors/__init__.py` -- re-exported.
- `frontend/callbacks/nearest_neighbors.py:15` -- calls `find_nearest` on scatter click.
- `TESTS/tests/test_nearest_neighbors.py` -- dedicated test suite.

## Known Gotchas & Non-Obvious Behavior
- The query point is excluded from results by setting `sims[q_idx] = -2.0` (below the minimum possible cosine similarity of -1.0).
- `np.argpartition` is used instead of `np.argsort` for O(N) top-K selection, then the top-K indices are sorted by similarity for the final output. This is faster than a full sort for large N.
- All similarity computation is done in float64 (via `.astype(np.float64)`) to avoid float32 precision issues in cosine similarity ranking.
- The function does **not** build a persistent index -- it recomputes similarities from scratch on every call. This is appropriate for click-driven queries but not for batch lookups.

## Dependencies
- `numpy`
