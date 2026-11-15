# cache.py

## Purpose
In-memory LRU-style cache for 2D reduction results. Stores computed `(N, 2)` projection arrays keyed on `(method, params, vector content fingerprint, extra)` so that toggling between reduction methods or revisiting the same view is instant. A single instance lives on the GUI object and is threaded through every callback.

## Motivation
Dimensionality reduction (especially t-SNE and UMAP) is expensive -- seconds to tens of seconds on moderate datasets. The dashboard triggers reductions on every parameter change, method switch, or point selection. Without caching, every UI interaction would re-run the full algorithm. This cache ensures the same matrix with the same parameters is computed exactly once per process lifetime, using a content-based fingerprint so re-sampled matrices with different values do not collide.

## Key Classes / Functions

### `ReductionCache`
- **What it does:** A dict-backed FIFO cache with a configurable maximum entry count. Evicts oldest entries when full.
- **Constructor args:**
  - `max_entries` (int, default `64`): maximum number of cached projections
- **Example:**
  ```python
  cache = ReductionCache(max_entries=32)
  ```

#### `ReductionCache.make_key` (staticmethod)
- **What it does:** Builds a deterministic cache key string from method name, vectors, params dict, and extra tuple.
- **Args:**
  - `method` (str): reduction method name
  - `vectors` (np.ndarray): the input matrix
  - `params` (dict[str, Any]): reduction parameters
  - `extra` (tuple, default `()`): additional key components
- **Returns:** `str` -- a 16-char hex hash via `stable_hash`. Params are sorted by key before hashing for order-independence.
- **Example call:**
  ```python
  key = ReductionCache.make_key("pca", vectors, {"random_state": 42})
  ```

#### `ReductionCache.get`
- **What it does:** Looks up a cached projection. Returns `None` on miss.
- **Args:** same as `make_key`
- **Returns:** `np.ndarray | None` -- the cached `(N, 2)` projection, or `None`.

#### `ReductionCache.put`
- **What it does:** Stores a projection in the cache. Evicts the oldest entry if the cache is full. No-ops if the key already exists.
- **Args:** same as `make_key`, plus:
  - `proj` (np.ndarray): the `(N, 2)` projection to cache

#### `ReductionCache.clear`
- **What it does:** Empties the entire cache.

#### `ReductionCache.__len__`
- **What it does:** Returns the current number of cached entries.

## Inputs & Outputs
- **In:** Method strings, numpy arrays of shape `(N, D)`, parameter dicts, and optional extra tuples.
- **Out:** Cached numpy arrays of shape `(N, 2)` or `None` on miss.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| First call with new vectors | `get` returns `None` | Cache miss |
| Second call with same inputs | `get` returns cached array | Cache hit |
| Same params in different dict order | Same cache key | Params are sorted before hashing |
| Same shape, different content | Different cache key | `stable_hash` fingerprints array bytes |
| Cache full (> `max_entries`) | Oldest entry evicted on next `put` | FIFO eviction |
| `put` with existing key | No-op, no duplicate | Entry already present |
| `clear` called | All entries removed | `__len__` returns 0 |

## Where This Is Used
- `backend/reduction_2d/base.py:20` -- imported and used by `reduce_2d` for cache get/put
- `backend/reduction_2d/__init__.py` -- re-exported as part of the public API
- `backend/spots/reduction_2D.py` -- imported via `__init__` for the spots scatter module
- `backend/centroids/projection.py:33` -- imported for centroid projection caching
- `TESTS/tests/test_density_contour.py` -- instantiated in tests

## Known Gotchas & Non-Obvious Behavior
- Eviction is FIFO (oldest inserted, not least recently used). The `_order` list tracks insertion order, and `pop(0)` removes from the front. For the typical dashboard use case (small number of distinct reductions), this is effectively LRU since old entries are rarely revisited.
- The cache stores references to the numpy arrays, not copies. If a caller mutates a returned projection in-place, the cached version is also mutated. In practice this is safe because `reduce_2d` applies `recenter` which creates a new array.
- The `make_key` method sorts `params.items()` before hashing, so `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` produce the same key.

## Dependencies
- `numpy`
- Internal: `backend.common.hashing.stable_hash`
