# hashing.py

## Purpose
Provides stable, deterministic hashing utilities for building cache keys. The reduction cache needs to know whether the same set of vectors with the same parameters has already been computed. This module produces short (16-char hex) fingerprints from heterogeneous inputs -- numpy arrays, strings, tuples, dicts -- so cache lookups are fast and collisions are negligible.

## Motivation
Cache keys for 2D reductions must capture both the identity of the session set AND the actual vector content. Hashing only session ids would miss when an upstream stage re-samples the same ids with different vectors; hashing the full matrix with a cryptographic hash every call is too slow on large arrays. This module provides a middle ground: a fast content fingerprint combining shape, dtype, and full byte content via BLAKE2b (64-bit digest).

## Key Classes / Functions

### `_array_fingerprint`
- **What it does:** Produces a cheap 16-character hex fingerprint of a numpy array by hashing its shape, dtype, and full byte content.
- **Args:**
  - `arr` (np.ndarray): the array to fingerprint
- **Returns:** `str` -- a 16-char hex string. For empty arrays, returns `"empty:{shape}:{dtype}"`.
- **Example call:**
  ```python
  fp = _array_fingerprint(np.array([[1.0, 2.0], [3.0, 4.0]]))
  # "a3f1e2d409b7c8e1"  (example)
  ```

### `stable_hash`
- **What it does:** Hashes a heterogeneous tuple of arguments into a deterministic 16-char hex string. Order matters. Numpy arrays are fingerprinted via `_array_fingerprint`; everything else is `repr()`-ed.
- **Args:**
  - `*parts` (Any): variable positional arguments -- strings, numbers, tuples, dicts, np.ndarrays
- **Returns:** `str` -- 16-char hex hash.
- **Example call:**
  ```python
  key = stable_hash("pca", (("n_components", 2),), vectors, ("extra",))
  ```

### `hash_session_subset`
- **What it does:** Produces an order-sensitive hash of a sequence of session_id strings.
- **Args:**
  - `session_ids` (Iterable[str]): ordered sequence of session ids
- **Returns:** `str` -- 16-char hex hash. Different orderings produce different hashes.
- **Example call:**
  ```python
  h = hash_session_subset(["sess_01", "sess_02", "sess_03"])
  ```

## Inputs & Outputs
- **In:** Python primitives (str, int, float, tuple, dict), numpy arrays of any shape/dtype, and ordered sequences of session_id strings.
- **Out:** 16-character hexadecimal strings (64-bit BLAKE2b digests).

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Same inputs, same order | Identical hash | Deterministic by design |
| Same inputs, different order | Different hash | Order-sensitive for both `stable_hash` and `hash_session_subset` |
| Empty numpy array | `"empty:{shape}:{dtype}"` as fingerprint | Does not hash bytes of empty array |
| Empty session_ids list | Valid 16-char hex string | Hashes the empty sequence |
| Dict arguments | Hashed via `repr()` | Dict key order matters in Python 3.7+ (insertion order) |
| Very large numpy array | Full `tobytes()` hashed | No sampling -- hashes all content |

## Where This Is Used
- `backend/reduction_2d/cache.py:16` -- imports `stable_hash` to build cache keys for reduction results
- `backend/common/__init__.py` -- re-exports `stable_hash` and `hash_session_subset`
- `TESTS/tests/test_hashing.py` -- comprehensive unit tests for all three functions

## Known Gotchas & Non-Obvious Behavior
- These are NOT cryptographic hashes -- they use 64-bit BLAKE2b digests. Collision resistance is adequate for cache keys (birthday bound at ~2^32 entries) but not for security purposes.
- `_array_fingerprint` hashes the full byte content via `arr.tobytes()`, which means two arrays with different strides but identical logical content may produce different fingerprints if their in-memory layouts differ. In practice this is not an issue because vectors are always contiguous float32 arrays.
- The pipe character `b"|"` is used as a separator between parts in `stable_hash` to prevent `("ab", "c")` from colliding with `("a", "bc")`.

## Dependencies
- `hashlib` (standard library)
- `numpy`
