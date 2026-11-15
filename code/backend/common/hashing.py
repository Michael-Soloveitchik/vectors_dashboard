"""Stable hashing for cache keys.

Cache keys are tuples of (method, *params, content_hash). The
content hash is a fast 64-bit hash of the *session id list* the
reduction was computed on, plus a content fingerprint of the actual
vector matrix. Hashing only ids would miss when an upstream stage
re-samples the same ids; hashing the full matrix every call is too
slow on big arrays. We do both — the matrix fingerprint is a
shape+strides+sum trick, not a real cryptographic hash.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np


def _array_fingerprint(arr: np.ndarray) -> str:
    """Cheap 16-char fingerprint of a numpy array.

    NOT a cryptographic hash. Combines shape, dtype, and the byte hash
    of a few sentinel rows so different (N, D) arrays with different
    contents collide essentially never.
    """
    if arr.size == 0:
        return f"empty:{arr.shape}:{arr.dtype}"
    h = hashlib.blake2b(digest_size=8)
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def stable_hash(*parts: Any) -> str:
    """Hash a heterogeneous tuple into a deterministic 16-char string.

    np.ndarray entries are fingerprinted via _array_fingerprint;
    everything else is repr()'d. Order matters.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(_array_fingerprint(p).encode())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


def hash_session_subset(session_ids: Iterable[str]) -> str:
    """Order-sensitive hash of a session_id sequence."""
    h = hashlib.blake2b(digest_size=8)
    for sid in session_ids:
        h.update(sid.encode())
        h.update(b",")
    return h.hexdigest()
