"""In-memory cache for 2D reductions.

The cache is keyed on `(method, params_repr, vectors_fingerprint, extra)`
so the same matrix with the same params is recomputed exactly once per
process. The fingerprint is a content hash of `vectors` so a re-sampled
matrix that happens to have the same shape but different values does
NOT collide.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..common.hashing import stable_hash

CacheKey = str


class ReductionCache:
    """Tiny dict-backed cache. One instance lives on the Gui object."""

    def __init__(self, max_entries: int = 64) -> None:
        self._store: dict[CacheKey, np.ndarray] = {}
        self._order: list[CacheKey] = []
        self.max_entries = max_entries

    @staticmethod
    def make_key(
        method: str,
        vectors: np.ndarray,
        params: dict[str, Any],
        extra: tuple = (),
    ) -> CacheKey:
        ordered_items = tuple(sorted(params.items()))
        return stable_hash(method, ordered_items, vectors, extra)

    def get(
        self,
        method: str,
        vectors: np.ndarray,
        params: dict[str, Any],
        extra: tuple = (),
    ) -> np.ndarray | None:
        key = self.make_key(method, vectors, params, extra)
        return self._store.get(key)

    def put(
        self,
        method: str,
        vectors: np.ndarray,
        params: dict[str, Any],
        extra: tuple,
        proj: np.ndarray,
    ) -> None:
        key = self.make_key(method, vectors, params, extra)
        if key in self._store:
            return
        self._store[key] = proj
        self._order.append(key)
        while len(self._order) > self.max_entries:
            evict = self._order.pop(0)
            self._store.pop(evict, None)

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._store)
