"""Concurrency primitives for the compute engine.

The server runs one thread per request, so several callbacks routinely
ask for the same expensive result at the same moment (five panels after
one click), or for results nobody wants any more (a slider dragged
through ten values). Three small tools cover every such case:

``FlightCache``  bounded LRU + single-flight. The first caller of a key
                 computes; concurrent callers of the SAME key wait for
                 that one result; DIFFERENT keys run in parallel. No
                 lock is held while computing.
``HeavyGate``    at most ``slots`` heavy jobs (fit, t-SNE/UMAP, pair
                 sampling) at once — bounds CPU on a shared box — plus
                 latest-wins: a queued job no browser tab still wants
                 is dropped when its turn comes (``Superseded``).
``freeze``       marks shared arrays read-only so one request can never
                 mutate what another is reading.

Everything here is bounded: caches by entry count, the client table by
``MAX_CLIENTS``, threads by the gate. Nothing spawns a thread.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Hashable

import numpy as np

MAX_CLIENTS = 256


class Superseded(Exception):
    """The request that asked for this work is no longer the latest."""


class _Flight:
    __slots__ = ("done", "value", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None


class FlightCache:
    def __init__(self, name: str, max_entries: int) -> None:
        self.name = name
        self.max_entries = max(1, int(max_entries))
        self._mu = threading.Lock()
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._flights: dict[Hashable, _Flight] = {}
        self.stats = {"hit": 0, "computed": 0, "joined": 0, "evicted": 0}

    def get(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        while True:
            with self._mu:
                if key in self._data:
                    self._data.move_to_end(key)
                    self.stats["hit"] += 1
                    return self._data[key]
                flight = self._flights.get(key)
                owner = flight is None
                if owner:
                    flight = self._flights[key] = _Flight()
                else:
                    self.stats["joined"] += 1
            if not owner:
                flight.done.wait()
                if isinstance(flight.error, Superseded):
                    continue        # the owner gave up; this caller still wants it
                if flight.error is not None:
                    raise flight.error
                return flight.value
            try:
                flight.value = compute()
                with self._mu:
                    self._data[key] = flight.value
                    self.stats["computed"] += 1
                    while len(self._data) > self.max_entries:
                        self._data.popitem(last=False)
                        self.stats["evicted"] += 1
                return flight.value
            except BaseException as exc:    # failures are never cached
                flight.error = exc
                raise
            finally:
                with self._mu:
                    self._flights.pop(key, None)
                flight.done.set()

    def peek(self, key: Hashable) -> Any:
        with self._mu:
            return self._data.get(key)

    def clear(self) -> None:
        with self._mu:
            self._data.clear()

    def __len__(self) -> int:
        with self._mu:
            return len(self._data)


class KindCache:
    """One small ``FlightCache`` per kind of derived result, so e.g.
    normalized-score matrices for old alpha values are evicted without
    touching the ROC curves."""

    def __init__(self, caps: dict[str, int], default: int = 3) -> None:
        self._caps, self._default = dict(caps), default
        self._mu = threading.Lock()
        self._kinds: dict[str, FlightCache] = {}

    def get(self, kind: str, key: Hashable, compute: Callable[[], Any]) -> Any:
        with self._mu:
            cache = self._kinds.get(kind)
            if cache is None:
                cache = self._kinds[kind] = FlightCache(
                    kind, self._caps.get(kind, self._default))
        return cache.get(key, compute)

    def sizes(self) -> dict[str, int]:
        with self._mu:
            return {k: len(c) for k, c in self._kinds.items()}


class HeavyGate:
    def __init__(self, slots: int = 2) -> None:
        self.slots = max(1, int(slots))
        self._sem = threading.BoundedSemaphore(self.slots)
        self._mu = threading.Lock()
        self._latest: OrderedDict[tuple[str, str], Hashable] = OrderedDict()
        self.stats = {"ran": 0, "superseded": 0, "peak_waiting": 0}
        self._waiting = 0

    def want(self, client: str | None, kind: str, key: Hashable) -> None:
        """Record that ``client`` (one browser tab) now wants ``key``."""
        if not client:
            return
        with self._mu:
            self._latest[(client, kind)] = key
            self._latest.move_to_end((client, kind))
            while len(self._latest) > MAX_CLIENTS:
                self._latest.popitem(last=False)

    def _wanted(self, client: str | None, kind: str, key: Hashable) -> bool:
        if not client:
            return True                     # programmatic caller: always runs
        with self._mu:
            if (client, kind) not in self._latest:
                return True                 # table rolled over: never starve
            return any(k[1] == kind and v == key
                       for k, v in self._latest.items())

    @contextmanager
    def slot(self, client: str | None = None, kind: str = "",
             key: Hashable = None):
        with self._mu:
            self._waiting += 1
            self.stats["peak_waiting"] = max(self.stats["peak_waiting"],
                                             self._waiting)
        self._sem.acquire()
        try:
            with self._mu:
                self._waiting -= 1
            if not self._wanted(client, kind, key):
                with self._mu:
                    self.stats["superseded"] += 1
                raise Superseded(kind)
            with self._mu:
                self.stats["ran"] += 1
            yield
        finally:
            self._sem.release()


def freeze(a: np.ndarray) -> np.ndarray:
    """Read-only view contract for arrays shared between request threads."""
    a.setflags(write=False)
    return a
