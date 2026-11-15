"""Sign convention — module-level single source of truth.

HIGHER SCORE ⇒ CLOSER / MORE-SIMILAR.

Flip ``CLOSEST_IS_MAX`` → every helper below rewires automatically and
every reduction/sort across reps + backend + frontend follows. Code
must NEVER call bare ``np.max`` / ``np.argmax`` / ``np.argsort`` on a
score array, nor use bare ``<`` / ``>`` to compare scores — route
through the helpers in this module.

Naming:
  • ``*_fn``    — callable helper (reducer, comparator, predicate)
  • ``*_order`` — returns an argsort-style index array (first/last)

This module is intentionally dependency-light: only ``numpy``. Anything
that needs the ``Representation`` class lives in ``base.py``; anything
specific to the aggregator resolver lives in ``aggregator_utils.py``.
"""

from __future__ import annotations

import numpy as np


CLOSEST_IS_MAX: bool = True


def closest_score_fn(S: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Reducer that picks the closest score. ``np.max`` by default."""
    return np.max(S, axis=axis) if CLOSEST_IS_MAX else np.min(S, axis=axis)


def arg_closest_score_fn(S: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Index of the closest. ``np.argmax`` by default."""
    return np.argmax(S, axis=axis) if CLOSEST_IS_MAX else np.argmin(S, axis=axis)


def farthest_score_fn(S: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Reducer that picks the FARTHEST (un-closest) score.
    ``np.min`` under CLOSEST_IS_MAX=True (since min = farthest)."""
    return np.min(S, axis=axis) if CLOSEST_IS_MAX else np.max(S, axis=axis)


def arg_farthest_score_fn(S: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Index of the farthest. ``np.argmin`` by default."""
    return np.argmin(S, axis=axis) if CLOSEST_IS_MAX else np.argmax(S, axis=axis)


def closest_first_order(S: np.ndarray, axis: int = -1) -> np.ndarray:
    """Indices sorted closest-first (descending under current convention)."""
    return np.argsort(-S, axis=axis) if CLOSEST_IS_MAX else np.argsort(S, axis=axis)


def closest_last_order(S: np.ndarray, axis: int = -1) -> np.ndarray:
    """Indices sorted farthest-first (ascending under current convention)."""
    return np.argsort(S, axis=axis) if CLOSEST_IS_MAX else np.argsort(-S, axis=axis)


def topk_closest_fn(S: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Indices of the ``k`` closest entries (unordered; pipe through
    ``closest_first_order`` on the result for a ranked list)."""
    k = min(k, S.shape[axis])
    target = -S if CLOSEST_IS_MAX else S
    return np.argpartition(target, kth=k - 1, axis=axis).take(
        indices=range(k), axis=axis
    )


def farthest_score_sentinel_fn() -> float:
    """Sentinel for "impossibly far". ``-inf`` under current convention —
    safe to use as a mask value in ``arg_closest_score_fn`` inputs."""
    return float("-inf") if CLOSEST_IS_MAX else float("inf")


def a_closer_than_b_fn(
    a: float | np.ndarray, b: float | np.ndarray,
) -> bool | np.ndarray:
    """True where ``a`` is closer than ``b`` under the convention.

    THE ONLY comparator allowed on score values. Bare ``<`` / ``>``
    on scores is banned anywhere in the code — use this instead.
    """
    return (a > b) if CLOSEST_IS_MAX else (a < b)


def threshold_compare_fn(
    values: np.ndarray, thr: np.ndarray | float,
) -> np.ndarray:
    """Boolean mask — True where ``values`` passes ``thr`` under the
    convention. Equivalent to ``a_closer_than_b_fn(values, thr)`` but
    explicit for threshold-aware predict paths (kills hard-coded ``>``
    in rejection logic)."""
    return a_closer_than_b_fn(values, thr)


def score_to_distance_fn(s: np.ndarray) -> np.ndarray:
    """Monotone flip score → pseudo-distance (min = identical). ``-s``
    under current convention. Use when calling sklearn APIs that want
    a distance-shaped input."""
    return -np.asarray(s) if CLOSEST_IS_MAX else np.asarray(s)


def distance_to_score_fn(d: np.ndarray) -> np.ndarray:
    """Monotone flip distance (min=closest) → convention score
    (higher=closer). ``-d`` under current convention. Inverse of
    ``score_to_distance_fn``. Drop-in for subclasses whose native
    formula is a distance (SRC residual, L2 etc.)."""
    return -np.asarray(d) if CLOSEST_IS_MAX else np.asarray(d)


def closest_k_by_class_fn(
    S: np.ndarray, y: np.ndarray, k: int,
) -> dict[object, np.ndarray]:
    """Per-class indices of the ``k`` closest entries under convention.

    ``S`` : (N,) or (1D over candidates) — scores to each candidate.
    ``y`` : (N,) — class label per candidate.
    Returns ``{label: indices (k_c,)}`` where ``k_c = min(k, n_members)``.
    """
    y = np.asarray(y)
    out: dict[object, np.ndarray] = {}
    for lbl in np.unique(y):
        mask_idx = np.flatnonzero(y == lbl)
        sub = S[mask_idx]
        kk = min(k, sub.size)
        if kk == 0:
            out[lbl] = np.empty(0, dtype=np.int64); continue
        local = topk_closest_fn(sub, k=kk, axis=0)
        out[lbl] = mask_idx[local[closest_first_order(sub[local])]]
    return out


def pair_kernel_closest_mass_fn(
    M: np.ndarray, atol: float = 1e-9,
) -> bool:
    """Convention-aware kernel-peak check — True if every row of the
    pair-kernel matrix ``M`` has its diagonal entry as the closest entry
    under the current convention (``closest_score_fn(M[i]) == M[i, i]``).

    Use inside unit tests or when registering a new ``_pair_kernel``
    override to verify it still obeys ``K(x, x) is closest``.
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(
            f"pair_kernel_closest_mass_fn expects square matrix, got {M.shape}"
        )
    row_best = closest_score_fn(M, axis=1)
    diag = np.diag(M)
    return bool(np.all(
        a_closer_than_b_fn(diag, row_best)
        | (np.abs(diag - row_best) <= atol)
    ))


__all__ = [
    "CLOSEST_IS_MAX",
    "closest_score_fn",
    "arg_closest_score_fn",
    "closest_first_order",
    "closest_last_order",
    "topk_closest_fn",
    "farthest_score_sentinel_fn",
    "a_closer_than_b_fn",
    "threshold_compare_fn",
    "score_to_distance_fn",
    "distance_to_score_fn",
    "closest_k_by_class_fn",
    "pair_kernel_closest_mass_fn",
]
