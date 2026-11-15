"""Shared mathematical utilities for hyperspherical representations.

See the taxonomy in ``summaries/representations/represenations.md``.
Every primitive a method needs that isn't class-specific lives here —
kernels, pair indexing, normalization, spherical k-means.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- norms


def l2_normalize(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Project rows of ``X`` onto the unit hypersphere."""
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.maximum(norm, eps)


# --------------------------------------------------------- pair indexing


def upper_triangle_indices(n: int, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Row/col indices of the strict (k=1) upper triangle of an n×n matrix."""
    return np.triu_indices(n, k=k)


def pair_labels(y: np.ndarray, k: int = 1) -> np.ndarray:
    """Per-pair same-label indicator aligned with ``upper_triangle_indices(n, k)``."""
    y = np.asarray(y)
    ri, ci = upper_triangle_indices(len(y), k=k)
    return (y[ri] == y[ci]).astype(np.int32)


def embedded_pair_matrix(Xn: np.ndarray) -> np.ndarray:
    """Full (N, N) inner-product matrix of the embedded vectors ``Xn``.

    Kernel-AGNOSTIC primitive: just ``Xn @ Xnᵀ``. Whether this matrix
    equals cosine, plain dot product, or something else depends ONLY on
    what ``Representation.transform`` produced. The base class default
    L2-normalizes (so the inner product equals cosine), but a subclass
    that overrides ``transform`` redefines the primitive accordingly.
    """
    return Xn @ Xn.T


# Back-compat alias — old name carried a cosine assumption that the
# new generic name avoids. Prefer ``embedded_pair_matrix`` in new code.
cosine_pair_matrix = embedded_pair_matrix


def pair_scores_return(
    M: np.ndarray, only_upper_triangle: bool
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Format a full (N, N) pair-score matrix for ``score_pairs`` callers.

    ``only_upper_triangle=False`` → return ``M`` as-is (shape ``(N, N)``).
    ``only_upper_triangle=True``  → return ``(ri, ci, M[ri, ci])`` with
    the strict upper triangle (``k=1``).
    """
    if only_upper_triangle:
        ri, ci = upper_triangle_indices(M.shape[0], k=1)
        return ri, ci, M[ri, ci]
    return M


# ---------------------------------------------------------------- kernels


def arc_cosine_kernel(cos_sim: np.ndarray, n: int = 1) -> np.ndarray:
    """Arc-cosine kernel of order ``n`` ∈ {0, 1, 2}: :math:`J_n(\\theta)/\\pi`.

    * ``n=0``: step kernel (Cho & Saul 2009).
    * ``n=1``: NTK of a 1-hidden-layer ReLU net (Jacot et al. 2018).
    * ``n=2``: quadratic-ReLU analogue.
    """
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    theta = np.arccos(cos_sim)
    sin_t = np.sin(theta)
    if n == 0:
        J = np.pi - theta
    elif n == 1:
        J = sin_t + (np.pi - theta) * cos_sim
    elif n == 2:
        J = 3.0 * sin_t * cos_sim + (np.pi - theta) * (1.0 + 2.0 * cos_sim ** 2)
    else:
        raise ValueError(f"arc_cosine_kernel supports n in {{0, 1, 2}}, got {n}")
    return J / np.pi


def bingham_geodesic_kernel(
    cos_sim: np.ndarray, gamma: float = 4.0, p: float = 2.0
) -> np.ndarray:
    """Bingham–Geodesic kernel :math:`\\exp(-\\gamma (\\arccos(|x^Ty|)/\\pi)^p)`.

    Antipodally invariant — x ≡ -x under this kernel. Use when polysemy
    is suspected to be antipodal; prefer arc-cosine when sign carries
    signal (e.g. speaker / face embeddings). See theory.md Part II.
    """
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    theta = np.arccos(np.abs(cos_sim))
    return np.exp(-gamma * (theta / np.pi) ** p)


# -------------------------------------------------------- spherical k-means


def spherical_kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 50,
    tol: float = 1e-4,
    random_state: int = 42,
) -> np.ndarray:
    """Mini spherical k-means — rows of ``X`` are assumed L2-normalized.

    Returns ``(k, D)`` L2-normalized centers. Initialization by random
    draw from ``X``; centers are re-normalized after each Lloyd step.
    Used by C.3 Multi-Proxy and III.4 AC-PAN to place proxies per class.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    if k >= n:
        return l2_normalize(X)
    from .base import arg_closest_score_fn, closest_score_fn
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    prev_assign = np.full(n, -1)
    prev_inertia = -np.inf
    for _ in range(max_iter):
        sims = X @ centers.T
        assign = arg_closest_score_fn(sims, axis=1)
        if np.array_equal(assign, prev_assign):
            break
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = X[mask].sum(axis=0)
            else:
                centers[j] = X[rng.integers(n)]
        centers = l2_normalize(centers)
        cur_inertia = float(closest_score_fn(sims, axis=1).mean())
        if abs(cur_inertia - prev_inertia) < tol:
            prev_assign = assign
        prev_inertia = cur_inertia
    return centers


# ----------------------------------------------------------- vote helper


def weighted_label_vote(
    labels: np.ndarray,
    weights: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Aggregate ``weights`` (N, M) over same-label columns → ``(N, K)``.

    ``labels[m]`` is the class label for weight column ``m`` (shape M).
    ``classes`` lists the K unique class labels (caller-provided so
    class order is stable across methods).
    """
    out = np.zeros((weights.shape[0], len(classes)), dtype=weights.dtype)
    for ki, cls in enumerate(classes):
        mask = labels == cls
        if mask.any():
            out[:, ki] = weights[:, mask].sum(axis=1)
    return out


def assert_polarity(rep, X: np.ndarray, atol: float = 1e-9) -> None:
    """Enforce the universal sign convention — HIGHER SCORE ⇒ CLOSER.

    Checks three invariants on a fitted ``rep``:
      1. ``score_xy(x, x) ≥ score_xy(x, y)`` for every ``y`` in ``X``.
      2. Diagonal of ``score_pairs(X)`` equals the per-row maximum.
      3. ``_score_per_class_raw`` of a training sample peaks at that
         sample's true class (checked only when ``rep._labels`` is set
         and ``y_of_X`` can be inferred from the prototype layout).

    Raises ``AssertionError`` with a per-rep diagnostic if any invariant
    is violated. Use to gate a new methodology at test time.
    """
    from .base import closest_score_fn, a_closer_than_b_fn
    Xn = np.asarray(X, dtype=np.float64)
    # Invariant 1: self-score is the CLOSEST entry of row i of score_xy(X, X).
    S = np.asarray(rep.score_xy(Xn, Xn), dtype=np.float64)
    row_best = closest_score_fn(S, axis=1)
    for i in range(S.shape[0]):
        ok = bool(a_closer_than_b_fn(S[i, i], row_best[i])) or abs(S[i, i] - row_best[i]) <= atol
        assert ok, (
            f"{type(rep).__name__}: polarity violated on score_xy — row {i} "
            f"self-score {S[i, i]:.6f} is not the CLOSEST entry "
            f"(row best {row_best[i]:.6f}). Invert the kernel so higher = closer."
        )
    # Invariant 2: score_pairs has the same property.
    P = rep.score_pairs(Xn)
    if not isinstance(P, tuple):
        P = np.asarray(P, dtype=np.float64)
        row_best = closest_score_fn(P, axis=1)
        for i in range(P.shape[0]):
            ok = bool(a_closer_than_b_fn(P[i, i], row_best[i])) or abs(P[i, i] - row_best[i]) <= atol
            assert ok, (
                f"{type(rep).__name__}: polarity violated on score_pairs — "
                f"row {i} diag {P[i, i]:.6f} not the CLOSEST (best {row_best[i]:.6f})."
            )


__all__ = [
    "l2_normalize",
    "upper_triangle_indices",
    "pair_labels",
    "embedded_pair_matrix",
    "cosine_pair_matrix",          # deprecated alias for back-compat
    "pair_scores_return",
    "arc_cosine_kernel",
    "bingham_geodesic_kernel",
    "spherical_kmeans",
    "weighted_label_vote",
    "assert_polarity",
]
