"""Aggregator resolver + centroid-solver dispatcher.

Used by prototype-building representations (NCM, MultiProxyAnchor,
ACPAN, VMFMixture). The aggregator is REQUIRED at fit-time for these
reps (caller passes it explicitly; ``Representation`` does not store
a public field).

Centroid extraction routes through a SOLVER CHAIN ordered by accuracy:

    A → B → F                     (with F always last)

where each tier attempts to solve

    c* = rep.arg_closest_score_fn_c   agg_i   score_xy(c, x_i)

over c ∈ ℝᴰ. If a tier doesn't apply (kernel mismatch / aggregator
mismatch / divergence), the dispatcher falls through to the next.

  • A — Cauchy-Schwarz CLOSED FORM. Cosine kernel + ``mean`` only.
        Global optimum. ``c* = sign · Σ x_i`` where ``sign`` flips with
        ``rep.CLOSEST_IS_MAX``. Linearity of mean is the prerequisite —
        median / min / max / percentile / trimmed_mean / callable do
        NOT fit A.
  • B — Weiszfeld IRLS. Cosine kernel + ``median`` only. Global on a
        convex problem. Iterative-reweighted least squares warm-started
        from the L2 mean.
  • F — DISCRETE MEDOID. Universal fallback. Direction = score-optimal
        class member ``Xn[i*]`` where
        ``i* = rep.arg_closest_score_fn(reducer(rep._pair_kernel(Xn Xnᵀ), axis=1))``.
        Works for ANY (kernel, aggregator) combo. Optimal among class
        members; suboptimal vs unconstrained ℝᴰ.

C / D / E (gradient ascent / Frank-Wolfe / sub-gradient) are NOT
implemented here. Document them as future-work for non-cosine kernels
(arc-cosine, Bingham) or non-smooth aggregators where A/B/F leave
optimality on the table.

KERNEL-AGNOSTIC contract: nothing assumes cosine. F always honors the
rep's own ``_pair_kernel``. A/B activate ONLY when the kernel reduces
to identity on the embedded-pair primitive (probed at runtime).

Magnitude: every solver's result is rescaled to
``target_mag = ‖Σ x_i‖ / N`` so PCA display is uniform across solvers
(cosine kernel is magnitude-invariant — only display changes).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Aggregator resolver — string spec / callable → (name, reducer fn).
# ─────────────────────────────────────────────────────────────────────
def aggregator_reducer_fn(
    agg: str | Callable[[np.ndarray, int], np.ndarray],
) -> tuple[str, Callable[[np.ndarray, int], np.ndarray]]:
    """Resolve an aggregator spec → (name, reducer) pair.

    Accepted specs:
      • ``"mean"``            — arithmetic mean along axis.
      • ``"median"``          — median along axis.
      • ``"min"``              — minimum along axis (most-pessimistic).
      • ``"max"``              — maximum along axis (most-optimistic).
      • ``"percentile:<q>"``   — qth percentile, ``q ∈ [0, 100]``.
      • ``"trimmed_mean:<p>"`` — trimmed mean, ``p ∈ [0, 50)``.
      • callable ``f(S, axis)`` — any reducer returning a vector.

    Raises ``ValueError`` on unknown spec.
    """
    if callable(agg):
        return ("callable", agg)
    s = str(agg).strip().lower()
    if s == "mean":
        return ("mean",   lambda S, axis: np.mean(S, axis=axis))
    if s == "median":
        return ("median", lambda S, axis: np.median(S, axis=axis))
    if s == "min":
        return ("min",    lambda S, axis: np.min(S, axis=axis))
    if s == "max":
        return ("max",    lambda S, axis: np.max(S, axis=axis))
    if s.startswith("percentile:"):
        q = float(s.split(":", 1)[1])
        if not (0.0 <= q <= 100.0):
            raise ValueError(f"percentile q must be in [0, 100], got {q}")
        return (f"percentile:{q}", lambda S, axis: np.percentile(S, q, axis=axis))
    if s.startswith("trimmed_mean:"):
        p = float(s.split(":", 1)[1])
        if not (0.0 <= p < 50.0):
            raise ValueError(f"trimmed_mean p must be in [0, 50), got {p}")
        from scipy.stats import trim_mean
        frac = p / 100.0
        return (f"trimmed_mean:{p}",
                lambda S, axis: trim_mean(S, proportiontocut=frac, axis=axis))
    raise ValueError(
        f"unknown aggregator={agg!r}; expected 'mean' | 'median' | 'min' | "
        f"'max' | 'percentile:<q>' | 'trimmed_mean:<p>' | callable"
    )


# ─────────────────────────────────────────────────────────────────────
# Kernel detection — runtime probe.
# ─────────────────────────────────────────────────────────────────────
def _kernel_is_identity(rep, Xn: np.ndarray) -> bool:
    """Probe whether ``rep._pair_kernel`` is identity on the embedded
    pair matrix.

    Used by the dispatcher to gate A / B closed forms — they assume
    cosine structure. Reps that warp the embedded primitive (ArcCosine,
    Bingham) override ``_pair_kernel`` and the probe rejects them.

    Bypasses ``rep.score_xy`` (which calls ``_require_fit`` and raises
    while ``_centroid_under_score`` runs *inside* ``fit``). Uses two
    distinct vectors so off-diagonal entries reveal the warp.
    """
    if Xn.shape[0] < 2:
        return True                                    # can't probe with < 2 vectors
    sub = Xn[:2]                                       # 2 distinct unit vectors
    M = sub @ sub.T                                    # 2x2 inner product
    try:
        M_warped = np.asarray(rep._pair_kernel(M, sub), dtype=np.float64)
    except Exception:
        return False                                   # kernel failed → not safe
    return bool(np.allclose(M_warped, M, atol=1e-9))


# ─────────────────────────────────────────────────────────────────────
# Solver A — Cauchy-Schwarz closed form.
#
# Optimization problem:
#   maximize_c  (1/N) Σ_i  cos(c, x_i)   over unit c
#            =  c · m̄ / ‖c‖             where m̄ = (1/N) Σ x_i
#         ≤    ‖m̄‖                       (Cauchy-Schwarz, equality iff c ∥ m̄)
#
# Solution: c* = sign · Σ x_i. Direction is the Karcher/Fréchet mean;
# magnitude is free (cosine is magnitude-invariant). Sign flips with
# ``rep.CLOSEST_IS_MAX`` so the result is the convention-closest c.
#
# Equivalent to least-squares Euclidean centroid in the embedded space —
# LS minimizes ``Σ ‖c − x_i‖²`` whose closed form is also ``Σ x_i / N``.
# Same family of "linear-in-scores" closed forms.
# ─────────────────────────────────────────────────────────────────────
def _solver_A_closed_form_mean(rep, Xn: np.ndarray, agg: str | Callable,
                               eps: float = 1e-12) -> np.ndarray | None:
    """Returns ``None`` if not applicable: requires cosine kernel +
    ``aggregator='mean'``. Linearity of the aggregator is the
    prerequisite — median / min / max / percentile do NOT fit A.
    """
    if not isinstance(agg, str) or agg.strip().lower() != "mean":
        return None
    if not _kernel_is_identity(rep, Xn):
        return None
    m = Xn.sum(axis=0)
    if float(np.linalg.norm(m)) < eps:
        return None
    sign = 1.0 if rep.CLOSEST_IS_MAX else -1.0
    return sign * m


# ─────────────────────────────────────────────────────────────────────
# Solver B — Weiszfeld IRLS for the geometric median.
#
# Optimization problem:
#   minimize_c  Σ_i ‖c − x_i‖           (Euclidean geometric median)
#         ↔   maximize_c  median_i cos(c, x_i)   on the sphere
#
# IRLS update — at each iter, weight inversely by current distance:
#   w_i = 1 / max(‖c_t − x_i‖, eps)
#   c_{t+1} = (Σ_i w_i x_i) / (Σ_i w_i)
#
# Warm-started from ``Σ x_i`` (the Solver A direction). Polarity-aware
# via sign flip on the result. Convex problem → IRLS converges to the
# global optimum.
# ─────────────────────────────────────────────────────────────────────
def _solver_B_weiszfeld_median(
    rep, Xn: np.ndarray, agg: str | Callable,
    max_iter: int = 64, tol: float = 1e-9, eps: float = 1e-12,
) -> np.ndarray | None:
    """Returns ``None`` if not applicable: requires cosine kernel +
    ``aggregator='median'``."""
    if not isinstance(agg, str) or agg.strip().lower() != "median":
        return None
    if not _kernel_is_identity(rep, Xn):
        return None
    c = Xn.sum(axis=0)
    if float(np.linalg.norm(c)) < eps:
        return None
    for _ in range(max_iter):
        d = np.maximum(np.linalg.norm(Xn - c, axis=1), eps)
        w = 1.0 / d
        c_new = (Xn * w[:, None]).sum(axis=0) / w.sum()
        if float(np.linalg.norm(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    sign = 1.0 if rep.CLOSEST_IS_MAX else -1.0
    return sign * c


# ─────────────────────────────────────────────────────────────────────
# Solver F — Discrete medoid. UNIVERSAL fallback.
#
#   For each candidate i ∈ class:
#       agg_score_i = aggregator_reducer_fn(agg)(rep._pair_kernel(Xn Xnᵀ)[i, :])
#   i* = rep.arg_closest_score_fn(agg_score_i)
#   c* = Xn[i*]
#
# Works for ANY (kernel, aggregator) combo. No smoothness, no
# linearity, no concavity assumptions. Optimal AMONG class members;
# suboptimal vs unconstrained ℝᴰ when a higher solver applies.
# ─────────────────────────────────────────────────────────────────────
def _solver_F_medoid(rep, Xn: np.ndarray, agg: str | Callable) -> np.ndarray:
    """Always succeeds. Returns unit-norm class member; caller rescales."""
    _, reducer = aggregator_reducer_fn(agg)
    S = rep._pair_kernel(Xn @ Xn.T, Xn)                  # (N, N)
    agg_scores = np.asarray(reducer(S, axis=1), dtype=np.float64)
    i_star = int(rep.arg_closest_score_fn(agg_scores))
    return Xn[i_star]


# Solver chain: best → fallback. F always last.
# A and B return None when not applicable; dispatcher continues down
# the chain. F always succeeds.
_SOLVER_CHAIN: tuple = (
    _solver_A_closed_form_mean,
    _solver_B_weiszfeld_median,
)


def _eval_centroid_score(rep, c: np.ndarray, Xn: np.ndarray,
                         aggregator, eps: float = 1e-12) -> float:
    """Aggregate score of a candidate centroid against the class.

    Routes through ``rep._pair_kernel`` (the kernel hook every rep
    declares) instead of ``rep.score_xy`` so this works DURING ``fit``
    — ``score_xy`` triggers ``_require_fit`` which would raise before
    ``_mark_fitted`` runs.

    Pipeline mirrors ``score_xy`` minus the fit gate:
      • normalize c to the embedded space (idempotent for unit-norm)
      • build the embedded-pair primitive ``c · Xnᵀ`` (unit cosine)
      • apply ``rep._pair_kernel`` (kernel-aware: cosine for NCM/VMF/
        MultiProxy, arc-cosine for ACPAN, etc.)
      • reduce via the aggregator's reducer
    """
    c_arr = np.atleast_2d(np.asarray(c, dtype=np.float64))
    nrm = np.maximum(np.linalg.norm(c_arr, axis=1, keepdims=True), eps)
    c_unit = c_arr / nrm                              # (1, D)
    M_pair = c_unit @ Xn.T                             # (1, N)
    S = rep._pair_kernel(M_pair, c_unit)               # rep's kernel
    s = np.asarray(S, dtype=np.float64).reshape(-1)
    _, reducer = aggregator_reducer_fn(aggregator)
    return float(reducer(s, axis=0))


def centroid_under_score(
    rep,
    X: np.ndarray,
    aggregator: str | Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Solve ``c* = arg_closest_c agg_i score_xy(c, x_i)`` over ℝᴰ.

    Dispatches through the solver chain ``A → B → F`` (best → fallback).
    Each higher-tier solver is GATED by an explicit comparison against
    F under the rep's own kernel: a closed-form result is accepted
    only when ``rep.a_closer_than_b_fn(agg_score(c_solver), agg_score(c_F))``.
    Otherwise dispatcher falls through to the next tier (and ultimately
    to F, which always wins by definition over its own output).

    Helpers used (all routed through ``rep`` instance):
      • ``rep.transform``                  embed input class
      • ``rep._pair_kernel``               kernel for pair scores
      • ``rep.score_xy``                   score for solver verification
      • ``rep.arg_closest_score_fn``       polarity-aware argmax (in F)
      • ``rep.a_closer_than_b_fn``         polarity-aware comparison
      • ``aggregator_reducer_fn(agg)``     reducer over scores

    All solver outputs are rescaled to ``target_mag = ‖Σ x_i‖ / N`` so
    PCA display is uniform across paths. Cosine-family kernels are
    magnitude-invariant via ``score_xy → transform`` re-normalization,
    so rescaling has zero impact on predict / score / fit; only display
    changes.

    ``aggregator`` is REQUIRED — ``None`` raises ``ValueError``. Helper
    runs inside subclass ``fit``, so ``_require_fit`` is bypassed.
    """
    if aggregator is None:
        raise ValueError(
            f"{type(rep).__name__}.centroid_under_score: aggregator must "
            f"be provided (None is invalid). Pass it via fit()."
        )
    Xn = rep.transform(X)
    if Xn.shape[0] == 0:
        raise ValueError("centroid_under_score: empty class")
    if Xn.shape[0] == 1:
        return Xn[0].copy()

    target_mag = float(np.linalg.norm(Xn.sum(axis=0)) / Xn.shape[0])

    def _rescale(c: np.ndarray) -> np.ndarray:
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-12 or target_mag < 1e-12:
            return c
        return (target_mag / nrm) * c

    # Always compute F first as the safety baseline + score under rep.
    c_F = _solver_F_medoid(rep, Xn, aggregator)
    score_F = _eval_centroid_score(rep, c_F, Xn, aggregator)

    # Try higher-tier solvers; accept ONLY if they beat F under rep.
    for solver in _SOLVER_CHAIN:
        try:
            c = solver(rep, Xn, aggregator)
        except Exception:
            c = None
        if c is None:
            continue
        try:
            score_c = _eval_centroid_score(rep, c, Xn, aggregator)
        except Exception:
            continue
        if rep.a_closer_than_b_fn(score_c, score_F):
            return _rescale(c)

    # F wins by default — universal fallback.
    return _rescale(c_F)


__all__ = ["aggregator_reducer_fn", "centroid_under_score"]
