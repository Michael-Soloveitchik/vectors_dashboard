"""Base class for hyperspherical representations.

Canonical contract — every representation implements the same API:

    rep = Cls(hyperparams_only)
    rep.fit(X, y, hints=None, aggregator=None)
                                              # returns self; stores self._labels.
                                              # Prototype-building reps REQUIRE
                                              # aggregator (raise on None); other
                                              # reps accept and ignore the kwarg.

    rep.transform(X) -> (N, D)                # embedding (L2-norm default)

    rep._score_per_class_raw(X) -> (N, K)     # raw per-class scores aligned
                                              # to self._labels. Subclass-
                                              # specific: cosine, kernel vote,
                                              # negative residual, svm decision,
                                              # already-a-posterior, etc.

    rep.predict(X, thresholds=None)
        -> (cls_arr, score_arr, votes_arr)    # cls_arr (N,) object: predicted
                                              # class or None when no threshold
                                              # passed. score_arr (N,) object:
                                              # winning raw score or None when
                                              # cls is None. votes_arr (N, K)
                                              # RAW per-class scores aligned to
                                              # class_order_fn() — populated
                                              # for every row. Callers needing
                                              # [0, 1] apply rep.normalize_score.

    rep.score_pairs(X, only_upper_triangle=False, cos_similarity=False)
        -> (N, N) or (ri, ci, vals)           # cos_similarity=True: plain
                                              # cosine. False: rep's kernel
                                              # via _pair_kernel().

    rep.get_centroids() -> {label: (n_i, D)}  # on-sphere prototypes, or {}

    rep.normalize_score(s) -> s'              # monotone -> [0, 1] for ROC
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .math_utils import l2_normalize, pair_scores_return

# Sign-convention helpers live in convention.py; aggregator resolver +
# centroid helper in aggregator_utils.py. They are re-exported below
# for backward compatibility — existing callers import them from .base.
from .convention import (
    CLOSEST_IS_MAX,
    closest_score_fn,
    arg_closest_score_fn,
    closest_first_order,
    closest_last_order,
    topk_closest_fn,
    farthest_score_fn,
    arg_farthest_score_fn,
    farthest_score_sentinel_fn,
    a_closer_than_b_fn,
    threshold_compare_fn,
    score_to_distance_fn,
    distance_to_score_fn,
    closest_k_by_class_fn,
    pair_kernel_closest_mass_fn,
)
from .aggregator_utils import aggregator_reducer_fn, centroid_under_score


class Representation(ABC):
    """Abstract hyperspherical representation.

    =========================================================================
    SIGN CONVENTION  (non-negotiable, enforced across every methodology)
    =========================================================================

    HIGHER SCORE ⇒ CLOSER / MORE-SIMILAR. LOWER SCORE ⇒ FARTHER / LESS-SIMILAR.

    This holds for ALL scoring surfaces — ``score_xy``, ``_pair_kernel``,
    ``_score_per_class_raw``, ``score_pairs``, ``predict`` (which now
    always returns RAW per-class scores; there is no ``raw`` flag).
    If a subclass' native formula is a *distance*, it MUST be inverted
    (negate, ``exp(-·)``, or ``1 − d``) before return so the convention
    holds. ``normalize_score`` is monotone non-decreasing and preserves
    direction.

    **All reductions/sorts over scores MUST route through the four
    convention helpers** — never call ``.max`` / ``np.argmax`` /
    ``argpartition(-S,…)`` / ``np.argsort`` directly on a score array:

        closest_score_fn(S, axis)       # the reducer that picks the closest
        arg_closest_score_fn(S, axis)   # index of the closest
        closest_first_order(S, axis)    # argsort closest-first (descending)
        closest_last_order(S, axis)     # argsort farthest-first (ascending)
        topk_closest_fn(S, k, axis)     # indices of the k closest
        farthest_score_sentinel_fn()    # mask value for "impossibly far"
        a_closer_than_b_fn(a, b)        # comparator — REPLACES <, >
        threshold_compare_fn(vals, t)   # mask; uses a_closer_than_b_fn
        score_to_distance_fn(s)         # convention → sklearn distance
        distance_to_score_fn(d)         # sklearn distance → convention
        closest_k_by_class_fn(S, y, k)  # per-class top-k indices
        pair_kernel_closest_mass_fn(M)  # diag-is-closest kernel check
        class_order_fn()                # read-only frozen label order
        self_consistency_check_fn(X)    # full post-fit diagnostic
        score_XY(X, Y)                  # batch × batch, via score_xy

    These switch direction from ``CLOSEST_IS_MAX = True`` if the
    convention ever flips in future — ONE place to change.

    ``score_xy`` is the ONE pair primitive. Every derived score —
    per-class, pairwise, predict output — must be expressed as
    reductions over ``score_xy`` via the helpers above. No shortcuts.

    Kernel directions (per-family):
      • cosine, arc-cosine, Bingham-geodesic — higher = closer.
      • RBF on cosine distance (LabelProp) — higher = closer, ``exp(−γ d²)``
        peaks at ``d = 0``.
      • SRC — returns ``−||residual||`` (less-negative = closer).
      • SVM decision — higher margin toward class = closer to that class.

    Audit invariant, post-fit:
        ``score_xy(x, x) >= score_xy(x, y)``  for any ``y``.
    Diagonals of ``score_pairs`` = per-row max.

    =========================================================================
    PUBLIC METHOD SURFACE  (what you can call from outside)
    =========================================================================

    ``fit(X, y, hints=None)`` → self
        Learn prototypes / kernel state / model weights.
        ↳ uses:    ``transform`` (per-sample embedding);
                   ``_centroid_under_score`` (for prototype-building reps
                   honoring ``aggregator``).
        ↳ must call: ``_mark_fitted()`` at the very end.

    ``transform(X)`` → (N, D) unit-norm
        Embed raw vectors onto the sphere (L2-normalize by default).
        ↳ uses:    ``math_utils.l2_normalize``.
        ↳ called by: ``score_xy``, ``score_pairs``, ``fit``, every
                     subclass scoring path.

    ``score_xy(x, y)`` → float | ndarray       ← THE PAIR PRIMITIVE
        Similarity between two RAW (un-normalized) vectors, 1-D or 2-D.
        Applies the full pipeline: embed + kernel. Returns scalar for
        (D,)×(D,), 1-D for (N,D)×(D,), 2-D (N, M) for batch × batch.
        ↳ uses:    ``_require_fit``, ``transform``, ``_pair_kernel``.
        ↳ called by: every subclass ``_score_per_class_raw`` (prototype
                     scoring, k-NN / SVM Gram rows, kernel voting).
        ↳ may be overridden when pair scoring is fundamentally non-cosine
          (e.g. LabelPropagation uses RBF in the embedded space).

    ``predict(X, thresholds=None)`` → (cls_arr, score_arr, votes_arr)
        Three length-N arrays. ``cls_arr`` and ``score_arr`` are object
        arrays — entries are ``None`` for samples that fail every
        threshold (when ``thresholds`` is supplied). ``votes_arr`` is a
        ``(N, K)`` float matrix of RAW per-class scores aligned to
        ``class_order_fn()`` — populated for every row regardless of
        threshold rejection. Callers that want posterior/[0,1] apply
        ``rep.normalize_score`` themselves (monotone, AUC-preserving).
        ↳ uses:    ``_require_fit``, ``_score_per_class_raw``.

    ``score_pairs(X, only_upper_triangle=False, cos_similarity=False)``
        Pairwise matrix over rows of X. Representation-agnostic output
        for AUC / ROC use.
        ↳ uses:    ``_require_fit``, ``transform``,
                   ``math_utils.cosine_pair_matrix``, ``_pair_kernel``,
                   ``math_utils.pair_scores_return``.

    ``get_centroids()`` → ``{label: (n_i, D)}``
        On-sphere prototypes, or ``{}`` when the rep has none (k-NN, SVM,
        label propagation, SRC).
        ↳ uses:    ``_require_fit`` in subclasses that override.

    ``normalize_score(scores)`` → scores'
        Rank-transform to [0, 1] — monotone, AUC-preserving.
        ↳ uses:    ``scipy.stats.rankdata``.
        ↳ some subclasses override to clip to the kernel's native range
          (arc-cosine, Bingham, KNN).

    =========================================================================
    INTERNAL EXTENSION POINTS  (subclasses implement these)
    =========================================================================

    ``_score_per_class_raw(X)`` → (N, K)   ABSTRACT
        Raw per-class scores aligned to ``self._labels``. Free-form:
        cosine against prototypes, k-NN vote, kernel max, neg-residual,
        SVM decision, or already-normalized posterior (subclass sets
        ``_raw_is_posterior = True`` in that case).
        ↳ every subclass routes sample-vs-prototype comparisons through
          ``score_xy`` so the pair primitive remains the single source
          of truth.

    ``_pair_kernel(M_pair, Xn)``
        Transform the cosine pair-matrix into the rep's native kernel.
        Identity by default; overridden by arc-cosine (``J_n``) and
        Bingham-Geodesic kernels.
        ↳ called by: ``score_xy``, ``score_pairs``.

    =========================================================================
    BASE-CLASS HELPERS  (inherited — you rarely override)
    =========================================================================

    ``_require_fit()``
        Raises ``RuntimeError`` if called before ``fit``.
        ↳ called by: ``score_xy``, ``predict``, ``score_pairs``, and
                     every subclass ``_score_per_class_raw`` /
                     ``get_centroids`` override.

    ``_mark_fitted()``
        Trips the ``_fitted`` flag. Must be the last line of every
        subclass ``fit``.

    ``_posterior_matrix(X)`` → (N, K) simplex
        Softmax over ``_score_per_class_raw`` (or clip-renormalize when
        ``_raw_is_posterior`` is set).
        ↳ called by: ``predict`` when ``raw=False``.

    ``_centroid_under_score(X, aggregator=None)`` → (D,) unit-norm
        Returns the on-sphere prototype whose aggregated ``score_xy``
        against the class is maximal under the chosen aggregator:
          • ``"mean"``   → ``argmax_c  mean_i  score_xy(c, x_i)``
          • ``"median"`` → ``argmax_c  median_i score_xy(c, x_i)``
        Candidates = the class members (medoid under the rep's kernel,
        built from ``_pair_kernel(Xn @ Xnᵀ, Xn)`` — same pipeline as
        ``score_xy``, minus the fit gate).
        ↳ uses:    ``transform``, ``_pair_kernel``.
        ↳ called by: every prototype-building subclass ``fit``
                     (NCM, VMFMixture, MultiProxyAnchor, ACPAN).

    =========================================================================
    STATE
    =========================================================================

    ``_fitted: bool``       — tripped by ``_mark_fitted()``; gates every
                              public method via ``_require_fit``.
    ``_labels: ndarray``    — class labels in the column order produced
                              by ``_score_per_class_raw``.
    ``_raw_is_posterior``   — subclass flag: when True the raw scores
                              are already a simplex and ``predict``
                              skips the softmax.
    ``_uses_aggregator: bool`` — class-level capability flag. True for
                              representations whose ``fit`` requires an
                              explicit aggregator (NCM, MultiProxyAnchor,
                              ACPAN, VMFMixture). Frontend uses this to
                              decide whether to enable the mean/median
                              toggle.
    ``_aggregator``         — the aggregator spec frozen at fit-time.
                              Read via ``get_aggregator()``; written
                              ONLY by ``fit``. Consumers
                              (``_centroid_under_score``) take it as an
                              explicit argument — no fallback to a
                              stored field.
    """

    # Subclass flag: True when `_score_per_class_raw` already returns a
    # proper simplex (rows sum to 1, non-negative). The base class skips
    # the softmax. Default False -> raw scores get softmaxed.
    _raw_is_posterior: bool = False

    # Set by `fit` in every subclass. Ordering defines the column order
    # of `_score_per_class_raw` output.
    _labels: np.ndarray | None = None

    # Tripped by fit() at the end of every subclass. Readers (predict,
    # score, score_xy, score_pairs, get_centroids) must call
    # ``_require_fit`` before doing anything stateful.
    _fitted: bool = False

    # Capability flag — True for representations whose fit() requires an
    # explicit aggregator (NCM, MultiProxyAnchor, ACPAN, VMFMixture).
    # Reps that don't build score-medoid prototypes leave it False; their
    # fit() accepts the aggregator kwarg for interface uniformity but
    # ignores it.
    _uses_aggregator: bool = False

    # Frozen at fit-time inside subclasses that consume aggregator. Read
    # via ``get_aggregator()``. There is NO public setter — the only way
    # to change it is to re-call fit().
    _aggregator: str | None = None

    # Polarity flag — exposed as a class attribute for subclass introspection
    # ONLY. The single source of truth is the module-level ``CLOSEST_IS_MAX``
    # constant above; the helpers below read that. Do NOT shadow on subclasses.
    CLOSEST_IS_MAX: bool = CLOSEST_IS_MAX

    # ── closeness convention helpers — delegates to module-level so
    #    backend/frontend callers share the EXACT same single switchpoint.
    closest_score_fn             = staticmethod(closest_score_fn)
    arg_closest_score_fn         = staticmethod(arg_closest_score_fn)
    closest_first_order          = staticmethod(closest_first_order)
    closest_last_order           = staticmethod(closest_last_order)
    topk_closest_fn              = staticmethod(topk_closest_fn)
    farthest_score_fn            = staticmethod(farthest_score_fn)
    arg_farthest_score_fn        = staticmethod(arg_farthest_score_fn)
    farthest_score_sentinel_fn   = staticmethod(farthest_score_sentinel_fn)
    a_closer_than_b_fn           = staticmethod(a_closer_than_b_fn)
    threshold_compare_fn         = staticmethod(threshold_compare_fn)
    score_to_distance_fn         = staticmethod(score_to_distance_fn)
    distance_to_score_fn         = staticmethod(distance_to_score_fn)
    closest_k_by_class_fn        = staticmethod(closest_k_by_class_fn)
    pair_kernel_closest_mass_fn  = staticmethod(pair_kernel_closest_mass_fn)

    # ── fit bookkeeping ────────────────────────────────────────────
    def _require_fit(self) -> None:
        """Raise if this method is called before ``fit``.

        Every public entry point that depends on learned state
        (``predict``, ``score_xy``, ``score_pairs``, ``get_centroids``)
        calls this first.
        """
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__}: fit(X, y) must run before this "
                f"method. The representation is not ready — nothing to "
                f"score against."
            )

    def _mark_fitted(self, X_tr: np.ndarray | None = None) -> None:
        """Subclasses call this at the end of fit().

        When ``X_tr`` is supplied (recommended) we:
          1. Capture the training raw-score distribution as
             ``self._norm_ref`` — the reference grid ``normalize_score``
             ranks test scores against.
          2. Run a polarity sanity check (``HIGHER = CLOSER``) on a
             small sample of ``X_tr``. Fails LOUD at fit time, never
             silently at inference.
          3. Freeze the label order — subsequent calls to predict
             return scores in ``self._labels`` order and that order
             does NOT change until the next ``fit``.
        """
        self._fitted = True
        if X_tr is not None:
            try:
                raw = np.asarray(self._score_per_class_raw(X_tr),
                                 dtype=np.float64).reshape(-1)
                raw = raw[np.isfinite(raw)]
                if raw.size > 0:
                    # Stored in DISTANCE domain via score_to_distance_fn.
                    # Sorted ascending → closer-first under EITHER polarity.
                    # normalize_score does the polarity-symmetric lookup.
                    d = self.score_to_distance_fn(raw)
                    self._norm_ref = np.sort(d)
            except Exception:
                # Never let a normalization-cache failure break fit.
                pass
            # Polarity sanity — cheap on a handful of rows.
            try:
                from .math_utils import assert_polarity
                n_probe = min(32, X_tr.shape[0])
                assert_polarity(self, X_tr[:n_probe])
            except AssertionError:
                raise                                        # LOUD: polarity must hold
            except Exception:
                pass                                         # non-polarity hiccups ignored
        # Freeze the label order — store immutable snapshot.
        if self._labels is not None:
            self._labels = np.asarray(self._labels)
            self._labels.setflags(write=False)

    # ── pair / batch scoring — the single source of truth ──────────
    def score_xy(
        self, x: np.ndarray, y: np.ndarray,
    ) -> float | np.ndarray:
        """Similarity between TWO RAW (un-normalized) vectors.

        The sole scoring primitive. It applies the representation's full
        pipeline — embedding (``transform``) + kernel (``_pair_kernel``)
        — on any two raw vectors. Every method that scores pairs must
        route through this function; ``predict`` and ``score`` call it.

        Parameters
        ----------
        x, y : array-like
            Raw vectors. 1-D ``(D,)`` or 2-D ``(N, D)`` / ``(M, D)``.
        Returns
        -------
        float for (D,) x (D,), 1-D for (N, D) x (D,) or (D,) x (M, D),
        2-D (N, M) for batch x batch.
        Raises
        ------
        RuntimeError if called before ``fit()``.
        """
        self._require_fit()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x1 = x.ndim == 1
        y1 = y.ndim == 1
        if x1:
            x = x[None, :]
        if y1:
            y = y[None, :]
        xn = self.transform(x)
        yn = self.transform(y)
        M_pair = xn @ yn.T
        # Route through the rep's native kernel — arc-cosine / Bingham /
        # identity — exactly as score_pairs already does. Subclasses with
        # fundamentally non-kernel scoring (KNN vote, SVM decision) override
        # score_xy directly.
        M = self._pair_kernel(M_pair, xn)
        if x1 and y1:
            return float(M[0, 0])
        if x1:
            return np.asarray(M[0], dtype=np.float64)
        if y1:
            return np.asarray(M[:, 0], dtype=np.float64)
        return np.asarray(M, dtype=np.float64)

    # ── score_XY — explicit batch-×-batch alias, routed through score_xy ──
    def score_XY(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Batch-explicit alias of ``score_xy``. ``X`` shape ``(N, D)``,
        ``Y`` shape ``(M, D)`` → ``(N, M)`` score matrix. Uses the same
        convention (higher = closer) and the SAME pipeline as ``score_xy``
        — this is a lint-enforceable entry point for batch-scoring code
        that never wants to worry about scalar/1-D degenerate shapes.
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
        return np.asarray(self.score_xy(X, Y), dtype=np.float64)

    # ── read-only public accessor for the frozen class-label order ───
    def class_order_fn(self) -> tuple:
        """Read-only tuple of class labels in the order ``predict``
        returns them. Set at ``fit``; DOES NOT CHANGE until the next
        ``fit`` call. External code needing the label order MUST call
        this — direct access to ``rep._labels`` is discouraged.
        """
        self._require_fit()
        if self._labels is None:
            raise RuntimeError(f"{type(self).__name__}: no labels — unsupervised fit")
        return tuple(self._labels.tolist())

    # ── aggregator-over-score_xy centroid (medoid under the rep kernel) ──
    # Delegates to module-level helpers in ``aggregator_utils.py``. The
    # instance-method form is kept so subclasses can call
    # ``self._aggregator_reducer_fn(agg)`` and
    # ``self._centroid_under_score(X, aggregator=...)`` exactly as before.
    def _aggregator_reducer_fn(self, agg):
        """Thin wrapper — see ``aggregator_utils.aggregator_reducer_fn``."""
        return aggregator_reducer_fn(agg)

    def _centroid_under_score(
        self,
        X: np.ndarray,
        aggregator: str | callable,
    ) -> np.ndarray:
        """Thin wrapper — see ``aggregator_utils.centroid_under_score``."""
        return centroid_under_score(self, X, aggregator)

    # ── display / normalization ─────────────────────────────────────
    # Global blend weight: norm = α · ECDF + (1-α) · minmax-distance.
    # α=1.0 → pure ECDF (legacy, gaps collapse).
    # α=0.0 → pure min-max in distance domain (raw shape, just rescaled).
    # 0<α<1 → ECDF ordering preserved AND density gaps stay visible.
    # Monotone in raw ⇒ per-class ROC unchanged. Pair·normalized may
    # shift slightly because it uses normalized magnitudes in a dot
    # product before the final rank step.
    NORM_ALPHA: float = 0.5

    def normalize_score(self, scores: np.ndarray, alpha: float | None = None) -> np.ndarray:
        """Map scores to ``[0, 1]`` using the FIT-time reference grid.

        Blend formula: ``alpha · ECDF + (1-alpha) · minmax`` against the
        same fit-time reference distances. Default ``alpha`` comes from
        the class attribute ``NORM_ALPHA`` so a single edit changes the
        domain everywhere (dist plot, slider readout, ROC x-axis).

        ``self._norm_ref`` (sorted training raw scores, populated by
        ``_mark_fitted(X_tr)``) is the parameter that comes from fit.
        Fallback when ``_norm_ref`` is missing: blend computed on the
        input itself. Shape of input is preserved.
        """
        s = np.asarray(scores, dtype=np.float64)
        a = float(self.NORM_ALPHA if alpha is None else alpha)
        d = self.score_to_distance_fn(s)
        ref = getattr(self, "_norm_ref", None)
        if ref is not None and ref.size > 0:
            # ECDF component (closeness): farther/N as before.
            idx_le = np.searchsorted(ref, d, side="right")
            ecdf = np.clip((ref.size - idx_le) / ref.size, 0.0, 1.0)
            # Min-max component (gap-preserving): map distance into
            # [0,1] using ROBUST percentiles of the ref distribution.
            # Pure ref[0]..ref[-1] gets crushed by outliers (one
            # near-orthogonal training pair → ref_hi ≫ bulk → bulk all
            # collapses near mm≈1, blend looks like ECDF). 2..98% bracket
            # keeps the bulk-spread visible. Clip into the bracket so
            # tail values stick to 0/1.
            ref_lo, ref_hi = (float(np.percentile(ref, 2)),
                              float(np.percentile(ref, 98)))
            if ref_hi > ref_lo:
                d_clip = np.clip(d, ref_lo, ref_hi)
                mm = 1.0 - (d_clip - ref_lo) / (ref_hi - ref_lo)
            else:
                mm = np.zeros_like(d)
            return np.clip(a * ecdf + (1.0 - a) * mm, 0.0, 1.0)
        # Fallback — same blend on the input batch.
        from scipy.stats import rankdata
        n = s.size
        if n <= 1:
            return np.zeros_like(s)
        d_flat = d.reshape(-1)
        rank_dist = rankdata(d_flat, method="average")
        ecdf = (n - rank_dist) / max(n - 1, 1)
        d_lo, d_hi = float(d_flat.min()), float(d_flat.max())
        if d_hi > d_lo:
            mm = 1.0 - (d_flat - d_lo) / (d_hi - d_lo)
        else:
            mm = np.zeros_like(d_flat)
        out = a * ecdf + (1.0 - a) * mm
        return np.clip(out.reshape(s.shape), 0.0, 1.0)

    # ── pair-level scores (canonical pipelines) ─────────────────────
    def cosine_pairs(self, X: np.ndarray) -> np.ndarray:
        """Upper-triangular embedded-pair primitive on ``X``.

        ``score_pairs(skip_kernel=True)`` returns ``Xn @ Xnᵀ`` directly
        (kernel bypassed). When ``transform`` L2-normalizes (the base
        default), this equals plain cosine — hence the legacy method
        name. With a non-normalizing ``transform`` it equals the
        inner product on whatever embedding the subclass produces.
        Output passes through ``self.normalize_score`` so the result is
        already in the ``[0, 1]`` ROC domain. Returns the triu(1) vector
        aligned with ``np.triu_indices(len(X), k=1)``.
        """
        self._require_fit()
        _, _, pc_raw = self.score_pairs(X, only_upper_triangle=True,
                                        skip_kernel=True)
        return self.normalize_score(np.asarray(pc_raw, dtype=np.float64))

    # ── embedding ───────────────────────────────────────────────────
    def transform(self, X: np.ndarray) -> np.ndarray:
        """L2-normalize onto the unit sphere (default hyperspherical embedding)."""
        return l2_normalize(
            np.asarray(X, dtype=np.float64),
            eps=getattr(self, "eps", 1e-12),
        )

    # ── training ────────────────────────────────────────────────────
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        hints: dict | None = None,
        aggregator: str | callable | None = None,
    ) -> "Representation":
        """Fit the method. Must set ``self._labels``. Returns ``self``.

        ``aggregator`` is the SOLE entry point for the centroid
        aggregator. Subclasses with ``_uses_aggregator = True`` MUST
        raise ``ValueError`` when ``aggregator is None``; they assign
        ``self._aggregator`` once at the top of fit() and never again.
        Subclasses that don't build score-medoid prototypes accept the
        kwarg for interface uniformity and ignore it.
        """

    # ── public read-only accessor for the fit-time aggregator ───────
    def get_aggregator(self) -> str | callable | None:
        """Return the aggregator string (or callable) frozen at fit-time.

        ``None`` for representations that don't consume one (KNN, SVM,
        SRC, LabelProp, plain cosine, ArcCos kernel, Bingham, CosFace).
        There is NO setter — only ``fit(..., aggregator=...)`` writes
        this value.
        """
        return self._aggregator

    # ── RAW per-class scores — the sole subclass scoring method ────
    @abstractmethod
    def _score_per_class_raw(self, X: np.ndarray) -> np.ndarray:
        """Return ``(N, K)`` raw per-class scores aligned to ``self._labels``.

        "Higher = more likely class c". Free-form: cosine, kernel vote,
        negative residual, svm decision, or an already-normalized posterior
        (in which case the subclass sets ``_raw_is_posterior = True``).
        """

    # ── posterior derivation (uniform across reps) ─────────────────
    def _posterior_matrix(self, X: np.ndarray) -> np.ndarray:
        """``(N, K)`` posterior P(c|x). Always a simplex per row.

        Stabilization subtracts the row's CLOSEST score before exp(),
        routed through ``closest_score_fn`` so a polarity flip in
        ``convention.CLOSEST_IS_MAX`` rewires this code with no edits.
        """
        raw = np.asarray(self._score_per_class_raw(X), dtype=np.float64)
        if self._raw_is_posterior:
            P = np.maximum(raw, 0.0)
        else:
            row_closest = self.closest_score_fn(raw, axis=1)[:, None]
            r = raw - row_closest
            P = np.exp(r)
        P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)
        return P

    def predict(
        self,
        X: np.ndarray,
        thresholds: dict | None = None,
        normalized_thresholds: bool = False,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict class label, score, and full per-class vote matrix.

        Returns ``(cls_arr, score_arr, votes_arr)`` where each is a
        length-N container (single-vector input → length-1 arrays).

          • ``cls_arr``   : object array, shape ``(N,)``. Predicted
            class label per sample, or ``None`` if thresholds were
            provided AND the sample failed every per-class threshold.
          • ``score_arr`` : object array, shape ``(N,)``. Raw winning
            score for ``cls_arr[i]``, or ``None`` when ``cls_arr[i]``
            is ``None``.
          • ``votes_arr`` : float array, shape ``(N, K)``. RAW per-class
            scores aligned to ``self.class_order_fn()`` — populated for
            every sample regardless of threshold rejection. Callers that
            need [0, 1] apply ``normalize_score`` per column.

        ``thresholds`` semantics:
          • ``None``, ``{}``, or no float values ⇒ ``arg_closest_score_fn``
            over every sample, no rejection possible (cls_arr fully filled).
          • ``{class_label: float}`` ⇒ class ``c`` passes iff comparison
            succeeds in the chosen domain (raw or normalized). Samples
            failing every threshold get ``cls=None, score=None`` while
            ``votes`` keeps the full raw row.

        ``normalized_thresholds`` (default ``False``):
          • ``False`` — RAW thresholds via ``threshold_compare_fn``.
          • ``True``  — NORMALIZED ∈ [0,1] via ``>=`` after per-column
            ``normalize_score(..., alpha)``.

        Winner is always picked via ``arg_closest_score_fn`` on raw
        values — preserves polarity-aware tie-breaking regardless of
        which domain was used to gate.
        """
        self._require_fit()
        X = np.asarray(X)
        is_single = (X.ndim == 1)
        if is_single:
            X = X[None, :]
        labels    = np.asarray(self.class_order_fn(), dtype=self._labels.dtype)
        values    = np.asarray(self._score_per_class_raw(X), dtype=np.float64)
        farthest  = self.farthest_score_sentinel_fn()

        if normalized_thresholds:
            cmp_values = np.column_stack([
                self.normalize_score(values[:, k], alpha=alpha)
                for k in range(values.shape[1])
            ])
            missing_sentinel = -1.0
            cmp_fn = lambda v, t: v >= t
        else:
            cmp_values = values
            missing_sentinel = farthest
            cmp_fn = self.threshold_compare_fn

        thr = np.full(len(labels), missing_sentinel, dtype=np.float64)
        any_set = False
        if isinstance(thresholds, dict):
            for i, c in enumerate(labels):
                v = thresholds.get(c)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    thr[i] = float(v)
                    any_set = True

        N = values.shape[0]
        cls_arr   = np.full(N, None, dtype=object)
        score_arr = np.full(N, None, dtype=object)

        if any_set:
            passed = cmp_fn(cmp_values, thr[None, :])                   # (N, K)
            masked = np.where(passed, values, farthest)
            chosen = self.arg_closest_score_fn(masked, axis=1)
            has_any = passed.any(axis=1)
            for i in np.flatnonzero(has_any):
                ki = int(chosen[i])
                cls_arr[i]   = labels[ki]
                score_arr[i] = float(values[i, ki])
        else:
            chosen = self.arg_closest_score_fn(values, axis=1)
            for i in range(N):
                ki = int(chosen[i])
                cls_arr[i]   = labels[ki]
                score_arr[i] = float(values[i, ki])

        return cls_arr, score_arr, values

    # ── diagnostic gate — a single call to sanity-check a fitted rep ──
    def self_consistency_check_fn(
        self, X_probe: np.ndarray | None = None, n_probe: int = 32,
    ) -> dict:
        """Thin wrapper — see ``diagnostics.self_consistency_check_fn``."""
        from .diagnostics import self_consistency_check_fn as _impl
        return _impl(self, X_probe=X_probe, n_probe=n_probe)

    # ── pairwise ────────────────────────────────────────────────────
    def _pair_kernel(self, M_pair: np.ndarray, Xn: np.ndarray) -> np.ndarray:
        """Transform the cosine pair-matrix into the rep's native kernel.

        Default: identity (plain cosine). Override in reps that wrap
        cosine in something non-linear — e.g. ``arc_cosine_kernel`` for
        the ArcCos family, ``bingham_geodesic_kernel`` for Bingham.

        ``Xn`` is the already-L2-normed batch, passed through in case a
        subclass needs it (none currently do).
        """
        return M_pair

    def score_pairs(
        self,
        X: np.ndarray,
        only_upper_triangle: bool = False,
        skip_kernel: bool = False,
        cos_similarity: bool | None = None,         # deprecated alias
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pairwise score matrix between rows of ``X``.

        ``skip_kernel=False`` (default) → return the rep's native pair
                                   score via ``_pair_kernel(M_pair, Xn)``.
                                   Identity for cosine reps; arc-cosine
                                   for the ArcCos family; Bingham-
                                   geodesic for Bingham; etc.
        ``skip_kernel=True``  → return the embedded-pair primitive
                                   ``Xn @ Xnᵀ`` directly, bypassing
                                   ``_pair_kernel``. Equals cosine ONLY
                                   when ``transform`` L2-normalizes
                                   (the base default); otherwise it is
                                   the inner product on whatever
                                   embedding the subclass-defined
                                   ``transform`` produces.

        ``transform(X)`` is applied exactly once inside this method — the
        subclass hook receives the already-computed embedded-pair matrix,
        never raw X.

        ``cos_similarity`` is a deprecated alias kept for back-compat;
        it maps directly to ``skip_kernel``.
        """
        self._require_fit()
        from .math_utils import embedded_pair_matrix
        if cos_similarity is not None:
            skip_kernel = bool(cos_similarity)
        Xn = self.transform(X)                              # single transform
        M_pair = embedded_pair_matrix(Xn)
        M = M_pair if skip_kernel else self._pair_kernel(M_pair, Xn)
        return pair_scores_return(M, only_upper_triangle)

    # ── optional geometric prototypes ──────────────────────────────
    def get_centroids(self) -> dict[object, np.ndarray]:
        """Per-class on-sphere prototypes, or ``{}`` if none."""
        return {}


__all__ = ["Representation"]
