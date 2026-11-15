"""Per-label ROC curve over cosine-similarity scores.

For each label, score every dataset point by the *maximum cosine
similarity* to any of that label's GMM centroids. Higher score =
more likely to belong to the label.

Ground truth: ``member_session_ids`` — all sessions belonging to the
label.  Centroids are fit on 50% (holdout split) so half the positive
scores are unbiased.  Only aggregated (selected) points are scored.

We render the ROC in log-FPR space because the interesting region for
verification systems is the very-low-FPR end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..centroids.gmm import CentroidSet


@dataclass
class LabelScores:
    """Per-label score vector aligned to a session_id list."""

    label: str
    session_ids: list[str]
    scores: np.ndarray        # cosine similarity, higher = more similar
    is_member: np.ndarray     # bool, True iff session_id belongs to `label`


@dataclass
class ROCCurve:
    """log-FPR ready ROC curve."""

    label: str
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    score_min: float = 0.0   # actual min cosine similarity in the data
    score_max: float = 1.0   # actual max cosine similarity in the data


def _cosine_score_against_centroids(
    vectors: np.ndarray,
    means: np.ndarray,
) -> np.ndarray:
    """Max cosine similarity from each vector to any centroid.

    Returns an (N,) array of scores in [-1, 1].
    """
    # L2-normalize vectors: (N, D)
    v_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    v_normed = vectors / np.maximum(v_norms, 1e-12)
    # L2-normalize centroids: (K, D)
    c_norms = np.linalg.norm(means, axis=1, keepdims=True)
    c_normed = means / np.maximum(c_norms, 1e-12)
    # Cosine similarity matrix: (N, K)
    sims = (v_normed @ c_normed.T).astype(np.float64)
    # Closest-centroid score for each vector — convention-routed.
    from ..representations.base import closest_score_fn
    return closest_score_fn(sims, axis=1)


def label_cosine_scores(
    centroid_set: CentroidSet,
    session_ids: list[str],
    vectors: np.ndarray,
    include_training: bool = False,
    agg_set: set[str] | None = None,
) -> dict[str, LabelScores]:
    """One LabelScores per label.

    Two modes, selected by ``include_training``:

    * ``False`` (default, unbiased): training sessions are excluded
      from the scoring pool and ``eval_session_ids`` are the positives.
    * ``True``: every point in ``session_ids`` is scored (including
      training sessions) and **all members** of the label are marked as
      positive ground truth.

    **`agg_set`** (population gate): when non-empty, restricts the
    entire score population to ``session_ids ∩ agg_set``. Positives
    are ``selected ∩ members(label)``; negatives are
    ``selected − members(label)``. All ROC quantities (TP, FP, AUC)
    are therefore relative to the selection, not to the full dataset.
    This matches the user spec:
      - single-class selection → ROC is a 90° leg (zero negatives)
      - two-class selection    → ROC reflects the real confusion
    Each label with ≥1 selected session still gets a LabelScores
    entry, even if positives or negatives are zero — so the legend
    on ROC / histogram lists every selected label.
    When ``agg_set`` is None or empty, full-membership semantics apply.
    """
    eval_ids = centroid_set.eval_session_ids or {}
    all_members = centroid_set.member_session_ids or {}
    out: dict[str, LabelScores] = {}
    for label in centroid_set.labels():
        means = centroid_set.centroids[label]
        scores = _cosine_score_against_centroids(vectors, means)
        members_set = set(all_members.get(label, []))
        gt_ids = eval_ids.get(label, [])
        if include_training:
            gt_set = members_set
            train_set: set[str] = set()
        else:
            gt_set = set(gt_ids)
            train_set = members_set - gt_set
        if agg_set:
            # Population gate: keep only sessions the user selected
            # (still excluding training sessions so the ROC does not
            # double-count the centroid's own fit data).
            keep = np.array(
                [s in agg_set and s not in train_set for s in session_ids],
                dtype=bool,
            )
        else:
            keep = np.array([s not in train_set for s in session_ids], dtype=bool)
        filtered_sids = [s for s, k in zip(session_ids, keep) if k]
        filtered_scores = scores[keep]
        # Positives in the selected population = selected ∩ members.
        # Negatives = selected − members. No ground-truth intersection
        # step — agg_set drove the `keep` mask above.
        is_member = np.array([s in members_set for s in filtered_sids],
                             dtype=bool)
        n_pos = int(is_member.sum())
        n_neg = len(is_member) - n_pos
        n_train_excl = int((~keep).sum())
        agg_note = f" | agg_set_size={len(agg_set)}" if agg_set else ""
        print(f"[roc-gt] {label}: include_training={include_training}, "
              f"eval_ids={len(gt_ids)}, "
              f"train_excluded={n_train_excl}, "
              f"pos_in_scored={n_pos}, neg_in_scored={n_neg}, "
              f"total_scored={len(filtered_sids)}{agg_note}", flush=True)
        out[label] = LabelScores(label, filtered_sids, filtered_scores, is_member)
    return out


def filter_to_window(
    sc: LabelScores,
    low: float,
    high: float,
) -> LabelScores:
    """Keep only samples whose score lies in ``[low, high]``.

    Use-case: per-label threshold window. The ROC computed on the
    filtered LabelScores is the "AUC given score ∈ window" — it
    measures how separable the data is *inside* the chosen band.
    At the global window (full score range) this reduces to the
    unfiltered ROC. Narrowing the window drops easy samples at the
    tails and reveals the overlap region.
    """
    # Polarity-aware band filter: keep samples whose score is at-least-
    # as-close as ``low`` AND at-most-as-close as ``high``. Under
    # CLOSEST_IS_MAX=True this == (s >= low) & (s <= high). Under
    # flipped polarity, threshold_compare_fn auto-inverts.
    from backend.representations.convention import (
        threshold_compare_fn, a_closer_than_b_fn,
    )
    not_closer_than_high = ~a_closer_than_b_fn(sc.scores, high)
    mask = threshold_compare_fn(sc.scores, low) & not_closer_than_high
    return LabelScores(
        label=sc.label,
        session_ids=[s for s, k in zip(sc.session_ids, mask) if bool(k)],
        scores=sc.scores[mask],
        is_member=sc.is_member[mask],
    )


def roc_curve_cosine(scores: LabelScores) -> ROCCurve:
    """ROC curve from cosine-similarity scores (higher = positive).

    Uses sklearn's roc_curve with ``drop_intermediate=False`` so EVERY
    scored data point produces exactly one threshold step on the ROC.
    Number of ROC points = number of scored samples + 1.
    Thresholds are actual cosine-similarity values from the data.
    """
    from sklearn.metrics import auc, roc_curve

    if scores.scores.size == 0:
        return ROCCurve(scores.label, np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                        np.array([1.0, -1.0]), auc=float("nan"),
                        score_min=0.0, score_max=1.0)
    s_min = float(scores.scores.min())
    s_max = float(scores.scores.max())

    n_pos = int(scores.is_member.sum())
    n_neg = len(scores.is_member) - n_pos
    if n_pos == 0:
        # No positives — degenerate: horizontal line at TPR=0
        return ROCCurve(scores.label,
                        np.array([0.0, 1.0]), np.array([0.0, 0.0]),
                        np.array([s_max + 0.01, s_min - 0.01]),
                        auc=0.0, score_min=s_min, score_max=s_max)
    if n_neg == 0:
        # All members, no negatives — 90-degree leg: FPR stays at 0,
        # TPR jumps from 0 to 1. AUC = 1.0 (perfect separation).
        return ROCCurve(scores.label,
                        np.array([0.0, 0.0, 1.0]),
                        np.array([0.0, 1.0, 1.0]),
                        np.array([s_max + 0.01, s_min - 0.01, s_min - 0.02]),
                        auc=1.0, score_min=s_min, score_max=s_max)

    # drop_intermediate=False: one threshold per scored sample
    fpr, tpr, thr = roc_curve(scores.is_member.astype(int), scores.scores,
                              drop_intermediate=False)
    # sklearn prepends +inf as sentinel; clamp to score_max + small margin
    thr = np.clip(thr, s_min - 0.01, s_max + 0.01)

    computed_auc = float(auc(fpr, tpr))
    print(f"[roc-compute] {scores.label}: scored={len(scores.scores)}, "
          f"ROC_points={len(fpr)}, AUC={computed_auc:.4f}, "
          f"score_range=[{s_min:.4f},{s_max:.4f}]", flush=True)

    return ROCCurve(
        label=scores.label,
        fpr=fpr.astype(np.float64),
        tpr=tpr.astype(np.float64),
        thresholds=thr.astype(np.float64),
        auc=computed_auc,
        score_min=s_min,
        score_max=s_max,
    )


@dataclass
class DETCurve:
    """DET curve — normal-deviate-warped FPR and FNR axes."""

    label: str
    x_ppf: np.ndarray   # norm.ppf(fpr)  — x axis
    y_ppf: np.ndarray   # norm.ppf(fnr)  — y axis
    fpr: np.ndarray
    fnr: np.ndarray
    thresholds: np.ndarray
    eer: float           # equal-error rate (FPR == FNR crossing)


def det_curve_from_roc(roc: ROCCurve) -> DETCurve:
    """Convert a ROCCurve to DET axes (normal deviate warped).

    DET is the biometrics-standard view: both axes are probit-scaled,
    which linearizes Gaussian-distributed scores and spreads the
    low-error region that matters for speaker verification.
    """
    from scipy.stats import norm

    fnr = 1.0 - roc.tpr
    # Clip to avoid ±inf at 0.0 and 1.0
    eps = 1e-7
    fpr_c = np.clip(roc.fpr, eps, 1.0 - eps)
    fnr_c = np.clip(fnr, eps, 1.0 - eps)
    x_ppf = norm.ppf(fpr_c)
    y_ppf = norm.ppf(fnr_c)

    # EER: where FPR ≈ FNR (the crossing point)
    diff = roc.fpr - fnr
    cross_idx = np.argmin(np.abs(diff))
    eer = float(0.5 * (roc.fpr[cross_idx] + fnr[cross_idx]))

    return DETCurve(
        label=roc.label,
        x_ppf=x_ppf,
        y_ppf=y_ppf,
        fpr=roc.fpr,
        fnr=fnr,
        thresholds=roc.thresholds,
        eer=eer,
    )
