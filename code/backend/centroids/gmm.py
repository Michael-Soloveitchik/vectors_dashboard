"""GMM-based centroid extraction per label.

Spec point 42 — find_centroids button. The user picks `n_per_label`
GMM components per label and we fit them on the *raw high-dim*
vectors so the centroids live in the same space as the original
embeddings (and not in the post-reduction 2D coordinates, which would
make their semantics depend on which projection method is active).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..common.data_access import SESSION_COL, VECTOR_COL, get_vectors_for_sessions


@dataclass
class CentroidSet:
    """Per-label GMM centroids in raw vector space.

    `centroids[label]`           : array of shape (k_label, D) of GMM means.
    `covariances[label]`         : array of shape (k_label, D, D) (full type).
    `weights[label]`             : array of shape (k_label,).
    `member_session_ids[label]`  : ALL session_ids belonging to this label.
    `eval_session_ids[label]`    : held-out session_ids NOT used for fitting
                                   (used as ROC ground truth positives).
    """

    centroids: dict[str, np.ndarray]
    covariances: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    member_session_ids: dict[str, list[str]]
    n_per_label: dict[str, int]
    eval_session_ids: dict[str, list[str]] | None = None

    def labels(self) -> list[str]:
        return sorted(self.centroids.keys())

    def total(self) -> int:
        return sum(c.shape[0] for c in self.centroids.values())


def fit_centroids(
    df: pd.DataFrame,
    label_series: pd.Series,
    n_per_label: dict[str, int] | int = 1,
    covariance_type: str = "full",
    random_state: int = 42,
    progress_callback=None,
    max_seconds: float | None = None,
    holdout_fraction: float = 0.5,
) -> CentroidSet:
    """Fit one GMM per unique label value in `label_series`.

    `n_per_label` may be an int (same K for all labels) or a dict
    (label -> K). Empty labels and labels with too few samples to fit
    a GMM (n_samples <= K) are silently dropped.

    `max_seconds` caps total wall-clock time. When the budget runs out,
    remaining labels are skipped and only the already-fitted centroids
    are returned. The per-label EM iteration count is also scaled down
    proportionally so each label's fit respects the remaining budget.

    `holdout_fraction` (0.0–1.0) controls the train/eval split per
    label. Centroids are fit on the training portion only; the held-out
    sessions go into ``eval_session_ids`` for unbiased ROC evaluation.
    Labels with < 4 sessions skip the split (all used for fitting).
    """
    import time
    from sklearn.mixture import GaussianMixture

    centroids: dict[str, np.ndarray] = {}
    covariances: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    members: dict[str, list[str]] = {}
    eval_members: dict[str, list[str]] = {}
    n_used: dict[str, int] = {}
    rng = np.random.RandomState(random_state)

    t0 = time.perf_counter()
    groups = list(df.groupby(label_series, sort=True))
    n_labels = sum(1 for lbl, _ in groups if lbl and lbl != "")

    for label_idx, (label, group) in enumerate(groups):
        if label == "" or label is None:
            continue
        # Time budget: stop fitting if wall-clock limit exceeded
        if max_seconds is not None:
            elapsed = time.perf_counter() - t0
            if elapsed >= max_seconds:
                break
        # Extract directly from the group — avoids rebuilding the full
        # session-id index and per-row iloc lookups on every label.
        raw_vecs = group[VECTOR_COL].to_numpy()
        raw_sids = group[SESSION_COL].to_numpy()
        rows = []
        used = []
        for sid, v in zip(raw_sids, raw_vecs):
            arr = np.asarray(v).reshape(-1)
            if arr.size == 0:
                continue
            rows.append(arr)
            used.append(str(sid))
        if len(rows) < 2:
            continue
        all_vecs = np.stack(rows, axis=0).astype(np.float32, copy=False)
        all_sids = list(used)
        if all_vecs.shape[0] < 2:
            continue
        # Train/eval split for unbiased ROC evaluation.
        # Even labels with 2–3 sessions get a leave-one-out holdout
        # so every matched label produces a valid ROC curve.
        # (Previously required n>=4, causing AUC=nan for small labels.)
        n_total = len(all_sids)
        if holdout_fraction > 0 and n_total >= 2:
            n_eval = max(1, int(n_total * holdout_fraction))
            n_fit = n_total - n_eval
            # GMM k is clamped to n_fit later, so n_fit=1 works for
            # single-component fits (mean = the one sample, covariance
            # = reg_covar * I).  Always keep at least 1 eval sample.
            if n_fit < 1:
                n_fit = 1
                n_eval = n_total - n_fit
            perm = rng.permutation(n_total)
            fit_idx = perm[:n_fit]
            eval_idx = perm[n_fit:]
            vecs = all_vecs[fit_idx]
            used = [all_sids[i] for i in fit_idx]
            eval_sids = [all_sids[i] for i in eval_idx]
        else:
            vecs = all_vecs
            used = all_sids
            eval_sids = []
        k_target = (
            int(n_per_label.get(label, 1)) if isinstance(n_per_label, dict)
            else int(n_per_label)
        )
        k = max(1, min(k_target, vecs.shape[0]))
        # Scale max_iter by remaining time budget and data dimension.
        # Full-covariance GMM on D=1024 is O(D^2) per iteration; cap
        # iterations so each label fits in its share of the budget.
        dim = vecs.shape[1] if vecs.ndim == 2 else 1
        base_max_iter = 30 if dim > 256 else 100
        max_iter = base_max_iter
        if max_seconds is not None:
            remaining = max_seconds - (time.perf_counter() - t0)
            labels_left = max(n_labels - label_idx, 1)
            per_label_budget = remaining / labels_left
            # Scale: ~10 iters per second of budget for high-dim data
            iter_budget = max(5, int(per_label_budget * 10)) if dim > 256 \
                else max(5, int(per_label_budget * 50))
            max_iter = min(base_max_iter, iter_budget)
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
            reg_covar=1e-4,
            max_iter=max_iter,
        )
        gmm.fit(vecs)
        centroids[str(label)] = gmm.means_.astype(np.float32)
        covariances[str(label)] = gmm.covariances_.astype(np.float32)
        weights[str(label)] = gmm.weights_.astype(np.float32)
        members[str(label)] = all_sids  # ALL sessions for this label
        eval_members[str(label)] = eval_sids  # held-out only
        n_used[str(label)] = k
        if progress_callback is not None:
            progress_callback(str(label), label_idx)
    return CentroidSet(
        centroids=centroids,
        covariances=covariances,
        weights=weights,
        member_session_ids=members,
        n_per_label=n_used,
        eval_session_ids=eval_members,
    )
