"""k-NN label propagation from labeled anchor sessions.

Given a set of anchor sessions with known labels and a full set of
embedding vectors, propagate labels to unlabeled sessions using
majority vote among k nearest neighbors (cosine similarity).

This implements the Atlas/TensorBoard-style semi-automated labeling
workflow: the user labels 2-3 anchor sessions, and the dashboard
auto-extends those labels via k-NN in vector space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PropagationResult:
    """Output of k-NN label propagation.

    `propagated_labels`: Series aligned to session_ids with propagated labels.
    `confidence`       : Series of float [0, 1] — fraction of k neighbors
                         that agreed on the assigned label.
    `is_anchor`        : Series of bool — True for anchor (original) labels.
    """

    propagated_labels: pd.Series
    confidence: pd.Series
    is_anchor: pd.Series


def propagate_labels(
    vectors: np.ndarray,
    session_ids: list[str],
    anchor_labels: dict[str, str],
    k: int = 5,
) -> PropagationResult:
    """Propagate labels from anchor sessions to all sessions via k-NN.

    `anchor_labels` maps session_id -> label for the known anchors.
    Unlabeled sessions get the majority label among their k nearest
    neighbors (by cosine similarity). Anchor sessions keep their
    original label with confidence=1.0.
    """
    from sklearn.neighbors import NearestNeighbors

    n = vectors.shape[0]
    if n == 0:
        empty = pd.Series(dtype=object)
        return PropagationResult(empty, pd.Series(dtype=float), pd.Series(dtype=bool))

    # L2-normalize for cosine similarity via euclidean on unit sphere
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.maximum(norms, 1e-12)

    # Build index of anchor sessions
    sid_to_idx = {s: i for i, s in enumerate(session_ids)}
    anchor_indices = []
    anchor_label_list = []
    for sid, lbl in anchor_labels.items():
        idx = sid_to_idx.get(sid)
        if idx is not None and lbl:
            anchor_indices.append(idx)
            anchor_label_list.append(lbl)

    if not anchor_indices:
        return PropagationResult(
            pd.Series([""] * n, dtype=object),
            pd.Series([0.0] * n),
            pd.Series([False] * n),
        )

    # Fit k-NN on all points, query neighbors for each point
    k_eff = min(k, n)
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", algorithm="auto")
    nn.fit(normed)
    distances, indices = nn.kneighbors(normed)

    # For each point, look at its k neighbors — if any are anchors,
    # count votes and assign majority label
    anchor_set = set(anchor_indices)
    anchor_idx_to_label = dict(zip(anchor_indices, anchor_label_list))

    labels_out = [""] * n
    confidence_out = [0.0] * n
    is_anchor = [False] * n

    for i in range(n):
        sid = session_ids[i]
        if sid in anchor_labels:
            labels_out[i] = anchor_labels[sid]
            confidence_out[i] = 1.0
            is_anchor[i] = True
            continue

        # Count votes from anchor neighbors
        votes: dict[str, int] = {}
        for j_idx in indices[i]:
            if j_idx in anchor_set:
                lbl = anchor_idx_to_label[j_idx]
                votes[lbl] = votes.get(lbl, 0) + 1

        if votes:
            best_label = max(votes, key=votes.get)  # type: ignore[arg-type]
            labels_out[i] = best_label
            confidence_out[i] = votes[best_label] / k_eff
        # else: no anchor neighbors found, label stays ""

    return PropagationResult(
        propagated_labels=pd.Series(labels_out, dtype=object),
        confidence=pd.Series(confidence_out, dtype=float),
        is_anchor=pd.Series(is_anchor, dtype=bool),
    )
