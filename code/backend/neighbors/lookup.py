"""Top-K nearest neighbor lookup by cosine similarity.

Click a point in the scatter → see the K most similar sessions with
their cosine distances. Uses sklearn NearestNeighbors fitted once on
L2-normalized vectors (cosine similarity via euclidean on the unit
sphere).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NearestResult:
    """Result of a nearest-neighbor query for a single point.

    `query_sid`     : session_id of the query point.
    `neighbor_sids` : list of K nearest session_ids (excluding query).
    `distances`     : cosine distances (1 - similarity), shape (K,).
    `similarities`  : cosine similarities, shape (K,).
    """

    query_sid: str
    neighbor_sids: list[str]
    distances: np.ndarray
    similarities: np.ndarray


def find_nearest(
    vectors: np.ndarray,
    session_ids: list[str],
    query_sid: str,
    k: int = 5,
) -> NearestResult | None:
    """Find the K nearest neighbors of `query_sid` by cosine similarity."""
    sid_to_idx = {s: i for i, s in enumerate(session_ids)}
    q_idx = sid_to_idx.get(query_sid)
    if q_idx is None or vectors.shape[0] < 2:
        return None

    # L2-normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.maximum(norms, 1e-12)

    from ..representations.base import (
        closest_first_order, topk_closest_fn, farthest_score_sentinel_fn,
    )
    # Cosine similarity of query vs all
    sims = (normed @ normed[q_idx]).astype(np.float64)
    sims[q_idx] = farthest_score_sentinel_fn()          # exclude self

    # Top-K by similarity — convention-routed.
    k_eff = min(k, len(session_ids) - 1)
    top_indices = topk_closest_fn(sims, k=k_eff, axis=0)
    top_indices = top_indices[closest_first_order(sims[top_indices])]

    neighbor_sids = [session_ids[i] for i in top_indices]
    top_sims = sims[top_indices]
    top_dists = 1.0 - top_sims

    return NearestResult(
        query_sid=query_sid,
        neighbor_sids=neighbor_sids,
        distances=top_dists,
        similarities=top_sims,
    )
