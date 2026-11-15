"""2D UMAP wrapper.

`umap-learn` needs n_neighbors < n_samples, so we clamp. Cosine is the
default metric since speaker embeddings are L2-normalized — Euclidean
on the sphere is monotone with cosine but UMAP fits cosine faster on
sparse high-d clouds in our case.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def run_umap(vectors: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    import umap

    n = vectors.shape[0]
    n_neighbors = int(params.get("n_neighbors", 15))
    n_neighbors = max(2, min(n_neighbors, n - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(params.get("min_dist", 0.1)),
        metric=params.get("metric", "cosine"),
        random_state=int(params.get("random_state", 42)),
        verbose=False,
    )
    return reducer.fit_transform(vectors)
