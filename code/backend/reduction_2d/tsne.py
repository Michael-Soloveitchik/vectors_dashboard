"""2D t-SNE wrapper.

The dashboard always passes a perplexity from the controls panel,
so the param defaulting here is just for safety / direct script use.
sklearn raises if perplexity >= n_samples, so we clamp.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def run_tsne(vectors: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = vectors.shape[0]
    perplexity = float(params.get("perplexity", 30))
    perplexity = max(2.0, min(perplexity, max(2.0, (n - 1) / 3.0)))
    # sklearn renamed n_iter -> max_iter in 1.5; pass whichever is supported.
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init=params.get("init", "pca"),
        random_state=int(params.get("random_state", 42)),
        learning_rate="auto",
    )
    try:
        tsne = TSNE(max_iter=int(params.get("n_iter", 750)), **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=int(params.get("n_iter", 750)), **kwargs)
    return tsne.fit_transform(vectors)
