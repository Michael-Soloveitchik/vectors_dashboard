"""2D PCA wrapper.

PCA gets its own file because it's the *fast* path: the front-end
defaults here on first paint, so the import has to be cheap. We
import sklearn lazily inside the function so the dashboard module
graph doesn't pull all of sklearn just to draw configs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def run_pca(vectors: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Project to 2 components.

    sklearn's PCA needs n_samples >= n_components, which we already
    enforce in `reduce_2d` (n_samples < 2 short-circuits earlier).
    """
    from sklearn.decomposition import PCA

    n = vectors.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)
    n_components = min(2, n - 1, vectors.shape[1])
    pca = PCA(n_components=n_components, random_state=params.get("random_state", 42))
    proj = pca.fit_transform(vectors)
    if proj.shape[1] < 2:
        # 1D fallback — pad with zeros so the front-end can plot a line.
        pad = np.zeros((n, 2 - proj.shape[1]), dtype=proj.dtype)
        proj = np.concatenate([proj, pad], axis=1)
    return proj
