"""Mesh-based decision-band projection for the left scatter.

Strategy:
1. Sample ``N`` uniformly-distributed unit vectors in the full-dim
   space. Uniform on the unit sphere means each direction is equally
   likely; cosine similarity against a centroid is the inner product
   after L2-norm, so these samples probe the whole cos-sim space
   uniformly.
2. For each centroid, compute the mesh points' cosine similarity to
   the centroid. This is a pure function of direction.
3. At render time, project the mesh through the *same* 2D reduction
   that produced the scatter. For PCA this is exact (linear). For
   t-SNE / UMAP it is NOT exact — those reductions are non-parametric
   and cannot project new points without re-fitting. We therefore
   only render the decision band when the reduction is PCA.

The band is the set of mesh points with ``cos(v, c_label) ∈ [low, high]``.
Rendering: as a dashed line-scatter (mode="markers", symbol="x-thin",
dashed line via Scatter) or a convex hull polyline. See
``figures/left_scatter_helpers.py`` for the figure-level integration.

The mesh stays in the server-side store (not pickled into dcc.Store)
because (N=2000) × (D=1024 full dim) ≈ 16 MB is too big for the
browser.  Instead we only persist the 2D-projected mesh + per-label
cosine similarities, which is (N * (2 + L)) floats ≈ 100 KB for
typical configs.
"""

from __future__ import annotations

import numpy as np


def sample_unit_sphere(n: int, d: int, seed: int = 0) -> np.ndarray:
    """Return ``(n, d)`` array of unit vectors uniform on S^{d-1}.

    Uses the classic Gaussian trick: normalize iid normal vectors.
    Rows are unit-norm up to floating point. Fixed seed for
    reproducibility.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size=(n, d), dtype=np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return v / norms


def mesh_cosine_to_centroids(
    mesh_full: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Max cosine similarity of each mesh row to any row of centroids.

    ``mesh_full`` : (N, D), each row already unit-norm.
    ``centroids`` : (K, D).
    Returns (N,) — the max cosine against any centroid for the label.
    """
    c_norm = np.linalg.norm(centroids, axis=1, keepdims=True)
    c_norm = np.maximum(c_norm, 1e-12)
    c_normed = centroids / c_norm
    from ..representations.base import closest_score_fn
    sims = mesh_full @ c_normed.T  # (N, K)
    return closest_score_fn(sims, axis=1).astype(np.float32)


def build_mesh_payload(
    centroids_by_label: dict[str, np.ndarray],
    dim: int,
    n_mesh: int = 2000,
    seed: int = 42,
) -> dict:
    """Return a JSON-safe mesh payload ready to ship to the frontend.

    Keys:
      - ``n`` : int, number of mesh samples
      - ``dim`` : int, full-dim size (for sanity checking)
      - ``cos`` : dict ``{label: list[float] of length n}`` — per-label
                  max-cos-sim for every mesh row. The 2D projection is
                  computed at render time using the current PCA fit.
    The mesh itself (N × D) is NOT serialized — the frontend's
    ``_render`` callback recomputes the 2D projection by sampling the
    same mesh from the same seed on demand, or re-fits PCA on the
    vectors and projects a freshly-sampled mesh. See
    ``figures/left_scatter_helpers.py``.
    """
    mesh = sample_unit_sphere(n_mesh, dim, seed=seed)
    cos_by_label: dict[str, list[float]] = {}
    for label, means in centroids_by_label.items():
        means_arr = np.asarray(means, dtype=np.float64)
        if means_arr.ndim == 1:
            means_arr = means_arr.reshape(1, -1)
        if means_arr.shape[1] != dim:
            # Incompatible dims — skip this label rather than crash.
            continue
        cos_by_label[label] = mesh_cosine_to_centroids(mesh, means_arr).tolist()
    return {
        "n": int(n_mesh),
        "dim": int(dim),
        "seed": int(seed),
        "cos": cos_by_label,
    }


def project_mesh_pca(
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    seed: int,
    n_mesh: int,
    dim: int,
) -> np.ndarray:
    """Re-sample the mesh (same seed) and project it through the PCA.

    Returns (N, 2) 2D coordinates.

    ``pca_components`` : (2, D) — PCA transform matrix.
    ``pca_mean`` : (D,) — centering.
    Sampling the same ``seed`` as ``build_mesh_payload`` guarantees
    every mesh row's 2D position corresponds to the same cosine
    similarity value stored in the payload.
    """
    mesh = sample_unit_sphere(n_mesh, dim, seed=seed)
    centered = mesh - pca_mean[None, :]
    return centered @ pca_components.T  # (N, 2)
