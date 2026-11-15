"""Empirical decision-boundary contour for PCA-projected centroids.

Problem
-------
The current "dashed threshold circle" on the left scatter is a projected
sphere radius — correct only if all data lies on the PCA 2-plane and
every direction in high-D maps linearly to a fixed radius in 2D. For
general data it's an approximation; for t-SNE / UMAP it's meaningless.

Approach (PCA-only)
-------------------
1. Build a dense 2D grid covering the projected data range.
2. Lift the grid back to high-D through the PCA inverse: every 2D point
   becomes a vector on the PCA 2-plane in the original feature space.
3. For each centroid, compute cosine similarity on the lifted grid.
4. Extract the contour at `similarity == threshold` via ``contourpy``.
5. Return the contour xy-paths ready to drop into a Plotly ``scatter``.

t-SNE / UMAP have no inverse, so this is **PCA-only**. Caller gates on
``method == 'pca'`` and hides the dashed boundary otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContourPath:
    """One closed/open contour segment. NaN separates sub-segments."""
    label: str
    threshold: float
    xs: np.ndarray
    ys: np.ndarray


def build_projected_grid(
    points_2d: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    margin: float = 0.15,
    n_side: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid the 2D projected space and lift back to high-D via PCA.

    Args:
        points_2d: (N, 2) projected data, used to size the grid.
        pca_components: (2, D) PCA components (sklearn ``components_``).
        pca_mean: (D,) per-feature mean (sklearn ``mean_``).
        margin: fractional padding around the data bounding box.
        n_side: mesh resolution per axis.

    Returns:
        xs, ys, grid_hd — 1D x array (n_side,), 1D y array (n_side,),
        and (n_side*n_side, D) lifted high-dim grid.
    """
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)
    pad_x = (x_max - x_min) * margin + 1e-6
    pad_y = (y_max - y_min) * margin + 1e-6
    xs = np.linspace(x_min - pad_x, x_max + pad_x, n_side)
    ys = np.linspace(y_min - pad_y, y_max + pad_y, n_side)
    xx, yy = np.meshgrid(xs, ys)
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_hd = grid_2d @ pca_components + pca_mean
    return xs, ys, grid_hd


def _cosine_field(grid_hd: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine similarity from each grid point to centroid.

    L2-normalizes both sides. Accepts centroid as 1D (D,) or 2D (k, D);
    for k>1 the per-grid-point similarity is the max over centroids
    (matches the "any centroid accepts" semantics used in ROC scoring).
    Returns field reshaped to (n_side, n_side).
    """
    g_norms = np.linalg.norm(grid_hd, axis=1, keepdims=True)
    g_norms = np.where(g_norms < 1e-12, 1.0, g_norms)
    grid_unit = grid_hd / g_norms
    c = np.atleast_2d(centroid)
    c_norms = np.linalg.norm(c, axis=1, keepdims=True)
    c_norms = np.where(c_norms < 1e-12, 1.0, c_norms)
    c_unit = c / c_norms
    from ..representations.base import closest_score_fn
    sim = closest_score_fn(grid_unit @ c_unit.T, axis=1)
    n_side = int(np.sqrt(sim.size))
    return sim.reshape(n_side, n_side)


def extract_contour(
    xs: np.ndarray, ys: np.ndarray,
    field: np.ndarray, level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract iso-contour at `level` from a 2D scalar field.

    Falls back to matplotlib contour (always available) since
    ``contourpy`` may not be installed.
    Returns NaN-separated xy-arrays ready for a Plotly ``scatter`` trace.
    """
    try:
        from contourpy import contour_generator
        cg = contour_generator(x=xs, y=ys, z=field)
        lines = cg.lines(level)
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cs = ax.contour(xs, ys, field, levels=[level])
        lines = []
        for seg in cs.allsegs[0]:
            if len(seg):
                lines.append(seg)
        plt.close(fig)
    if not lines:
        return np.array([]), np.array([])
    parts_x, parts_y = [], []
    for seg in lines:
        seg = np.asarray(seg)
        if seg.ndim != 2 or seg.shape[0] < 2:
            continue
        parts_x.extend(seg[:, 0].tolist())
        parts_x.append(np.nan)
        parts_y.extend(seg[:, 1].tolist())
        parts_y.append(np.nan)
    return np.asarray(parts_x), np.asarray(parts_y)


def build_decision_boundary(
    label: str,
    centroid_hd: np.ndarray,
    threshold: float,
    points_2d: np.ndarray,
    pca_components: np.ndarray,
    pca_mean: np.ndarray,
    n_side: int = 160,
    margin: float = 0.15,
) -> ContourPath:
    """High-level entry — returns a ContourPath ready to render."""
    xs, ys, grid_hd = build_projected_grid(
        points_2d, pca_components, pca_mean,
        margin=margin, n_side=n_side,
    )
    field = _cosine_field(grid_hd, centroid_hd)
    cxs, cys = extract_contour(xs, ys, field, threshold)
    return ContourPath(label=label, threshold=float(threshold), xs=cxs, ys=cys)
