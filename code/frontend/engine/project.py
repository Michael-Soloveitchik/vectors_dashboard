"""2D projection of the unit-sphere vectors + centroid placement + hulls.

The projection depends only on (model, reduction, params, max_points) —
NOT on methodology or thresholds — so it is computed once and reused
while the user flips methodologies or drags sliders.

Centroids: a unit-norm prototype sits ~1/mean-cos further from the PCA
origin than its members (high-dim noise is orthogonal to the PC plane),
which is why the old panel drew centroids far outside their clusters.
PCA is linear, so we project ``c * mean(cos(member, c))`` — the foot of
the members on the centroid direction. For t-SNE / UMAP (no transform)
the marker goes to the 2D mean of the members.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import Analysis, Engine
from .concurrency import freeze
from .data import ModelFrame

MAX_POINTS_RANGE = (200, 50_000)


@dataclass
class Projection:
    rows: np.ndarray            # displayed row indices into the frame
    xy: np.ndarray              # (len(rows), 2)
    method: str                 # "pca" | "tsne" | "umap"
    pca: object | None = None
    offset: np.ndarray | None = None
    explained: tuple[float, float] | None = None

    @property
    def pos(self) -> dict[int, int]:
        return {int(r): i for i, r in enumerate(self.rows)}


def display_rows(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(42).choice(n, max_points,
                                                    replace=False))


def norm_method(method: str | None) -> str:
    return (method or "PCA").strip().lower().replace("-", "")


def project(engine: Engine, frame: ModelFrame, method: str,
            params: dict, max_points: int,
            client: str | None = None) -> Projection:
    """Cached, single-flight per (model, method, params, max_points).
    PCA runs immediately; t-SNE / UMAP take a heavy slot and are dropped
    if the tab has already asked for a different projection."""
    m = norm_method(method)
    if m not in ("pca", "tsne", "umap"):
        raise ValueError(f"unknown reduction {method!r}")
    lo, hi = MAX_POINTS_RANGE
    max_points = int(min(hi, max(lo, int(max_points or 4000))))
    key = (frame.name, m, tuple(sorted(params.items())), max_points)
    engine.gate.want(client, "proj", key)

    def compute() -> Projection:
        rows = display_rows(frame.n, max_points)
        X = frame.Vn[rows]
        if m == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42).fit(X)
            xy = pca.transform(X)
            off = np.median(xy, axis=0, keepdims=True)
            ev = pca.explained_variance_ratio_
            return Projection(rows=rows, xy=freeze(xy - off), method=m,
                              pca=pca, offset=off,
                              explained=(float(ev[0]), float(ev[1])))
        from backend.reduction_2d import reduce_2d
        with engine.gate.slot(client, "proj", key):
            res = reduce_2d(m, X, {"random_state": 42, **params})
        return Projection(rows=rows,
                          xy=freeze(np.array(res.proj, dtype=float)), method=m)
    return engine.projections.get(key, compute)


def centroid_xy(an: Analysis, proj: Projection) -> list[tuple[str, int, float, float]]:
    """[(label, sub_index, x, y)] — one per prototype row."""
    out = []
    Vn = an.frame.Vn
    shown = np.zeros(an.frame.n, dtype=bool)
    shown[proj.rows] = True
    pos = proj.pos
    for lbl, C in an.centroids.items():
        k = an.labels.index(lbl) if lbl in an.labels else -1
        members = np.flatnonzero(an.fit_mask & (an.true_idx == k)) if k >= 0 \
            else np.array([], dtype=int)
        Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        owner = (np.argmax(Vn[members] @ Cn.T, axis=1)
                 if members.size and Cn.shape[0] > 1
                 else np.zeros(members.size, dtype=int))
        for j in range(Cn.shape[0]):
            mine = members[owner == j] if members.size else members
            if proj.pca is not None:
                scale = float((Vn[mine] @ Cn[j]).mean()) if mine.size else 1.0
                p = proj.pca.transform((Cn[j] * scale)[None, :])[0] - proj.offset[0]
                out.append((lbl, j, float(p[0]), float(p[1])))
            else:
                vis = [pos[int(r)] for r in mine if shown[r]]
                if vis:
                    p = proj.xy[vis].mean(axis=0)
                    out.append((lbl, j, float(p[0]), float(p[1])))
    return out


def hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Closed convex-hull polygon of 2D points (None if degenerate)."""
    if points.shape[0] < 3:
        return None
    try:
        from scipy.spatial import ConvexHull
        h = ConvexHull(points)
    except Exception:
        return None
    v = np.append(h.vertices, h.vertices[0])
    return points[v, 0], points[v, 1]


def accepted_hulls(an: Analysis, proj: Projection,
                   own_ok: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per class: hull of displayed members their own class accepts — the
    'accepted envelope'. Valid in any reduction (the old cosine ring was
    only meaningful in PCA and blew past the viewport at thr=0)."""
    out = {}
    t = an.true_idx[proj.rows]
    ok = own_ok[proj.rows]
    for k, lbl in enumerate(an.labels):
        h = hull(proj.xy[(t == k) & ok])
        if h is not None:
            out[lbl] = h
    return out
