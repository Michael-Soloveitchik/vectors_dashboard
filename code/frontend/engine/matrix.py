"""Matrix panel data: pair-score matrix, centroid matrix, confusion matrix.

Vector mode is capped (stratified) at ``MATRIX_CAP`` rows — an N×N
heatmap of the 46k-row real embeddings is neither computable nor
readable. The cap is reported in the panel subtitle.
"""
from __future__ import annotations

import numpy as np

from .analysis import Analysis, Decision

MATRIX_CAP = 500
SYNTHETIC_SORTS = ("class", "hierarchical", "random")


def pick_rows(an: Analysis, sel_rows: list[int] | None,
              cap: int = MATRIX_CAP) -> np.ndarray:
    """Rows shown in vector mode: the selection (or everything), thinned
    per class so every class keeps a proportional share."""
    base = (np.asarray(sorted(sel_rows), dtype=int) if sel_rows
            else np.arange(an.frame.n))
    if base.size <= cap:
        return base
    rng = np.random.default_rng(42)
    y = an.y[base]
    keep = []
    for lbl in np.unique(y):
        idx = base[y == lbl]
        take = max(2, int(round(cap * idx.size / base.size)))
        keep.append(idx if idx.size <= take
                    else rng.choice(idx, take, replace=False))
    return np.sort(np.concatenate(keep))


def order_rows(an: Analysis, rows: np.ndarray,
               sort_keys: list[str] | None) -> np.ndarray:
    """Apply sort keys left→right as a stable multi-key sort.
    ``hierarchical`` / ``random`` set the base order; column keys (and
    ``class``) are then stable-sorted on top, so e.g. [class,
    hierarchical] gives class blocks with a dendrogram order inside."""
    keys = list(sort_keys or [])
    rows = np.asarray(rows, dtype=int)
    if "random" in keys:
        rows = rows[np.random.default_rng(42).permutation(rows.size)]
    if "hierarchical" in keys and rows.size >= 3:
        base = rows

        def leaves() -> np.ndarray:      # O(n²) linkage: once per row list
            from scipy.cluster.hierarchy import leaves_list, linkage
            X = an.frame.Vn[base]
            d = np.clip(1.0 - X @ X.T, 0.0, 2.0)
            iu = np.triu_indices(base.size, k=1)
            return leaves_list(linkage(d[iu], method="average"))
        rows = base[an.memo.get("hier_order", base.tobytes(), leaves)]
    cols = [k for k in keys if k not in ("random", "hierarchical")]
    for key in reversed(cols):
        if key == "class":
            vals = an.y[rows].astype(str)
        elif key in an.frame.df.columns:
            vals = an.frame.df[key].iloc[rows].astype(str).to_numpy()
        else:
            continue
        rows = rows[np.argsort(vals, kind="stable")]
    return rows


def pair_matrix(an: Analysis, rows: np.ndarray) -> np.ndarray:
    """``rep.score_XY`` on the displayed rows — the methodology's own pair
    primitive, not raw cosine. Memoized on the row SET (bounded,
    single-flight); callers reorder."""
    srt = np.sort(rows)

    def compute() -> np.ndarray:
        V = an.frame.V[srt]
        return an.score_xy(V, V)
    M = an.memo.get("pair_matrix", srt.tobytes(), compute)
    pos = np.searchsorted(srt, rows)
    return M[np.ix_(pos, pos)]


def centroid_matrix(an: Analysis) -> tuple[np.ndarray, list[str]]:
    names, rows = [], []
    for lbl in an.labels:
        C = an.centroids.get(lbl)
        if C is None:
            continue
        for j in range(C.shape[0]):
            names.append(lbl if C.shape[0] == 1 else f"{lbl} #{j + 1}")
            rows.append(C[j])
    if not rows:
        return np.zeros((0, 0)), []
    C = np.vstack(rows)
    return an.score_xy(C, C), names


def class_tree(an: Analysis) -> dict | None:
    """Average-linkage tree over the classes (one prototype direction per
    class, cosine distance). → {"labels" (leaf order), "y" per node,
    "h" per node, "children", "members"} with nodes 0..K-1 = leaves."""
    names = [lb for lb in an.labels if lb in an.centroids]
    if len(names) < 2:
        return None

    def compute() -> dict:
        from scipy.cluster.hierarchy import leaves_list, linkage
        C = np.vstack([an.centroids[lb].mean(axis=0) for lb in names])
        C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        k = len(names)
        d = np.clip(1.0 - C @ C.T, 0.0, 2.0)
        Z = linkage(d[np.triu_indices(k, 1)], method="average")
        order = leaves_list(Z)
        y = {int(leaf): float(i) for i, leaf in enumerate(order)}
        h = {i: 0.0 for i in range(k)}
        members = {i: [names[i]] for i in range(k)}
        children = {}
        for j, (a, b, dist, _n) in enumerate(Z):
            a, b, node = int(a), int(b), k + j
            children[node] = (a, b)
            y[node], h[node] = (y[a] + y[b]) / 2.0, float(dist)
            members[node] = members[a] + members[b]
        return {"labels": [names[i] for i in order], "y": y, "h": h,
                "children": children, "members": members, "k": k}
    return an.memo.get("class_tree", 0, compute)


def transform(M: np.ndarray, abs_on: bool, power: float) -> np.ndarray:
    """abs → sign-preserving power (same order as backend transforms)."""
    if M.size == 0:
        return M
    if abs_on:
        M = np.abs(M)
    p = float(power or 1.0)
    if p != 1.0:
        M = np.sign(M) * (np.abs(M) ** p)
    return M


def confusion(an: Analysis, d: Decision,
              rows: list[int] | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    """counts[true, pred] with a trailing 'rejected' column. Rows of
    classes that were skipped at fit time are left out."""
    K = an.k
    t, p = an.true_idx, d.pred
    if rows:
        idx = np.asarray(rows, dtype=int)
        t, p = t[idx], p[idx]
    has = t >= 0
    t, p = t[has], np.where(p[has] < 0, K, p[has])
    counts = np.zeros((K, K + 1), dtype=int)
    np.add.at(counts, (t, p), 1)
    return counts, list(an.labels), list(an.labels) + ["rejected"]


def runs(values: np.ndarray) -> list[tuple[int, int, str]]:
    """Contiguous runs [(start, end_exclusive, value)] of a label array."""
    out = []
    if values.size == 0:
        return out
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or values[i] != values[start]:
            out.append((start, i, str(values[start])))
            start = i
    return out
