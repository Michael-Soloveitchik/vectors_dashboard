"""ROC / DET curves, EER and operating points — all in normalized space.

Curves are computed once per (analysis, alpha) in the analysis' bounded
single-flight memo and STORED thinned to ``STORE_POINTS`` (AUC / EER are
taken from the full curve first): a 400k-pair ROC kept at full length
for every alpha visited was the largest avoidable allocation. Plots thin
further. The old panel shipped every pair as a marker with a per-point
hover string.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import Analysis

_trapz = getattr(np, "trapezoid", None) or np.trapz
PAIR_EXACT_MAX_N = 1200       # all pairs up to here (~720k)
PAIR_SAMPLE = 400_000
STORE_POINTS = 2000
# Detection cost (NIST SRE / VoxCeleb convention): C_miss = C_fa = 1.
P_TARGET = 0.01


def dcf(fpr, fnr, p_target: float = P_TARGET):
    """Normalized detection cost at given error rates (scalar or array)."""
    cost = p_target * np.asarray(fnr) + (1.0 - p_target) * np.asarray(fpr)
    return cost / min(p_target, 1.0 - p_target)


@dataclass
class Curve:
    key: str            # class label, "pair-cos", "pair-norm", "selection"
    name: str
    kind: str           # "class" | "pair" | "selection"
    fpr: np.ndarray
    tpr: np.ndarray
    thr: np.ndarray
    auc: float
    eer: float
    eer_thr: float
    dcf: float = float("nan")       # min normalized detection cost
    dcf_thr: float = float("nan")   # threshold where it is reached


def _curve(key: str, name: str, kind: str, pos: np.ndarray,
           score: np.ndarray) -> Curve | None:
    from sklearn.metrics import roc_curve
    n_pos = int(pos.sum())
    if n_pos == 0 or n_pos == pos.size:
        return None
    fpr, tpr, thr = roc_curve(pos.astype(np.int8), score,
                              drop_intermediate=True)
    thr = np.where(np.isfinite(thr), thr, 1.0)
    fnr = 1.0 - tpr
    j = int(np.argmin(np.abs(fnr - fpr)))
    costs = dcf(fpr, fnr)
    m = int(np.argmin(costs))
    full = Curve(key=key, name=name, kind=kind, fpr=fpr, tpr=tpr, thr=thr,
                 auc=float(_trapz(tpr, fpr)),
                 eer=float((fnr[j] + fpr[j]) / 2.0), eer_thr=float(thr[j]),
                 dcf=float(costs[m]), dcf_thr=float(thr[m]))
    full.fpr, full.tpr, full.thr = decimate(full, STORE_POINTS)
    return full


def _a(alpha: float | None) -> float:
    return round(float(1.0 if alpha is None else alpha), 3)


def decimate(c: Curve, n: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """≤ ~n points: uniform in index plus uniform in log-FPR so the
    low-FPR corner stays resolved on log / DET axes."""
    m = c.fpr.size
    if m <= n:
        return c.fpr, c.tpr, c.thr
    lin = np.linspace(0, m - 1, n // 2).astype(int)
    logs = np.searchsorted(c.fpr, np.logspace(-5, 0, n // 2))
    idx = np.unique(np.clip(np.concatenate([lin, logs, [0, m - 1]]), 0, m - 1))
    return c.fpr[idx].copy(), c.tpr[idx].copy(), c.thr[idx].copy()


def class_curves(an: Analysis, alpha: float | None) -> list[Curve]:
    """One-vs-rest curve per fitted class, score = that class's column."""
    def compute() -> list[Curve]:
        P = an.norm(alpha)
        out = []
        for k, lbl in enumerate(an.labels):
            c = _curve(lbl, lbl, "class", an.true_idx == k, P[:, k])
            if c is not None:
                out.append(c)
        return out
    return an.memo.get("class_curves", _a(alpha), compute)


def operating_points(an: Analysis, thr: dict | None,
                     alpha: float | None) -> dict[str, tuple[float, float, float]]:
    """{label: (fpr, tpr, thr)} at the current per-class thresholds."""
    P = an.norm(alpha)
    t = an.thr_vector(thr)
    out = {}
    for k, lbl in enumerate(an.labels):
        pos = an.true_idx == k
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        acc = P[:, k] >= t[k]
        out[lbl] = (float((acc & ~pos).sum() / n_neg),
                    float((acc & pos).sum() / n_pos), float(t[k]))
    return out


def eer_thresholds(an: Analysis, alpha: float | None) -> dict[str, float]:
    return {c.key: float(np.clip(c.eer_thr, 0.0, 1.0))
            for c in class_curves(an, alpha)}


def dcf_thresholds(an: Analysis, alpha: float | None) -> dict[str, float]:
    return {c.key: float(np.clip(c.dcf_thr, 0.0, 1.0))
            for c in class_curves(an, alpha)}


def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n <= PAIR_EXACT_MAX_N:
        return np.triu_indices(n, k=1)
    rng = np.random.default_rng(42)
    ri = rng.integers(0, n, PAIR_SAMPLE)
    ci = rng.integers(0, n, PAIR_SAMPLE)
    keep = ri != ci
    return ri[keep], ci[keep]


def _rowdot(A: np.ndarray, ri: np.ndarray, ci: np.ndarray,
            chunk: int = 50_000) -> np.ndarray:
    out = np.empty(ri.size, dtype=np.float64)
    for s in range(0, ri.size, chunk):
        e = s + chunk
        out[s:e] = np.einsum("ij,ij->i", A[ri[s:e]], A[ci[s:e]])
    return out


def pair_curves(an: Analysis, alpha: float | None) -> list[Curve]:
    """Same-class vs different-class pairs.

    pair·cosine     — raw embedding cosine: a methodology-independent
                      baseline. (Routing it through ``normalize_score``
                      clipped it to a constant for reps whose reference
                      grid is not cosine-scaled, e.g. SVM → AUC 0.5.)
    pair·normalized — inner product of normalized vote rows, rank-mapped
                      within the pair vector (its own distribution).
    """
    def compute() -> list[Curve]:
        from scipy.stats import rankdata
        ri, ci = _pair_indices(an.frame.n)
        same = an.y[ri] == an.y[ci]
        sampled = an.frame.n > PAIR_EXACT_MAX_N
        tag = f" · {ri.size // 1000}k sampled" if sampled else ""
        out = []
        c = _curve("pair-cos", f"pair · cosine{tag}", "pair", same,
                   _rowdot(an.frame.Vn, ri, ci))
        if c is not None:
            out.append(c)
        pn = _rowdot(an.norm(alpha), ri, ci)
        pn = (rankdata(pn, method="average") - 1.0) / max(pn.size - 1, 1)
        c = _curve("pair-norm", f"pair · normalized{tag}", "pair", same, pn)
        if c is not None:
            out.append(c)
        return out
    return an.memo.get("pair_curves", _a(alpha), compute)


def selection_curve(an: Analysis, alpha: float | None,
                    rows: list[int] | None) -> Curve | None:
    """Selected rows vs the rest, score = best normalized vote."""
    if not rows:
        return None
    n = an.frame.n
    sel = np.zeros(n, dtype=bool)
    sel[np.asarray(rows, dtype=int)] = True
    if sel.sum() < 2 or sel.sum() >= n:
        return None
    return _curve("selection", f"selection vs rest (n={int(sel.sum())})",
                  "selection", sel, an.norm(alpha).max(axis=1))


def to_det(x: np.ndarray, y_tpr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(FPR, TPR) → probit(FPR), probit(FNR)."""
    from scipy.stats import norm
    eps = 1e-5
    return (norm.ppf(np.clip(x, eps, 1 - eps)),
            norm.ppf(np.clip(1.0 - y_tpr, eps, 1 - eps)))


DET_TICKS = [0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
DET_LABELS = ["0.1%", "1%", "5%", "20%", "50%", "80%", "95%", "99%"]
