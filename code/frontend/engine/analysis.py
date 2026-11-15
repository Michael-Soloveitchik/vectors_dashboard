"""Analysis — one fitted methodology + every array the panels share.

A ``FitSpec`` (the sidebar state that changes the score function) maps to
one cached ``Analysis``. All panels read raw / normalized scores and
decisions from it, so they can never disagree and nothing is refit or
re-predicted per render.

Thresholds live in NORMALIZED score space [0, 1]: class ``c`` accepts a
sample iff ``norm[:, c] >= thr[c]``. That single rule drives the scatter
markers, the distribution lines, the ROC operating points, the confusion
matrix and the CSV export.

Threading (see ``concurrency.py``): an ``Analysis`` is immutable after
``_fit`` — shared arrays are frozen, derived results live in bounded
single-flight caches (``memo``), and every call into the fitted
representation goes through ``rep_lock`` because the 13 backend classes
are not audited for concurrent use.
"""
from __future__ import annotations

from collections import OrderedDict

import threading
from dataclasses import dataclass, field

import numpy as np

from .concurrency import FlightCache, HeavyGate, KindCache, freeze
from .data import DataHub, ModelFrame
from .fit import fit_analysis

MULTI_CENTROID_ARG = {
    "NCM": "n_clusters",
    "VMFMixture": "n_components",
    "MultiProxyAnchor": "n_proxies",
}
MAX_CLASSES = 200
MAX_N_CENTROIDS = 12
# entries kept per kind of derived result, per analysis
MEMO_CAPS = {"norm": 3, "class_curves": 3, "pair_curves": 2, "pair_matrix": 3}


@dataclass(frozen=True)
class FitSpec:
    model: str
    label_keys: tuple[str, ...]
    methodology: str
    aggregator: str
    n_global: int = 1
    per_class_n: tuple[tuple[str, int], ...] = ()
    skip: tuple[str, ...] = ()
    labels_rev: int = 0        # revision of the user's own labels, if used

    @classmethod
    def from_dict(cls, d: dict) -> "FitSpec":
        def n(v) -> int:
            return min(MAX_N_CENTROIDS, max(1, int(v or 1)))
        return cls(
            model=str(d["model"]),
            label_keys=tuple(str(k) for k in (d.get("label_keys") or ())),
            methodology=str(d.get("methodology") or "NCM"),
            aggregator=str(d.get("aggregator") or "mean"),
            n_global=n(d.get("n_global")),
            per_class_n=tuple(sorted(
                (str(k), n(v)) for k, v in (d.get("per_class_n") or {}).items()
                if v)),
            skip=tuple(sorted(str(s) for s in (d.get("skip") or ()))),
            labels_rev=int(d.get("labels_rev") or 0),
        )

    def to_dict(self) -> dict:
        return {"model": self.model, "label_keys": list(self.label_keys),
                "methodology": self.methodology, "aggregator": self.aggregator,
                "n_global": self.n_global,
                "per_class_n": dict(self.per_class_n),
                "skip": list(self.skip), "labels_rev": self.labels_rev}


@dataclass
class Decision:
    accept: np.ndarray   # (N, K) bool — class k accepts row i
    pred: np.ndarray     # (N,) int — winning accepted class, -1 = rejected
    own_ok: np.ndarray   # (N,) bool — accepted by its TRUE class
    correct: np.ndarray  # (N,) bool — pred == true class


@dataclass
class Analysis:
    spec: FitSpec
    frame: ModelFrame
    y: np.ndarray                      # (N,) str class label per row
    fit_mask: np.ndarray               # (N,) bool — row took part in fit
    labels: list[str] = field(default_factory=list)   # class order of `raw`
    rep: object | None = None
    raw: np.ndarray | None = None      # (N, K) raw per-class scores
    true_idx: np.ndarray | None = None  # (N,) int, -1 if class not fitted
    centroids: dict[str, np.ndarray] = field(default_factory=dict)
    error: str | None = None
    memo: KindCache = field(default_factory=lambda: KindCache(MEMO_CAPS),
                            repr=False)
    rep_lock: threading.RLock = field(default_factory=threading.RLock,
                                      repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None and self.raw is not None

    @property
    def k(self) -> int:
        return len(self.labels)

    def norm(self, alpha: float | None) -> np.ndarray:
        """(N, K) normalized scores for blend ``alpha``. ``alpha`` is passed
        to the rep (no shared mutable NORM_ALPHA); read-only result."""
        a = round(float(1.0 if alpha is None else alpha), 3)

        def compute() -> np.ndarray:
            with self.rep_lock:
                cols = [np.asarray(self.rep.normalize_score(self.raw[:, k],
                                                            alpha=a),
                                   dtype=np.float64)
                        for k in range(self.raw.shape[1])]
            return freeze(np.column_stack(cols))
        return self.memo.get("norm", a, compute)

    def score_xy(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """The methodology's pair primitive, serialized per analysis."""
        with self.rep_lock:
            return np.asarray(self.rep.score_XY(A, B), dtype=np.float64)

    def thr_vector(self, thr: dict | None) -> np.ndarray:
        thr = thr or {}
        out = np.array([float(thr.get(lbl) or 0.0) for lbl in self.labels])
        return np.clip(np.nan_to_num(out), 0.0, 1.0)

    def argmax(self) -> np.ndarray:
        return np.asarray(self.rep.arg_closest_score_fn(self.raw, axis=1))

    def decide(self, thr: dict | None, alpha: float | None) -> Decision:
        P = self.norm(alpha)
        accept = P >= self.thr_vector(thr)[None, :]
        masked = np.where(accept, self.raw,
                          self.rep.farthest_score_sentinel_fn())
        pred = np.asarray(self.rep.arg_closest_score_fn(masked, axis=1))
        pred = np.where(accept.any(axis=1), pred, -1)
        has = self.true_idx >= 0
        own = np.zeros(len(pred), dtype=bool)
        own[has] = accept[np.flatnonzero(has), self.true_idx[has]]
        return Decision(accept=accept, pred=pred, own_ok=own,
                        correct=has & (pred == self.true_idx))

    def summary(self, thr: dict | None, alpha: float | None) -> dict:
        has = self.true_idx >= 0
        n = int(has.sum())
        if not self.ok or n == 0:
            return {}
        d = self.decide(thr, alpha)
        nontarget = np.ones_like(d.accept)
        nontarget[np.flatnonzero(has), self.true_idx[has]] = False
        nontarget[~has] = False
        return {
            "n": n, "k": self.k, "dim": self.frame.dim,
            "top1": float((self.argmax()[has] == self.true_idx[has]).mean()),
            "correct": float(d.correct[has].mean()),
            "rejected": float((d.pred[has] == -1).mean()),
            "fr": float((~d.own_ok[has]).mean()),
            "fa": float(d.accept[nontarget].mean()) if nontarget.any() else 0.0,
        }


class Engine:
    """Fitted analyses + projections, shared by every request thread.

    ``analysis(spec)`` is single-flight per spec: five panels asking for
    the same new fit trigger ONE fit; a different spec fits in parallel,
    limited by the heavy gate. ``client`` (a browser-tab id) enables
    latest-wins: a queued fit that no tab wants any more is skipped.
    """

    def __init__(self, hub: DataHub, max_entries: int = 4,
                 heavy_slots: int = 2) -> None:
        self.hub = hub
        self.fits = FlightCache("fits", max_entries)
        self.projections = FlightCache("projections", 12)
        self.ptsne: OrderedDict = OrderedDict()   # p-t-SNE models (ptsne.py)
        self.gate = HeavyGate(heavy_slots)

    def analysis(self, spec: FitSpec | dict | None,
                 client: str | None = None) -> Analysis | None:
        if not spec:
            return None
        if isinstance(spec, dict):
            if not spec.get("model"):
                return None
            spec = FitSpec.from_dict(spec)
        self.gate.want(client, "fit", spec)

        def compute() -> Analysis:
            frame = self.hub.frame(spec.model)
            with self.gate.slot(client, "fit", spec):
                return fit_analysis(spec, frame)
        return self.fits.get(spec, compute)

    def stats(self) -> dict:
        """Cache sizes / counters — used by the leak tests and the log."""
        return {"fits": {**self.fits.stats, "size": len(self.fits)},
                "projections": {**self.projections.stats,
                                "size": len(self.projections)},
                "gate": dict(self.gate.stats),
                "frames": len(self.hub._frames)}
