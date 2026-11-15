"""Hyperspherical representations with a uniform API.

Every class takes only hyperparameters in ``__init__`` and exposes:

    rep.fit(X, y) -> self               # learn from labeled data
    rep.predict(X) -> (cls, score, votes)  # cls/score (N,), votes (N, K)
    rep.score(X)                        # max per-class similarity (N,)  — NOT argmax
    rep.score_pairs(X)                  # (ri, ci, scores) for pairwise verification

Swap methods by changing one import in the caller. See the taxonomy in
``summaries/representations/represenations.md`` (Clusters A–E + the
Bingham–Geodesic kernel in §III.3 and AC-PAN in §III.4).
"""

from .ac_pan import ACPAN
from .arc_cosine import ArcCosineKernel
from .arc_cosine_knn import ArcCosineKNN
from .arc_cosine_svm import ArcCosineSVM
from .base import Representation
from .bingham_geodesic import BinghamGeodesic
from .cosface import CosFace
from .cosine import Cosine
from .knn import KNN
from .label_propagation import LabelPropagation
from .methodology import METHOD_REGISTRY, Methodology
from .multi_proxy import MultiProxyAnchor
from .ncm import NCM
from .src import SRC
from .vmf_mixture import VMFMixture

__all__ = [
    "Representation",
    "Methodology",
    "METHOD_REGISTRY",
    # A — global aggregation
    "NCM",
    "VMFMixture",
    # B — local manifold / graph
    "KNN",
    "LabelPropagation",
    # C — learned subspace / margin
    "CosFace",
    "MultiProxyAnchor",
    # D — kernel
    "ArcCosineKernel",
    "ArcCosineSVM",
    "ArcCosineKNN",
    # E — reconstruction
    "SRC",
    # III.3 / III.4 — new kernel + grand merger
    "BinghamGeodesic",
    "ACPAN",
    # Baseline
    "Cosine",
]
