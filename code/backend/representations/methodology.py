"""Methodology — single facade over every concrete representation.

Instead of importing each concrete class and constructing it directly,
callers build one ``Methodology()`` and pick the underlying class at
fit-time:

    rep = Methodology()
    rep.fit(
        X, y,
        method='NCM',
        method_args={'n_clusters': 1},
        aggregator='mean',
    )
    cls, score, votes = rep.predict(X_te)
    rep.get_centroids()

Rules:
  • ``method`` and ``method_args`` are passed to ``fit`` (NOT
    ``__init__``). Each call to ``fit`` instantiates a fresh underlying
    impl from ``method_args``, so changing methodology = re-fit, no
    surgery on ``self``.
  • ``aggregator`` is forwarded to the impl's ``fit``. Prototype reps
    raise if ``None``; non-prototype reps ignore it.
  • Re-fit pattern: omitting ``method`` / ``method_args`` reuses the
    previous values (used by the aggregator toggle in ``interactive.py``
    + the animation cell in ``run.ipynb``).
  • Every public method, every protected ``_*`` extension point, and
    every shared convention helper is forwarded EXPLICITLY (not via
    ``__getattr__``). Method bodies are one-liners; the contract is
    visible at a glance.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

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
from .multi_proxy import MultiProxyAnchor
from .ncm import NCM
from .src import SRC
from .vmf_mixture import VMFMixture


METHOD_REGISTRY: dict[str, type[Representation]] = {
    "ACPAN":            ACPAN,
    "ArcCosineKernel":  ArcCosineKernel,
    "ArcCosineKNN":     ArcCosineKNN,
    "ArcCosineSVM":     ArcCosineSVM,
    "BinghamGeodesic":  BinghamGeodesic,
    "CosFace":          CosFace,
    "Cosine":           Cosine,
    "KNN":              KNN,
    "LabelPropagation": LabelPropagation,
    "MultiProxyAnchor": MultiProxyAnchor,
    "NCM":              NCM,
    "SRC":              SRC,
    "VMFMixture":       VMFMixture,
}


# Names of underlying-impl attributes that the facade surfaces as
# read-only properties. Each entry → ``Methodology.<name>`` returns
# ``self.impl.<name>``. Keep alphabetical inside groups.
_FORWARDED_PROPERTIES: tuple[str, ...] = (
    # Frozen state populated by fit().
    "_aggregator",
    "_fitted",
    "_labels",
    "_norm_ref",
    "_raw_is_posterior",
    "_uses_aggregator",
    # Concrete-rep prototype storage (used by interactive.py / nb_plots).
    "_centroids",
    "_means",                     # back-compat alias for old NCM/VMF name
    "_owner",
    "_proxies",
    "_proxy_labels",
    "_X",
    "_y",
    # Concrete-rep hyperparameters surfaced verbatim.
    "C",
    "covariance_type",
    "eps",
    "gamma",
    "k",
    "max_iter",
    "n_clusters",
    "n_components",
    "n_proxies",
    "order",
    "p",
    "random_state",
    "scale",
    "temperature",
    "variant",
)


class Methodology:
    """Facade — one class, methodology selected at fit-time.

    The facade owns ``self._impl`` after fit and forwards every call
    explicitly. Each concrete ``Representation`` is still importable
    and usable directly; the facade is purely additive.
    """

    # ──────────────────────────────────────────────────────────────────
    # Convention helpers — ``Representation`` exposes these as static
    # methods so subclasses can call ``self.closest_score_fn(...)``.
    # Re-export here so callers using ``rep.closest_first_order(scores)``
    # still work without going through ``rep.impl``.
    # ──────────────────────────────────────────────────────────────────
    closest_score_fn            = staticmethod(Representation.closest_score_fn)
    arg_closest_score_fn        = staticmethod(Representation.arg_closest_score_fn)
    closest_first_order         = staticmethod(Representation.closest_first_order)
    closest_last_order          = staticmethod(Representation.closest_last_order)
    topk_closest_fn             = staticmethod(Representation.topk_closest_fn)
    farthest_score_fn           = staticmethod(Representation.farthest_score_fn)
    arg_farthest_score_fn       = staticmethod(Representation.arg_farthest_score_fn)
    farthest_score_sentinel_fn  = staticmethod(Representation.farthest_score_sentinel_fn)
    a_closer_than_b_fn          = staticmethod(Representation.a_closer_than_b_fn)
    threshold_compare_fn        = staticmethod(Representation.threshold_compare_fn)
    score_to_distance_fn        = staticmethod(Representation.score_to_distance_fn)
    distance_to_score_fn        = staticmethod(Representation.distance_to_score_fn)
    closest_k_by_class_fn       = staticmethod(Representation.closest_k_by_class_fn)
    pair_kernel_closest_mass_fn = staticmethod(Representation.pair_kernel_closest_mass_fn)
    CLOSEST_IS_MAX              = Representation.CLOSEST_IS_MAX

    # ── construction / training ─────────────────────────────────────
    def __init__(self, normalize_data_on_sphere: bool = True) -> None:
        self._impl: Representation | None = None
        self._method_name: str | None = None
        self._method_args: dict[str, Any] = {}
        self.normalize_data_on_sphere = normalize_data_on_sphere

    def _sphere(self, X):
        """L2-normalize data vectors onto the unit sphere along last axis.

        No-op when ``normalize_data_on_sphere=False`` or ``X`` is None.
        Handles 1-D and 2-D inputs; zero-norm rows protected by eps.
        """
        if not self.normalize_data_on_sphere or X is None:
            return X
        X = np.asarray(X, dtype=np.float64)
        n = np.linalg.norm(X, axis=-1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return X / n

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        hints: dict | None = None,
        aggregator: str | Callable | None = None,
        method: str | None = None,
        method_args: dict[str, Any] | None = None,
    ) -> "Methodology":
        """Instantiate the underlying impl from ``method`` +
        ``method_args``, then call its ``fit`` with the same
        ``X, y, hints, aggregator``.

        ``method`` is a key in ``METHOD_REGISTRY`` (e.g. ``"NCM"``).
        ``method_args`` is a ``{kwarg: value}`` dict forwarded to the
        underlying ``__init__``. Both are remembered so a subsequent
        ``fit`` call (e.g. aggregator toggle) can omit them and reuse
        the prior selection.
        """
        if method is None:
            if self._method_name is None:
                raise ValueError(
                    "Methodology.fit: method=<class name> is required on "
                    f"the first call (one of {sorted(METHOD_REGISTRY)})."
                )
            method = self._method_name
            if method_args is None:
                method_args = self._method_args
        cls = METHOD_REGISTRY.get(method)
        if cls is None:
            raise ValueError(
                f"Methodology.fit: unknown method={method!r}. "
                f"Valid choices: {sorted(METHOD_REGISTRY)}"
            )
        kwargs = dict(method_args or {})
        self._impl = cls(**kwargs)
        self._method_name = method
        self._method_args = kwargs
        self._impl.fit(self._sphere(X), y, hints=hints, aggregator=aggregator)
        return self

    # ── inspection ──────────────────────────────────────────────────
    @property
    def impl(self) -> Representation:
        """Underlying concrete representation. Raises if not fit."""
        if self._impl is None:
            raise RuntimeError(
                "Methodology: not fit yet. Call "
                ".fit(..., method=..., method_args=...) first."
            )
        return self._impl

    @property
    def method_name(self) -> str | None:
        """Class name passed to the most recent ``fit`` call."""
        return self._method_name

    @property
    def method_args(self) -> dict[str, Any]:
        """Kwargs that were forwarded to the impl's ``__init__``."""
        return dict(self._method_args)

    def get_aggregator(self) -> str | Callable | None:
        return self.impl.get_aggregator()

    # ── public API — one-line passthroughs ──────────────────────────
    def transform(self, X):                  return self.impl.transform(self._sphere(X))
    def score_xy(self, x, y):                return self.impl.score_xy(self._sphere(x), self._sphere(y))
    def score_XY(self, X, Y):                return self.impl.score_XY(self._sphere(X), self._sphere(Y))
    def class_order_fn(self):                return self.impl.class_order_fn()
    def get_centroids(self):                 return self.impl.get_centroids()
    def normalize_score(self, scores, alpha=None):
        return self.impl.normalize_score(scores, alpha=alpha) \
            if "alpha" in self.impl.normalize_score.__code__.co_varnames \
            else self.impl.normalize_score(scores)
    def cosine_pairs(self, X):               return self.impl.cosine_pairs(self._sphere(X))

    def predict(self, X, thresholds=None,
                normalized_thresholds: bool = False, alpha: float | None = None):
        # Forward extra kwargs only if the impl supports them — keeps
        # back-compat with subclasses that haven't been updated yet.
        sig = self.impl.predict.__code__.co_varnames
        kw = {}
        if "normalized_thresholds" in sig:
            kw["normalized_thresholds"] = normalized_thresholds
        if "alpha" in sig:
            kw["alpha"] = alpha
        return self.impl.predict(self._sphere(X), thresholds=thresholds, **kw)

    def score_pairs(
        self,
        X,
        only_upper_triangle=False,
        skip_kernel=False,
        cos_similarity=None,            # deprecated alias
    ):
        return self.impl.score_pairs(
            self._sphere(X),
            only_upper_triangle=only_upper_triangle,
            skip_kernel=skip_kernel,
            cos_similarity=cos_similarity,
        )

    def self_consistency_check_fn(self, X_probe=None, n_probe=32):
        return self.impl.self_consistency_check_fn(
            X_probe=self._sphere(X_probe), n_probe=n_probe,
        )

    # ── protected extension points ─────────────────────────────────
    # These are normally called from inside subclass methods, but
    # ``self_consistency_check_fn`` and a few callers route through them
    # explicitly, so they need facade-level forwarding.
    def _require_fit(self):                  return self.impl._require_fit()
    def _mark_fitted(self, X_tr=None):       return self.impl._mark_fitted(self._sphere(X_tr))
    def _aggregator_reducer_fn(self, agg):   return self.impl._aggregator_reducer_fn(agg)
    def _posterior_matrix(self, X):          return self.impl._posterior_matrix(self._sphere(X))
    def _score_per_class_raw(self, X):       return self.impl._score_per_class_raw(self._sphere(X))
    def _pair_kernel(self, M_pair, Xn):       return self.impl._pair_kernel(M_pair, Xn)

    def _centroid_under_score(self, X, aggregator):
        return self.impl._centroid_under_score(self._sphere(X), aggregator=aggregator)


# ──────────────────────────────────────────────────────────────────────
# Read-only attribute forwarding via property factory.
#
# Concrete reps stash hyperparameters and fitted state on themselves
# (e.g. ``rep.temperature``, ``rep._means``). Code that historically
# used those attributes still works through the facade — the property
# factory turns each name into ``self.impl.<name>``.
#
# Name-explicit (no ``__getattr__`` magic) so typos surface as
# ``AttributeError`` at the read site, not a silent fall-through.
# ──────────────────────────────────────────────────────────────────────
def _make_forwarder(name: str):
    def _get(self):
        return getattr(self.impl, name)
    _get.__name__ = name
    _get.__doc__ = f"Read-only passthrough to ``self.impl.{name}``."
    return property(_get)


for _attr in _FORWARDED_PROPERTIES:
    setattr(Methodology, _attr, _make_forwarder(_attr))
del _attr


__all__ = ["Methodology", "METHOD_REGISTRY"]
