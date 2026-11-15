"""PCA cosine-threshold ring geometry — shared by build_pca_fig and callbacks."""
from __future__ import annotations

import numpy as np


__all__: list[str] = []


_RING_THETA = np.linspace(0.0, 2.0 * np.pi, 96)
_RING_COS = np.cos(_RING_THETA)
_RING_SIN = np.sin(_RING_THETA)


def _ring_xy(cx: float, cy: float, tau, pca_scale: float = 1.0):
    """Cosine-threshold ring scaled to the PCA coordinate system.

    The raw geodesic bound in unit-sphere space is r = sqrt(2 - 2τ).
    After PCA projection (which is a contraction), ring radii in the
    original space must be multiplied by ``pca_scale`` — the RMS spread
    of the projected data — so the ring visually encloses the relevant
    cluster rather than living in a mismatched coordinate system.

    ``pca_scale`` defaults to 1.0 so existing callers that don't pass it
    remain unchanged; build_pca_fig always passes the computed value.
    """
    # Default to τ=0 when not specified — matches slider default and keeps
    # the ring visible from the moment the figure is first rendered, before
    # the user has touched any slider (which previously left tau=None and
    # collapsed the ring to a single point).
    if tau is None:
        tau = 0.0
    if isinstance(tau, (int, float)) and not isinstance(tau, bool):
        t = float(np.clip(tau, -1.0, 1.0))
        r = float(np.sqrt(max(2.0 - 2.0 * t, 0.0))) * float(pca_scale)
    else:
        r = 0.0
    return cx + r * _RING_COS, cy + r * _RING_SIN
