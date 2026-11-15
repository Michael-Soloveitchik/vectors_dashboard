"""Plot package — figures, toggles, and the interactive dashboard entry.

Public API:
  • build_roc_fig       — pair·cosine + pair·normalized + per-device ROCs
  • build_dist_fig      — per-class score bins with raw/posterior toggle
  • build_pca_fig       — sample + centroid PCA scatter
  • build_interactive   — composes all three as a live FigureWidget dashboard
"""
from __future__ import annotations

from .style import _BTN
from .roc import build_roc_fig
from .dist import build_dist_fig
from .pca import build_pca_fig
from .interactive import build_interactive

__all__ = [
    "_BTN",
    "build_roc_fig", "build_dist_fig", "build_pca_fig", "build_interactive",
]
