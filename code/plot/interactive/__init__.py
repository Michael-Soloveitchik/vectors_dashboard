"""Interactive dashboard package — re-exports ``build_interactive``.

Internal sub-modules carry the implementation:
- ``_utils``    : pure helpers (slider options, score ranking, trace copy)
- ``_nudge``    : ◀/▶ nudge button row builder
- ``_snapshot`` : download button + snapshot payload
- ``_cm_panel`` : confusion-matrix figure + click-to-PCA-highlight controller
- ``_ring``     : per-class ring distance/score precompute helper
- ``builder``   : main ``build_interactive`` glue function
"""
from __future__ import annotations

from .builder import build_interactive

__all__ = ["build_interactive"]
