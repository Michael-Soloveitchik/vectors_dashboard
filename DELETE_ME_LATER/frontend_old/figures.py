"""Adapter layer over the toy `code/plot/` builders.

This module is a thin re-export shim — actual implementations live in:

  - `figures_common.py`  — palette, _restrict, _centroids, _make_reducer
  - `figures_pca.py`     — pca_fig + _override_pca_rings + add_cm_highlight
  - `figures_cm.py`      — cm_fig + _cm_fig_from_score_matrix + sort helpers
  - `figures_dist.py`    — dist_fig + _append_selection_overlay
  - `figures_roc.py`     — roc_fig + _convert_to_det + sel overlay

Split per CLAUDE.md auto-split rule (270-line cap on Claude-created
files).  External callers continue to import from `frontend.figures`
unchanged — the rest is naming hygiene.
"""
from __future__ import annotations

# Common helpers (palette + state extraction + reducer factory + trapz).
from .figures_common import (  # noqa: F401
    DEFAULT_PALETTE,
    palette_for,
    _restrict,
    _centroids,
    _y_from_label_keys,
    _make_reducer,
    _run_pca,
    _norm_to_raw_thresholds,
    _trapz,
)

# Per-panel adapters.
from .figures_pca import (  # noqa: F401
    pca_fig,
    _override_pca_rings,
    _add_density_contour,
    _apply_pca_interaction,
    add_cm_highlight,
)
from .figures_cm import (  # noqa: F401
    cm_fig,
    _cm_fig_from_score_matrix,
    _sort_df_for_cm,
    _hierarchical_sort,
)
from .figures_dist import (  # noqa: F401
    dist_fig,
    _append_selection_overlay,
)
from .figures_roc import (  # noqa: F401
    roc_fig,
    _convert_to_det,
    _append_roc_selection_overlay,
)


__all__ = [
    "pca_fig", "cm_fig", "dist_fig", "roc_fig",
    "palette_for", "add_cm_highlight",
]
