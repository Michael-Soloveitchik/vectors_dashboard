"""Spec alias: dashboard reads `backend/spots/reduction_2D.py`.

The implementation lives in `backend/reduction_2d/` to keep filenames
lowercase and importable, but the brief asks for this exact path so we
re-export everything here. Adding logic to this file would split the
algorithm in two — keep it as a forwarder.
"""

from ..reduction_2d import reduce_2d, ReductionResult, ReductionCache

__all__ = ["reduce_2d", "ReductionResult", "ReductionCache"]
