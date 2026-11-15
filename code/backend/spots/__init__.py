"""Thin re-export module so the spec-mandated path

    backend/spots/reduction_2D.py

resolves. The actual algorithm lives in `backend/reduction_2d/` —
this layer is a one-line forwarder so the dashboard import surface
matches the brief without duplicating code.
"""

from .reduction_2D import reduce_2d, ReductionResult, ReductionCache

__all__ = ["reduce_2d", "ReductionResult", "ReductionCache"]
