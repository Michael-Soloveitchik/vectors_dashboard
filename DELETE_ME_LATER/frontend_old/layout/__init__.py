"""Layout package — split into controls + grid for size budget.

`build_layout(df)` returns the Dash component tree.
"""
from ._grid import build_layout

__all__ = ["build_layout"]
