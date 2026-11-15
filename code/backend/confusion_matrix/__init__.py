"""Confusion matrix backend (V @ V.T over an aggregated session subset).

Public surface:
    build_matrix(df, session_ids) -> ConfusionMatrix
    sort_matrix(cm, df, sort_keys) -> ConfusionMatrix
    apply_color_transform(cm, abs_, power, lo, hi) -> ConfusionMatrix
"""

from .matrix import ConfusionMatrix, build_matrix
from .sort import sort_matrix, available_sort_keys
from .transforms import apply_color_transform

__all__ = [
    "ConfusionMatrix",
    "build_matrix",
    "sort_matrix",
    "available_sort_keys",
    "apply_color_transform",
]
