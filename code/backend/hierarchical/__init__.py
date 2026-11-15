"""Agglomerative clustering of the confusion matrix.

Public surface:
    cluster(cm, method="average", metric="cosine") -> HierarchyResult
    dendrogram_traces(result, style) -> dict (Plotly traces)
    HierarchyLabels                                  (label-state manager)
"""

from .clustering import HierarchyResult, cluster
from .dendrogram import dendrogram_traces, node_xy_lookup
from .labels import HierarchyLabels

__all__ = [
    "HierarchyResult",
    "cluster",
    "dendrogram_traces",
    "node_xy_lookup",
    "HierarchyLabels",
]
