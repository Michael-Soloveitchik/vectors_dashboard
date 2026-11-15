"""Dendrogram coordinate extraction for Plotly rendering.

scipy's `dendrogram(no_plot=True)` returns the segment coordinates we
need (icoord/dcoord) without drawing anything via matplotlib. We
re-pack them into Plotly-friendly traces and ALSO compute the (x, y)
position of every internal cluster node so the front-end can drop
invisible scatter markers on them — those markers are how the user
clicks a junction in the tree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import dendrogram

from .clustering import HierarchyResult


def dendrogram_traces(
    result: HierarchyResult,
    line_color: str = "#444444",
    line_width: float = 1.4,
) -> dict[str, Any]:
    """Return Plotly-ready segment lines + per-node click positions."""
    if result.n < 2:
        return {
            "lines_x": [],
            "lines_y": [],
            "node_ids": [],
            "node_x": [],
            "node_y": [],
            "color": line_color,
            "width": line_width,
        }

    dd = dendrogram(result.Z, no_plot=True, color_threshold=-1)
    lines_x: list[float | None] = []
    lines_y: list[float | None] = []
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        # Each segment is a 4-vertex path that draws an inverted U.
        for x, y in zip(xs, ys):
            lines_x.append(float(x))
            lines_y.append(float(y))
        lines_x.append(None)  # break between segments
        lines_y.append(None)

    node_ids, node_x, node_y = _internal_node_positions(result, dd)
    return {
        "lines_x": lines_x,
        "lines_y": lines_y,
        "node_ids": node_ids,
        "node_x": node_x,
        "node_y": node_y,
        "color": line_color,
        "width": line_width,
    }


def _internal_node_positions(
    result: HierarchyResult,
    dd: dict,
) -> tuple[list[int], list[float], list[float]]:
    """Each internal cluster id gets the (x, y) of its U-shape top center.

    The k-th row of Z corresponds to internal cluster id (n + k). dd's
    icoord and dcoord are emitted in the same order as Z (one entry per
    merge), so we can zip them. The top of the U is at
    (mean(xs[1:3]), ys[1])  -- midpoint of the upper bar.
    """
    n = result.n
    ids: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    for k, (icoord, dcoord) in enumerate(zip(dd["icoord"], dd["dcoord"])):
        cluster_id = n + k
        x_top = 0.5 * (icoord[1] + icoord[2])
        y_top = float(dcoord[1])
        ids.append(cluster_id)
        xs.append(float(x_top))
        ys.append(y_top)
    return ids, xs, ys


def node_xy_lookup(traces: dict[str, Any]) -> dict[int, tuple[float, float]]:
    """Reverse lookup: cluster_id -> (x, y) of its junction marker."""
    return {
        nid: (x, y)
        for nid, x, y in zip(traces["node_ids"], traces["node_x"], traces["node_y"])
    }
