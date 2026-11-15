"""build_dist_fig — per-class score bins with raw/normalized + threshold lines."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .style import _BTN


__all__ = ["build_dist_fig"]


def build_dist_fig(*, rep, V_te, dev_te, devices_sorted, cmap,
                   method_label, model_name, use_raw: bool, class_thresholds,
                   class_labels):
    """Per-class score bins with raw↔normalized toggle + per-bin threshold lines.

    Both views come from ``rep.predict`` — raw uses ``raw=True``; normalized
    applies ``rep.normalize_score`` to each class column (matches the ROC
    threshold domain exactly). Threshold lines are drawn ONLY in normalized
    view — slider values live in the normalized domain.
    """
    _, _, raw_mat = rep.predict(V_te)
    # Single source of truth: rep.normalize_score (blend ECDF+minmax with
    # global NORM_ALPHA). Same domain as ROC x-axis and slider readout.
    norm_mat = np.empty_like(raw_mat, dtype=np.float64)
    for ki in range(raw_mat.shape[1]):
        norm_mat[:, ki] = rep.normalize_score(raw_mat[:, ki])

    # Raw thresholds = slider values. The normalized-view dash needs the
    # same sample cutoff in normalized units — look up the (raw → norm)
    # pair produced by rep.normalize_score on that class's column. The
    # slider option grid is unique(raw), so every raw_thr hits an exact
    # sample value and the lookup is well-defined.
    def _raw_to_norm(raw_thr: float, ki: int) -> float:
        raw_col = raw_mat[:, ki]
        norm_col = norm_mat[:, ki]
        j = int(np.argmin(np.abs(raw_col - raw_thr)))
        return float(norm_col[j])

    thr_raw = np.full(len(class_labels), np.nan, dtype=np.float64)
    if isinstance(class_thresholds, dict):
        for i, c in enumerate(class_labels):
            v = class_thresholds.get(c)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                thr_raw[i] = float(v)
    has_thr = np.any(np.isfinite(thr_raw))
    thr_raw_vals = np.where(np.isfinite(thr_raw), thr_raw, 0.0) \
        if has_thr else np.zeros(len(class_labels))
    # Plot dashes at the slider value directly, clipped to the [0, 1] x-range.
    # The previous _raw_to_norm rank-mapping placed slider=0 at the median of
    # each class's score distribution (~middle of the plot), which made the
    # init position look "in the middle" instead of the requested 0.
    thr_norm_vals = np.clip(thr_raw_vals, 0.0, 1.0)

    fig = go.Figure()
    box_idxs: list[int] = []

    # Numeric x-axis (one integer per class bin) so threshold Scatter traces
    # can draw horizontal dashes at ki ± 0.42 without creating spurious
    # categories. Tick labels remain the class_labels strings.
    def add_box(matrix, visible, legend_only: bool):
        for ki, cl in enumerate(class_labels):
            col = matrix[:, ki]
            for d in devices_sorted:
                mask = dev_te == d
                if not mask.any():
                    continue
                n = int(mask.sum())
                box_idxs.append(len(fig.data))
                fig.add_trace(go.Box(
                    y=[ki] * n, x=col[mask], orientation="h",
                    name=str(d), legendgroup=str(d),
                    showlegend=(legend_only and ki == 0),
                    marker=dict(size=3, color=cmap[d], opacity=0.75),
                    line=dict(color=cmap[d], width=1),
                    fillcolor="rgba(0,0,0,0)",
                    boxpoints="all", jitter=0.35, pointpos=0.0, whiskerwidth=0.2,
                    visible=visible,
                    hovertemplate=(f"<b>bin:</b> {cl}<br><b>true:</b> {d}<br>"
                                   "<b>score:</b> %{x:.3f}<extra></extra>"),
                ))

    def add_thr(y_per_bin, visible):
        """One vertical dash per bin (rotated layout). All share
        legendgroup='threshold'; only the first trace shows in the legend —
        clicking it toggles the whole group on/off at once."""
        for ki, cl in enumerate(class_labels):
            v = float(y_per_bin[ki])
            fig.add_trace(go.Scatter(
                y=[ki - 0.42, ki + 0.42], x=[v, v], mode="lines",
                name="threshold", legendgroup="threshold",
                showlegend=(ki == 0),
                line=dict(color="rgba(255,215,0,0.9)", width=2, dash="dot"),
                hovertemplate=(f"<b>bin:</b> {cl}<br>"
                               "<b>threshold:</b> %{x:.3f}<extra></extra>"),
                visible=visible,
            ))

    def add_hist(matrix, visible, legend_only: bool, n_bins: int = 25):
        """Per-(class, device) histogram drawn as narrow Bar (low-opacity
        fill) + Scatter step polyline (bold colored outline). Y-axis is
        reversed, so to make columns rise UPWARD visually we anchor bars
        at the row's bottom edge (data y = ki + 0.45) and emit negative
        height (y = -h) so the bar spans [ki+0.45, ki+0.45 - h], i.e.
        data goes toward smaller y = visually upward."""
        lo = float(np.nanmin(matrix))
        hi = float(np.nanmax(matrix))
        if hi - lo < 1e-9:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0])
        max_h = 0.85
        for ki, cl in enumerate(class_labels):
            col = matrix[:, ki]
            per_dev: dict = {}
            for d in devices_sorted:
                mask = dev_te == d
                if not mask.any():
                    continue
                counts, _ = np.histogram(col[mask], bins=edges)
                per_dev[d] = counts
            if not per_dev:
                continue
            peak = max(int(c.max()) for c in per_dev.values()) or 1
            base_y = ki + 0.45
            for d, counts in per_dev.items():
                h = counts.astype(np.float64) / peak * max_h
                # Narrow bar with low-opacity fill — visual mass without
                # blocking the underlying step outline.
                fig.add_trace(go.Bar(
                    x=centers, y=-h, base=base_y, width=width * 0.25,
                    name=str(d), legendgroup=str(d),
                    showlegend=(legend_only and ki == 0),
                    marker=dict(color=cmap[d], opacity=0.18,
                                line=dict(width=0)),
                    visible=visible,
                    hovertemplate=(f"<b>bin:</b> {cl}<br><b>true:</b> {d}<br>"
                                   "<b>score:</b> %{x:.3f}<br>"
                                   "<b>count(norm):</b> %{base:.2f}<extra></extra>"),
                ))
                # Step polyline: hvh staircase across edges.
                xs = [edges[0]]
                ys = [0.0]
                for j in range(n_bins):
                    xs += [edges[j], edges[j + 1]]
                    ys += [h[j], h[j]]
                xs.append(edges[-1])
                ys.append(0.0)
                step_y = base_y - np.asarray(ys, dtype=np.float64)
                fig.add_trace(go.Scatter(
                    x=xs, y=step_y, mode="lines",
                    name=str(d), legendgroup=str(d), showlegend=False,
                    line=dict(color=cmap[d], width=1.4),
                    visible=visible, hoverinfo="skip",
                ))

    # ONE domain only: normalized (rep.normalize_score blend with global
    # NORM_ALPHA). Raw box + raw hist removed — slider, ROC, dash all
    # speak the same normalized language. Toggle is box ↔ hist only.
    # Trace order: box_norm | hist_norm | thr_norm. Thresholds LAST so
    # dashes draw on top of bars.
    i = lambda: len(fig.data)
    i0 = i(); add_box(norm_mat, True, True);  n_norm = i() - i0
    i0 = i(); add_hist(norm_mat, False, True); n_hn  = i() - i0
    i0 = i(); add_thr(thr_norm_vals, True);    n_tn  = i() - i0

    def _vis(kind: str) -> list[bool]:
        is_box = (kind == "box")
        return (
            [is_box] * n_norm +
            [(not is_box)] * n_hn +
            [True] * n_tn
        )

    vis_box  = _vis("box")
    vis_hist = _vis("hist")

    # Reserve a SELECTED lane at index = n_classes (always present even
    # without selection). Empty placeholder Box → overwritten by
    # `_append_selection_overlay` as selection changes. Avoids relayout
    # on every selection update.
    n_lanes = len(class_labels) + 1   # +1 for SELECTED
    fig.add_trace(go.Box(
        x=[], y=[float(len(class_labels))],
        orientation="h",
        marker=dict(color="#ffd700", size=6, opacity=0.95,
                    line=dict(color="#000", width=0.5)),
        line=dict(width=1.5, color="#ffd700"),
        fillcolor="rgba(255,215,0,0.15)",
        boxpoints="all", jitter=0.4, pointpos=0.0,
        hoverinfo="x",
        name="SELECTED",
        legendgroup="selection-overlay",
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(
            text=f"Per-class score distribution — {method_label} — {model_name}",
            y=0.97, yanchor="top", pad=dict(b=18),
        ),
        yaxis=dict(
            title=f"class bin (rep={method_label})",
            tickmode="array",
            tickvals=list(range(n_lanes)),
            ticktext=[str(c) for c in class_labels] + ["SELECTED"],
            range=[n_lanes - 0.5, -0.5],
            showgrid=False, zeroline=False, ticks="",
        ),
        xaxis=dict(title="normalized score (α-blend)",
                   type="linear", range=[-0.02, 1.02], autorange=False,
                   showgrid=False, zeroline=False),
        boxmode="group", barmode="overlay",
        template="plotly_dark", width=720, height=max(420, 110 * n_lanes),
        legend=dict(title_text="legend", font=dict(size=10), tracegroupgap=3),
        margin=dict(t=170),
        updatemenus=[
            dict(type="buttons", direction="right", x=0.01, y=1.10,
                 xanchor="left", yanchor="bottom", **_BTN, buttons=[dict(
                     label="→ hist", method="update",
                     args=[{"visible": vis_hist},
                           {"updatemenus[0].buttons[0].label": "→ box"}],
                     args2=[{"visible": vis_box},
                            {"updatemenus[0].buttons[0].label": "→ hist"}],
                 )]),
            dict(type="buttons", direction="right", x=0.18, y=1.10,
                 xanchor="left", yanchor="bottom", **_BTN, buttons=[dict(
                     label="points off", method="update",
                     args=[{"boxpoints": "outliers"},
                           {"updatemenus[1].buttons[0].label": "points on"},
                           box_idxs],
                     args2=[{"boxpoints": "all"},
                            {"updatemenus[1].buttons[0].label": "points off"},
                            box_idxs],
                 )]),
        ],
    )
    fig.layout.meta = {"vis_box": vis_box, "vis_hist": vis_hist,
                       "box_idxs": box_idxs}
    return fig
