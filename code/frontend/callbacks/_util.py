"""Helpers shared by the render callbacks."""
from __future__ import annotations

from dash import Patch, html, no_update


def patch(fig, tags: list[str]) -> Patch:
    """Patch holding only the trace props tagged with any of ``tags``
    (see ``figures.common.dyn``)."""
    p = Patch()
    for i, tr in enumerate(fig.data):
        dyn = (tr.meta or {}).get("dyn") or {}
        props = {path for t in tags for path in dyn.get(t, ())}
        if not props:
            continue
        js = tr.to_plotly_json()
        for path in sorted(props):
            src, dst = js, p["data"][i]
            *parents, leaf = path.split(".")
            for part in parents:
                src, dst = (src or {}).get(part, {}), dst[part]
            dst[leaf] = (src or {}).get(leaf)
    return p


def respond(fig, sig: str, aspects: dict[str, str], held: dict | None):
    """Decide what to send for a figure, given what the browser HOLDS.

    ``held`` is the panel's ``sig-*`` store echoed back as State: the
    structural signature and one hash per dynamic aspect (thresholds /
    selection / focus) of the last response the browser actually applied.
    Dash drops the older of two overlapping responses of one callback, so
    "what triggered me" is not enough — a dropped threshold patch would
    leave stale marks forever. Comparing against ``held`` instead makes
    every answer bring the figure fully up to date:

        structure differs  → the whole figure
        some aspects differ → one Patch covering ALL of them
        nothing differs    → no_update

    Returns ``(figure | Patch | no_update, new_held, changed_tags | None)``.
    """
    now = {"sig": sig, **aspects}
    if not isinstance(held, dict) or held.get("sig") != sig:
        return fig, now, None
    changed = [t for t, h in aspects.items() if held.get(t) != h]
    return (patch(fig, changed) if changed else no_update), now, changed


def chip(value: str, label: str, cls: str = "") -> html.Span:
    return html.Span([html.B(value), html.Span(label)],
                     className=f"chip {cls}".strip())
