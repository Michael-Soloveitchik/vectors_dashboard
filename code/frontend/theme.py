"""Design tokens + Plotly base layout. CSS mirrors these in assets/style.css."""
from __future__ import annotations

TOKENS = {
    "dark": {
        "bg": "#0a0e16", "panel": "#1d2332", "raised": "#262d3f",
        "border": "#333b50", "text": "#e6e9ef", "muted": "#9aa3b5",
        "grid": "rgba(255,255,255,0.07)", "axis": "rgba(255,255,255,0.22)",
        "accent": "#60cdff", "gold": "#f5c542", "bad": "#ff7b7b",
        "neutral": "#6b7385", "hover_bg": "#1d2332",
        "ring": "#4d7cff",      # marks the clicked vector: no class has it
    },
    "light": {
        "bg": "#eef2f9", "panel": "#ffffff", "raised": "#f4f6fb",
        "border": "#d5dbe8", "text": "#1c2330", "muted": "#5d6880",
        "grid": "rgba(20,30,60,0.08)", "axis": "rgba(20,30,60,0.25)",
        "accent": "#0067c0", "gold": "#b7791f", "bad": "#c93c3c",
        "neutral": "#9aa3b5", "hover_bg": "#ffffff",
        "ring": "#1e3a8a",
    },
}

# Class colours: softened, blue-led, still clearly distinct.
PALETTE = [
    "#6ea8fe", "#f4a261", "#6fcf97", "#f28b9b", "#b39ddb",
    "#d0a77d", "#f0a6ca", "#5ec8d8", "#e3c567", "#9aa9bd",
    "#ffb86b", "#7dd3c8", "#a3d977", "#c9a0dc", "#f7a8a0",
    "#8fc4ee", "#d4cc6a", "#8fe0b0", "#f5c0b8", "#b5d6f2",
]
FONT = ("'Segoe UI Variable Text', 'Segoe UI', Inter, system-ui, "
        "-apple-system, sans-serif")
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def tokens(theme: str | None) -> dict:
    return TOKENS["light" if theme == "light" else "dark"]


def class_colors(labels: list[str]) -> dict[str, str]:
    """Stable colour per class: index in the SORTED label list, so a class
    keeps its colour across panels, methodologies and skip-lists."""
    return {lbl: PALETTE[i % len(PALETTE)]
            for i, lbl in enumerate(sorted(labels))}


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def base_layout(theme: str | None, **over) -> dict:
    """Transparent figure on the card background, no figure title (the
    card header owns titles and toggles — nothing can overlap)."""
    t = tokens(theme)
    axis = dict(gridcolor=t["grid"], zerolinecolor=t["grid"],
                linecolor=t["axis"], tickfont=dict(size=10, color=t["muted"]),
                title=dict(font=dict(size=11, color=t["muted"])),
                automargin=True)
    lay = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=11, color=t["text"]),
        margin=dict(l=44, r=12, t=8, b=36),
        xaxis=dict(axis), yaxis=dict(axis),
        legend=dict(font=dict(size=10, color=t["text"]),
                    bgcolor="rgba(0,0,0,0)", tracegroupgap=2,
                    itemsizing="constant"),
        hoverlabel=dict(bgcolor=t["hover_bg"], bordercolor=t["border"],
                        font=dict(family=FONT, size=11, color=t["text"])),
        autosize=True,
    )
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(lay.get(k), dict):
            lay[k] = {**lay[k], **v}
        else:
            lay[k] = v
    return lay
