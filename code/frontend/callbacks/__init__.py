"""Callback registry."""
from __future__ import annotations

from ..engine import Engine
from . import (export, labels, panels, render, select, selection, state,
               thresholds)


def register(app, engine: Engine) -> None:
    state.register(app, engine)
    thresholds.register(app, engine)
    render.register(app, engine)
    select.register(app, engine)
    selection.register(app, engine)
    labels.register(app, engine)
    export.register(app, engine)
    panels.register(app)


__all__ = ["register"]
