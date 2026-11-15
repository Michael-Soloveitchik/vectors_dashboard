"""Callback registry — four modules wired in one call."""
from __future__ import annotations

import pandas as pd

from . import _dump, _fit, _interaction, _realdata, _render


def register(app, df: pd.DataFrame) -> None:
    _fit.register(app, df)
    _render.register(app, df)
    _interaction.register(app, df)
    _dump.register(app, df)
    _realdata.register(app, df)


__all__ = ["register"]
