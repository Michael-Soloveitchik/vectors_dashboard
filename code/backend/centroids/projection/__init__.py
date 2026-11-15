"""Auto-merger for the ``projection`` package — re-exports every public
symbol from sibling modules.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


_PKG_DIR = Path(__file__).parent
_exported: list[str] = []

for _info in pkgutil.iter_modules([str(_PKG_DIR)]):
    _mod = importlib.import_module(f"{__name__}.{_info.name}")
    _names = getattr(_mod, "__all__", None)
    if _names is None:
        _names = [n for n in dir(_mod) if not n.startswith("_")]
    for _n in _names:
        globals()[_n] = getattr(_mod, _n)
        if _n not in _exported:
            _exported.append(_n)

__all__ = list(_exported)

del _info, _mod, _names, _n, _PKG_DIR, _exported
del importlib, pkgutil, Path
