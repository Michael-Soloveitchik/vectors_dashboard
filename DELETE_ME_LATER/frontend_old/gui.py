"""Top-level Gui class.

Construct with the enriched dataframe; call ``.run(...)`` to start the
Dash dev server.

    from data.generate import load_and_enrich
    from frontend import Gui
    df, _ = load_and_enrich()
    Gui(df=df).run(port=8050, jupyter_mode='external')
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .layout import build_layout
from .callbacks import register

CODE_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_COLS = ["session_id", "model_name", "centroid"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical column names. The toy / generator emit ``voice_id``; the
    dashboard speaks ``speaker_id``. Rename at the boundary so downstream
    code stays clean.  Also ensures device_id exists."""
    if "voice_id" in df.columns and "speaker_id" not in df.columns:
        df = df.rename(columns={"voice_id": "speaker_id"})
    if "device_id" not in df.columns:
        df = df.copy()
        df["device_id"] = df.get("speaker_id", "unknown").fillna("unknown")
    return df


class Gui:
    def __init__(self, df: pd.DataFrame) -> None:
        for cache in CODE_ROOT.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
        df = _normalize_columns(df)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"df missing required columns: {missing}")
        self.df = df.reset_index(drop=True).copy()
        self.app = self._build_app()

    def _build_app(self):
        import dash
        from .debug_log import install_action_hook, dlog
        app = dash.Dash(
            __name__, suppress_callback_exceptions=True,
            title="vectors_dashboard",
        )
        install_action_hook(app.server)
        dlog.event("boot", n_rows=int(len(self.df)))
        app.layout = build_layout(self.df)
        register(app, self.df)
        return app

    def run(self, port: int = 8050, jupyter_mode: str | None = None,
            jupyter_height: int = 820, host: str = "0.0.0.0") -> None:
        """Bind 0.0.0.0 by default so WSL2 / remote SSH browsers can reach
        the dev server via the host machine's IP (not just 127.0.0.1)."""
        from .server import run_dash
        run_dash(self.app, host=host, port=port,
                 jupyter_mode=jupyter_mode, jupyter_height=jupyter_height,
                 in_jupyter=self._in_jupyter())

    @staticmethod
    def _in_jupyter() -> bool:
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            return ipy is not None and "IPKernelApp" in ipy.config
        except Exception:
            return False


__all__ = ["Gui"]
