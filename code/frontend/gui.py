"""Top-level Gui — the whole public API.

    from frontend import Gui

    Gui.demo().run()                               # synthetic data
    Gui.real().run()                               # real embeddings + audio
    Gui(df).run(port=8070)                         # your dataframe
    Gui.from_arrays(X, {"speaker": y}).run()       # plain numpy

    gui = Gui(df, labels=["device_id"], methodology="NCM", theme="light")
    gui.run(mode="inline")     # in the notebook AND at the printed URL
    gui.run(mode="external")   # print the URL only   ·   mode="tab": open a tab
    gui.url                    # http://127.0.0.1:8070/
    gui.analysis(methodology="KNN").summary({}, 1.0)   # same engine, no browser
    gui.stop()                 # or:  with Gui(df) as gui: ...

``df`` needs one vector per row (``centroid`` column, any array-like) and
at least one label column. ``session_id`` / ``model_name`` are optional
(filled in when missing).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .engine import DataHub, Engine, FitSpec
from .engine.data import VECTOR_COL

THEMES = (None, "auto", "dark", "light")
DUMPS = Path(__file__).resolve().parents[2] / "dumps"


def _prepare(df: pd.DataFrame, vector_col: str) -> pd.DataFrame:
    """Friendly validation + defaults for the optional columns."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Gui needs a non-empty pandas DataFrame")
    if vector_col not in df.columns:
        raise ValueError(
            f"no vector column {vector_col!r} — columns are {list(df.columns)}; "
            f"pass vector_col=… or use Gui.from_arrays(X, labels)")
    df = df.rename(columns={vector_col: VECTOR_COL}) \
        if vector_col != VECTOR_COL else df.copy()
    if "model_name" not in df.columns:
        df["model_name"] = "embedding"
    if "session_id" not in df.columns:
        df["session_id"] = [f"row_{i:06d}" for i in range(len(df))]
    dup = df.duplicated(["model_name", "session_id"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicated (model_name, session_id) "
                         "rows — ids must be unique within a model")
    return df.reset_index(drop=True)


class Gui:
    def __init__(self, df: pd.DataFrame, *,
                 loader: Callable[[str], pd.DataFrame] | None = None,
                 models: list[str] | None = None,
                 wav_dir: str | Path | None = None,
                 vector_col: str = VECTOR_COL,
                 labels: list[str] | str | None = None,
                 methodology: str = "NCM", aggregator: str = "mean",
                 reduction: str = "PCA", theme: str | None = "auto",
                 max_points: int = 4000, heavy_slots: int = 2,
                 blas_threads: int | None = None,
                 labels_dir: str | Path | None = DUMPS) -> None:
        """
        loader / models  lazily load other embedding models by name
        wav_dir          folder with ``**/<session_id>.wav`` → audio inspector
        labels …theme    initial control values (all changeable in the UI);
                         theme "auto" = saved choice, else the OS setting
        labels_dir       where your own labels are saved (None = memory only)
        heavy_slots      max concurrent heavy jobs (fits, t-SNE/UMAP)
        blas_threads     cap numpy/BLAS threads for this process (shared box)
        """
        if theme not in THEMES:
            raise ValueError(f"theme must be one of {THEMES[1:]}")
        self.hub = DataHub(_prepare(df, vector_col), loader=loader,
                           extra_models=models, wav_dir=wav_dir,
                           labels_dir=labels_dir)
        self.engine = Engine(self.hub, heavy_slots=heavy_slots)
        self._defaults = {
            "labels": [labels] if isinstance(labels, str) else labels,
            "methodology": methodology, "aggregator": aggregator,
            "reduction": reduction, "max_points": max_points,
            "theme": None if theme in (None, "auto") else theme}
        self._blas = None
        if blas_threads:
            from threadpoolctl import threadpool_limits
            self._blas = threadpool_limits(limits=int(blas_threads))
        self._port: int | None = None
        self.url: str | None = None
        self.app = self._build_app()

    # ── constructors ────────────────────────────────────────────────
    @classmethod
    def demo(cls, **kw) -> "Gui":
        """Synthetic speaker/device embeddings (data/generate.py)."""
        from data.generate import load_and_enrich
        return cls(load_and_enrich()[0], **kw)

    @classmethod
    def real(cls, model: str | None = None, **kw) -> "Gui":
        """Real embeddings: every parquet is selectable, wavs → inspector."""
        from data.real_data import EMB_DIR, WAVS_DIR, available_models, load_real
        models = available_models()
        if not models:
            raise FileNotFoundError(f"no embedding parquet in {EMB_DIR}")
        first = model if model in models else models[0]
        kw.setdefault("wav_dir", WAVS_DIR)
        return cls(load_real(first), loader=load_real, models=models, **kw)

    @classmethod
    def from_arrays(cls, vectors, labels, ids=None, model: str = "embedding",
                    **kw) -> "Gui":
        """``vectors`` (N, D); ``labels`` an (N,) array or {name: (N,) array}."""
        X = np.asarray(vectors, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"vectors must be (N, D), got shape {X.shape}")
        cols = labels if isinstance(labels, dict) else {"label": labels}
        for name, col in cols.items():
            if len(col) != len(X):
                raise ValueError(f"label {name!r} has {len(col)} entries for "
                                 f"{len(X)} vectors")
        df = pd.DataFrame({k: np.asarray(v).astype(str) for k, v in cols.items()})
        df[VECTOR_COL] = list(X)
        df["model_name"] = model
        if ids is not None:
            df["session_id"] = [str(i) for i in ids]
        kw.setdefault("labels", list(cols)[:1])
        return cls(df, **kw)

    # ── server ──────────────────────────────────────────────────────
    def _build_app(self):
        import dash
        from .audio import register_routes
        from .server import make_relocatable
        from .callbacks import register
        from .debug_log import dlog, install_action_hook
        from .layout import build_layout
        app = dash.Dash(
            __name__, title="vectors dashboard",
            assets_folder=str(Path(__file__).resolve().parent / "assets"),
            suppress_callback_exceptions=True, update_title=None)
        install_action_hook(app.server)
        make_relocatable(app.server)
        register_routes(app.server, self.hub)
        dlog.event("boot", models=self.hub.models)
        app.layout = build_layout(self.hub, self._defaults)
        register(app, self.engine)
        return app

    def run(self, port: int = 8070, mode: str | None = None, *,
            host: str = "0.0.0.0", height: int = 900,
            jupyter_mode: str | None = None,
            jupyter_height: int | None = None) -> None:
        """Start (or restart) in a background thread and return at once.
        ``host="127.0.0.1"`` keeps it off the network; the default binds
        all interfaces so WSL2 / SSH / LAN browsers can reach it."""
        from .server import run_dash
        if self._port is not None:
            self.stop()
        self.url, self._port = run_dash(
            self.app, host=host, port=port, jupyter_mode=mode or jupyter_mode,
            jupyter_height=jupyter_height or height,
            in_jupyter=self._in_jupyter())

    def stop(self) -> None:
        from .server import stop
        if self._port is not None:
            stop(self._port)
        self._port = self.url = None

    def __enter__(self) -> "Gui":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
        if self._blas is not None:
            self._blas.restore_original_limits()

    # ── programmatic access to the same engine ──────────────────────
    @property
    def models(self) -> list[str]:
        return list(self.hub.models)

    def analysis(self, model: str | None = None,
                 labels: list[str] | None = None, methodology: str | None = None,
                 aggregator: str | None = None, n_centroids: int = 1,
                 skip: list[str] | None = None):
        """Fit (or fetch from cache) exactly what the UI would: returns the
        ``Analysis`` — ``.norm(alpha)``, ``.decide(thr, alpha)``,
        ``.summary(thr, alpha)``, ``.centroids`` …"""
        model = model or self.hub.models[0]
        d = self._defaults
        return self.engine.analysis(FitSpec.from_dict({
            "model": model,
            "label_keys": labels or d["labels"] or self.hub.default_labels(model),
            "methodology": methodology or d["methodology"],
            "aggregator": aggregator or d["aggregator"],
            "n_global": n_centroids, "skip": skip or []}))

    def __repr__(self) -> str:
        state = f"running at {self.url}" if self.url else "not running"
        return f"<Gui {len(self.hub.models)} model(s) {self.hub.models[:3]} · {state}>"

    @staticmethod
    def _in_jupyter() -> bool:
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            return ipy is not None and "IPKernelApp" in ipy.config
        except Exception:
            return False


__all__ = ["Gui"]
