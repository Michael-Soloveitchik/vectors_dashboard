"""DataHub — per-model frames (df + stacked vectors), loaded once.

The dashboard never re-stacks the ``centroid`` column per callback: a
``ModelFrame`` holds the float matrix, its unit-norm copy and the id
lookups for one embedding model, and is cached for the process life.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .concurrency import FlightCache, freeze

VECTOR_COL = "centroid"
MY_LABEL = "my_label"         # the user's own labels (a real column)
UNLABELLED = "—"
MAX_LABEL_CARDINALITY = 200


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Generator emits ``voice_id``; the dashboard speaks ``speaker_id``."""
    if "voice_id" in df.columns and "speaker_id" not in df.columns:
        df = df.rename(columns={"voice_id": "speaker_id"})
    if "device_id" not in df.columns:
        df = df.copy()
        df["device_id"] = df.get("speaker_id", "unknown")
    return df


@dataclass
class ModelFrame:
    name: str
    df: pd.DataFrame          # reset index, no vector column
    Vn: np.ndarray            # (N, D) float64 unit-norm, READ-ONLY — the
    sids: list[str]           # space every representation works in
    labels_rev: int = 0       # bumped when the user's own labels change
    _row_of: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def V(self) -> np.ndarray:
        """Vectors as fed to the representations. They L2-normalize their
        input anyway, so one unit-norm copy is kept instead of two."""
        return self.Vn

    @property
    def n(self) -> int:
        return self.Vn.shape[0]

    @property
    def dim(self) -> int:
        return self.Vn.shape[1]

    def row_of(self, sid: str) -> int | None:
        return self._row_of.get(str(sid))

    def labels(self, label_keys: list[str] | tuple[str, ...]) -> np.ndarray:
        keys = [k for k in label_keys if k in self.df.columns]
        if not keys:
            return np.full(self.n, "all", dtype=object)
        if len(keys) == 1:
            return self.df[keys[0]].astype(str).to_numpy(dtype=object)
        return (self.df[keys].astype(str).agg(" | ".join, axis=1)
                .to_numpy(dtype=object))


def _build_frame(name: str, df_m: pd.DataFrame) -> ModelFrame:
    df_m = df_m.reset_index(drop=True)
    V = np.stack([np.asarray(v, dtype=np.float64).reshape(-1)
                  for v in df_m[VECTOR_COL].to_numpy()])
    norms = np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    sids = df_m["session_id"].astype(str).tolist()
    meta = df_m.drop(columns=[VECTOR_COL])
    if not np.isfinite(V).all():
        raise ValueError(f"model {name!r}: vectors contain NaN / inf")
    return ModelFrame(name=name, df=meta, Vn=freeze(V / norms), sids=sids,
                      _row_of={s: i for i, s in enumerate(sids)})


class DataHub:
    """Owns the dataframe(s). ``loader(model_name) -> df`` (optional) lets
    the model dropdown switch between embedding parquets lazily."""

    REQUIRED = ("session_id", "model_name", VECTOR_COL)

    def __init__(self, df: pd.DataFrame,
                 loader: Callable[[str], pd.DataFrame] | None = None,
                 extra_models: list[str] | None = None,
                 wav_dir: str | Path | None = None,
                 max_frames: int = 3,
                 labels_dir: str | Path | None = None) -> None:
        df = normalize_columns(df)
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"df missing required columns: {missing}")
        self._df = df
        self._loader = loader
        # Stacked matrices are the big objects (N×D float64 each): keep
        # only the most recently used models, rebuild the rest on demand.
        self._frames = FlightCache("frames", max_frames)
        self._lock = threading.RLock()
        in_df = sorted(df["model_name"].astype(str).unique().tolist())
        lazy = [m for m in (extra_models or []) if m not in in_df]
        self.models: list[str] = in_df + (lazy if loader else [])
        self._labels_dir = Path(labels_dir) if labels_dir else None
        self._revs: dict[str, int] = {}
        self._wav_dir = Path(wav_dir) if wav_dir else None
        self._wav_index: dict[str, Path] | None = None

    # ── frames ──────────────────────────────────────────────────────
    def frame(self, model: str) -> ModelFrame:
        """Single-flight: concurrent callbacks share one load / stack."""
        return self._frames.get(str(model), lambda: self._load(str(model)))

    def _load(self, model: str) -> ModelFrame:
        if model not in self.models:
            raise KeyError(f"unknown model {model!r}")
        sub = self._df[self._df["model_name"].astype(str) == model]
        if sub.empty and self._loader is not None:
            sub = normalize_columns(self._loader(model))
            sub = sub[sub["model_name"].astype(str) == model]
        if sub.empty:
            raise KeyError(f"no rows for model {model!r}")
        fr = _build_frame(model, sub)
        self._restore_labels(fr)
        return fr

    # ── the user's own labels ───────────────────────────────────────
    # One extra column (MY_LABEL). It is the only thing that ever changes
    # in a frame: the column is REPLACED (never edited in place) under the
    # hub lock, so readers see the old or the new Series, never a mix.
    # Saved next to the exports and restored on load, so labels survive
    # restarts and frame-cache eviction.
    def _labels_file(self, model: str) -> Path | None:
        if self._labels_dir is None:
            return None
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)
        return self._labels_dir / f"my_labels_{safe}.csv"

    def _restore_labels(self, fr: ModelFrame) -> None:
        f = self._labels_file(fr.name)
        fr.labels_rev = self._revs.get(fr.name, 0)
        if f is None or not f.exists():
            return
        try:
            saved = pd.read_csv(f, dtype=str).dropna()
            col = np.full(fr.n, UNLABELLED, dtype=object)
            for sid, lab in zip(saved["session_id"], saved[MY_LABEL]):
                r = fr.row_of(sid)
                if r is not None:
                    col[r] = lab
            fr.df[MY_LABEL] = col
        except Exception:
            pass                          # a broken file must not block the app

    def set_user_label(self, model: str, rows: list[int],
                       label: str | None) -> tuple[int, int]:
        """Label ``rows`` (``None`` / "" = remove their label; rows=None =
        remove all). → (labels revision, number of labelled vectors)."""
        with self._lock:
            fr = self.frame(model)
            col = (fr.df[MY_LABEL].to_numpy(dtype=object).copy()
                   if MY_LABEL in fr.df.columns
                   else np.full(fr.n, UNLABELLED, dtype=object))
            name = (str(label).strip() or UNLABELLED) if label else UNLABELLED
            if rows is None:
                col[:] = UNLABELLED
            else:
                idx = [int(r) for r in rows if 0 <= int(r) < fr.n]
                col[idx] = name[:60]
            fr.df[MY_LABEL] = col
            self._revs[model] = fr.labels_rev = self._revs.get(model, 0) + 1
            n = int((col != UNLABELLED).sum())
            f = self._labels_file(model)
            if f is not None:
                try:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    keep = np.flatnonzero(col != UNLABELLED)
                    pd.DataFrame({"session_id": [fr.sids[i] for i in keep],
                                  MY_LABEL: col[keep]}).to_csv(f, index=False)
                except OSError:
                    pass
            return fr.labels_rev, n

    def labels_rev(self, model: str) -> int:
        return self._revs.get(model, 0)

    def label_columns(self, model: str) -> list[str]:
        """Columns usable as class labels: categorical-ish, 2..200 values."""
        fr = self.frame(model)
        out = []
        for c in fr.df.columns:
            if c in ("session_id", "model_name"):
                continue
            s = fr.df[c]
            if pd.api.types.is_float_dtype(s):
                continue
            k = s.nunique(dropna=True)
            if 2 <= k <= min(MAX_LABEL_CARDINALITY, max(2, fr.n // 2)):
                out.append(c)
        return out

    def default_labels(self, model: str) -> list[str]:
        cols = self.label_columns(model)
        if "device_id" in cols:
            return ["device_id"]
        return cols[:1]

    # ── audio ───────────────────────────────────────────────────────
    @property
    def has_audio(self) -> bool:
        return self._wav_dir is not None and self._wav_dir.exists()

    def wav_path(self, sid: str) -> Path | None:
        """Resolve ``<wav_dir>/**/<sid>.wav``. Index built once (wavs sit in
        per-language subfolders, a flat ``wav_dir/sid.wav`` join misses)."""
        if not self.has_audio:
            return None
        if self._wav_index is None:
            with self._lock:
                if self._wav_index is None:
                    idx: dict[str, Path] = {}
                    for root, _dirs, files in os.walk(self._wav_dir):
                        for f in files:
                            if f.endswith(".wav"):
                                idx.setdefault(f[:-4], Path(root) / f)
                    self._wav_index = idx
        return self._wav_index.get(str(sid))
