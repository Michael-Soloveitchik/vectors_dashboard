"""WavLens inside the inspector.

``/_wavlens?model=<m>&rows=<i>[,<j>]`` returns WavLens' standalone HTML
(waveform ↔ mel-spectrogram, zoom, playhead, per-row mute, mix row) for
one or two sessions; the inspector embeds it in an iframe. Two rows are
zero-padded to a common length because WavLens shares one time line —
mute either with its checkbox to A/B them.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from .engine import DataHub

_SIBLING = Path(__file__).resolve().parents[3] / "WavLens"
MAX_SECONDS = 30.0


def _wavlens():
    try:
        import wavlens
        return wavlens
    except ImportError:
        if (_SIBLING / "wavlens" / "__init__.py").exists():
            sys.path.append(str(_SIBLING))
            import wavlens
            return wavlens
        raise


def _page(msg: str) -> str:
    return ("<!doctype html><body style='font:12px sans-serif;color:#667;"
            f"padding:12px;margin:0'>{msg}</body>")


def _load(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return x.mean(axis=1)[: int(MAX_SECONDS * sr)], int(sr)


def build_html(hub: DataHub, model: str, rows: tuple[int, ...],
               names: tuple[str, ...]) -> str:
    fr = hub.frame(model)
    clips = {}
    for r, name in zip(rows, names):
        p = hub.wav_path(fr.sids[r])
        if p is not None:
            clips[name] = _load(p)
    if not clips:
        return _page("no audio file for this vector")
    dur = max(x.size / sr for x, sr in clips.values())
    audio = {}
    for name, (x, sr) in clips.items():
        n = int(round(dur * sr))
        audio[name] = (np.pad(x, (0, max(0, n - x.size)))[:n], sr)
    try:
        wl = _wavlens()
    except ImportError:
        return _page("WavLens is not importable — pip install -e "
                     "Projects/Visualizations/WavLens")
    sr_max = max(sr for _, sr in audio.values())
    w = wl.show(audio=audio, height=84, sample_rate=sr_max,
                fmax=min(8000, sr_max // 2))
    return w.html(title="wavlens")


def register_routes(server, hub: DataHub) -> None:
    from flask import Response, request

    @lru_cache(maxsize=24)
    def _cached(model: str, rows: tuple[int, ...], names: tuple[str, ...]):
        return build_html(hub, model, rows, names)

    @server.route("/_wavlens")
    def _wavlens_route():
        try:
            model = request.args["model"]
            rows = tuple(int(v) for v in request.args["rows"].split(",") if v)
            names = tuple(request.args.get("names", "").split("|"))
            if len(names) != len(rows):
                names = tuple(f"audio_{i}" for i in range(len(rows)))
            html = _cached(model, rows[:2], names[:2])
        except Exception as exc:
            html = _page(f"audio unavailable: {type(exc).__name__}: {exc}")
        return Response(html, mimetype="text/html")
