"""Real data loader for the Vocal Embedding Dashboard.

Loads multi-language CSVs from Data/dfs/, joins with embeddings
from Data/embeddings/{model}.parquet, and returns a DataFrame
shaped for the Gui (must have: session_id, speaker_id, device_id,
model_name, centroid).

Single entry point: ``load_real(model_name)`` → pd.DataFrame.

Embeddings parquet schema:
    session_id  (str)
    embedding   (list[float] or np.ndarray — any 1-D flat array)

UMAP projections are cached to:
    Data/embeddings/cache/{model_name}_umap.parquet
    schema: session_id, umap_x, umap_y

Atomic file writes: .tmp + rename.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── path resolution ─────────────────────────────────────────────────────────
# CSV metadata + raw wavs live under the project-wide Data/ tree (source
# data, shared with other tools). Embedding cache + UMAP cache live
# under THIS visualization project so cache is portable / scoped to the
# dashboard. Override either via env var.
#
# Layout:
#   Research/Projects/
#     Data/Audio/{dfs, wavs}                 ← source CSVs + audio
#     Visualizations/vectors_dashboard/
#       data/embeddings/                      ← parquet cache (here)
#       data/embeddings/cache/                ← UMAP cache
_THIS = Path(__file__).resolve().parent
PROJECTS = _THIS.parent.parent.parent  # Research/Projects/

# Source data — try a few common layouts, take first existing.
_AUDIO_DATA_CANDIDATES = [
    PROJECTS / "RD" / "Audio_and_Acoustic" / "Data" / "Audio",  # current location
    PROJECTS / "Audio" / "Data",          # legacy
]
AUDIO_DATA = next((p for p in _AUDIO_DATA_CANDIDATES if p.exists()),
                  _AUDIO_DATA_CANDIDATES[0])
DFS_DIR = AUDIO_DATA / "dfs"
WAVS_DIR = AUDIO_DATA / "wavs"

# Embedding cache → INSIDE the visualization project (not Audio/Data).
EMB_DIR = Path(os.environ.get(
    "VECTORS_DASHBOARD_EMB_DIR",
    str(_THIS / "embeddings"),
))
EMB_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = EMB_DIR / "cache"

# Auto-import legacy parquets from old Data/Audio/embeddings location.
# One-time migration: symlink each legacy .parquet into EMB_DIR if not
# already present. Keeps old files in place; safe to delete originals.
_LEGACY_EMB = AUDIO_DATA / "embeddings"
if _LEGACY_EMB.exists() and _LEGACY_EMB != EMB_DIR:
    for _p in _LEGACY_EMB.glob("*.parquet"):
        _link = EMB_DIR / _p.name
        if not _link.exists():
            try:
                _link.symlink_to(_p.resolve())
            except OSError:
                pass

_CSV_FILES = [
    "arabic.csv",
    "farsi.csv",
    "turkish.csv",
    "english.csv",
    "simulator.csv",
    "other.csv",
]

# Columns to keep from the CSV (avoids huge stt/description text in memory)
_KEEP_COLS = [
    "session_id", "language", "dialect", "speaker_id", "gender",
    "age_bucket", "recording_type", "noise_level", "distance_bucket",
    "vocal_effort", "is_simulated", "snr_estimate_db", "duration_sec",
    "device_id", "source",
]


# ── multi-CSV loader ─────────────────────────────────────────────────────────

def load_metadata() -> pd.DataFrame:
    """Load and concatenate all per-language CSVs.

    Returns a DataFrame with _KEEP_COLS columns (subset that exists) plus
    a ``file_language`` column (arabic / farsi / etc.) for the UI language
    filter.  Always returns at least an empty DataFrame with the right cols.
    """
    frames: list[pd.DataFrame] = []
    for fname in _CSV_FILES:
        p = DFS_DIR / fname
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False, usecols=lambda c: c in _KEEP_COLS)
        lang_tag = fname.replace(".csv", "")
        df["file_language"] = lang_tag
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=_KEEP_COLS + ["file_language"])

    out = pd.concat(frames, ignore_index=True)

    # Normalise is_simulated to bool
    if "is_simulated" in out.columns:
        out["is_simulated"] = out["is_simulated"].fillna(False).astype(bool)

    # Fill categorical NAs with "unknown" so filters work cleanly
    for col in ["dialect", "gender", "age_bucket", "recording_type",
                "noise_level", "distance_bucket", "vocal_effort"]:
        if col in out.columns:
            out[col] = out[col].fillna("unknown").astype(str)

    return out


# ── embedding loader ─────────────────────────────────────────────────────────

def available_models() -> list[str]:
    """Return model names that have a .parquet embedding file ready."""
    if not EMB_DIR.exists():
        return []
    return sorted(
        p.stem for p in EMB_DIR.glob("*.parquet")
        if not p.stem.endswith("_umap")  # skip cache files
    )


def load_embeddings(model_name: str) -> Optional[pd.DataFrame]:
    """Load embeddings parquet for *model_name*.

    Returns DataFrame(session_id, embedding) or None if not found.
    """
    p = EMB_DIR / f"{model_name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "session_id" not in df.columns or "embedding" not in df.columns:
        return None
    return df[["session_id", "embedding"]]


# ── UMAP cache ────────────────────────────────────────────────────────────────

def _umap_cache_path(model_name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{model_name}_umap.parquet"


def load_umap_cache(model_name: str) -> Optional[pd.DataFrame]:
    """Return cached UMAP projection (session_id, umap_x, umap_y) or None."""
    p = _umap_cache_path(model_name)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def save_umap_cache(model_name: str, df: pd.DataFrame) -> None:
    """Atomic write: session_id, umap_x, umap_y."""
    p = _umap_cache_path(model_name)
    tmp = p.with_suffix(".tmp.parquet")
    df[["session_id", "umap_x", "umap_y"]].to_parquet(tmp, index=False)
    tmp.replace(p)


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 15,
                 min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """Compute 2-D UMAP projection. Returns (N, 2) float32 array."""
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=min_dist, random_state=random_state,
        )
        return reducer.fit_transform(embeddings).astype(np.float32)
    except ImportError:
        # Fall back to PCA if umap-learn not installed
        from sklearn.decomposition import PCA
        proj = PCA(n_components=2, random_state=random_state).fit_transform(embeddings)
        return proj.astype(np.float32)


# ── main entry point ──────────────────────────────────────────────────────────

def load_real(model_name: str) -> pd.DataFrame:
    """Return a DataFrame ready for Gui(df=...).

    Columns guaranteed: session_id, speaker_id, device_id, model_name,
    centroid (shape (1, D) ndarray), language, dialect, gender,
    age_bucket, recording_type, noise_level, distance_bucket,
    vocal_effort, is_simulated, snr_estimate_db, duration_sec,
    file_language.

    Rows: only sessions that have an embedding for *model_name*.
    If no embedding parquet exists, raises FileNotFoundError with instructions.
    """
    emb_df = load_embeddings(model_name)
    if emb_df is None:
        raise FileNotFoundError(
            f"No embedding parquet found for model '{model_name}'.\n"
            f"Expected: {EMB_DIR / model_name}.parquet\n"
            f"Run: python Audio/code/extract_embeddings.py --model {model_name}"
        )

    meta = load_metadata()

    # Join on session_id
    merged = emb_df.merge(meta, on="session_id", how="inner")
    if merged.empty:
        raise ValueError(
            f"Embedding parquet for '{model_name}' has no session_ids "
            f"matching the CSVs in {DFS_DIR}."
        )

    # Convert embedding column to (1, D) numpy arrays for centroid column
    def _to_centroid(v):
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        return arr.reshape(1, -1)

    merged["centroid"] = merged["embedding"].map(_to_centroid)
    merged["model_name"] = model_name

    # Ensure speaker_id exists (Gui requires it via _normalize_columns)
    if "speaker_id" not in merged.columns:
        merged["speaker_id"] = "unknown"

    # Ensure device_id exists
    if "device_id" not in merged.columns:
        merged["device_id"] = "unknown"

    drop_cols = ["embedding"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    return merged.reset_index(drop=True)


# ── wav path helper ───────────────────────────────────────────────────────────

def wav_path(session_id: str) -> Optional[Path]:
    """Return the wav path for a session_id, or None if not found.

    Wavs live in per-language subfolders (``wavs/<language>/<id>.wav``);
    the flat join alone never matched."""
    p = WAVS_DIR / f"{session_id}.wav"
    if p.exists():
        return p
    return next(WAVS_DIR.glob(f"*/{session_id}.wav"), None)


# ── stratified embedding extractor (for generating sample embeddings) ─────────

def extract_sample_embeddings(
    model_name: str,
    n_samples: int = 1000,
    device: str = "cpu",
) -> None:
    """Extract embeddings for a stratified sample (1 per speaker) and save parquet.

    Used to bootstrap the dashboard when no embeddings exist yet.
    Loads the SR model via RD/Audio_and_Acoustic/Models/sr/{model_name}/load.py.
    """
    import importlib.util
    from pathlib import Path as _P

    model_dir = PROJECTS / "RD" / "Audio_and_Acoustic" / "Weights" / "sr" / model_name
    load_py = model_dir / "load.py"
    if not load_py.exists():
        raise FileNotFoundError(f"load.py not found at {load_py}")

    spec = importlib.util.spec_from_file_location("_sr_load", str(load_py))
    sr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr_module)

    meta = load_metadata()
    # Stratified sample: 1 per unique speaker (alphabetically stable)
    sampled = (
        meta[meta["session_id"].notna()]
        .drop_duplicates(subset=["speaker_id"])
        .head(n_samples)
    )

    print(f"Loading SR model '{model_name}' on {device} …")
    model = sr_module.load_model(device=device)

    rows: list[dict] = []
    missing = 0
    t0 = time.time()
    for i, row in enumerate(sampled.itertuples()):
        wav_p = wav_path(row.session_id)
        if wav_p is None:
            missing += 1
            continue
        try:
            emb = sr_module.embed_wav(str(wav_p), model, device=device)
            emb_np = emb.detach().cpu().numpy().reshape(-1).tolist()
            rows.append({"session_id": row.session_id, "embedding": emb_np})
        except Exception as exc:
            print(f"  [warn] {row.session_id}: {exc}")
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(sampled)} ({elapsed:.1f}s) missing_wavs={missing}")

    if not rows:
        raise RuntimeError("No embeddings extracted — check wav paths.")

    out_df = pd.DataFrame(rows)
    out_path = EMB_DIR / f"{model_name}.parquet"
    tmp = out_path.with_suffix(".tmp.parquet")
    out_df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    print(f"Saved {len(out_df)} embeddings → {out_path}")
    print(f"Missing wavs: {missing}/{len(sampled)}")
