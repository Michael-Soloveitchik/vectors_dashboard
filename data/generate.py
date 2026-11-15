"""Synthetic demo data generator for the Vocal Embedding Dashboard.

Single entry point: ``load_and_enrich()`` → (df, params).

All tunables live in ``generation.json`` (same directory). The module
REFUSES to run if that file is missing — no hardcoded defaults.

The generator builds per-(voice, model) Gaussian clusters (or K-component
GMMs when ``sample_from_mixture_prob`` fires) in R^dim, optionally
L2-normalises the samples, and stores each vector as a (1, dim) ndarray
in the ``centroid`` column of the returned DataFrame.

``plot_dashboard(df, params)`` renders a 2×3 PCA diagnostic grid so the
notebook user can sanity-check the data before launching the Dash app.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
CONFIG_PATH = DATA_DIR / "generation.json"


# ── config loading ──────────────────────────────────────────────────

def _strip_comments(text: str) -> str:
    """Remove ``// ...`` line comments so the JSON parser doesn't choke."""
    return re.sub(r"//[^\n]*", "", text)


def load_config(path: Path | None = None) -> dict:
    p = path or CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"generation.json not found at {p}. "
            "The generator refuses to run without its config."
        )
    return json.loads(_strip_comments(p.read_text()))


# ── helpers ─────────────────────────────────────────────────────────

def parse_dim(model_name: str) -> int:
    """Extract the embedding dimension from a model name.

    Convention: everything after the LAST ``_`` is parsed as int.
    ``"PI_2_256"`` → 256,  ``"S2_1024"`` → 1024.
    """
    return int(model_name.rsplit("_", 1)[-1])


def build_model_names(cfg: dict) -> list[str]:
    """Cartesian product of base models × extension suffixes."""
    return [f"{m}{e}" for m in cfg["models"] for e in cfg["extensions"]]


# ── topology ────────────────────────────────────────────────────────

@dataclass
class VoiceAssignment:
    voice_id: str
    device_id: str


def build_topology(cfg: dict, rng: np.random.Generator) -> list[VoiceAssignment]:
    """Create (voice_id, device_id) pairs from the topology config."""
    topo = cfg["topology"]
    assignments: list[VoiceAssignment] = []
    voice_counter = 0

    # Single-voice devices
    for dev_idx in topo["single_voice_device_indices"]:
        assignments.append(VoiceAssignment(
            voice_id=f"voice_{voice_counter:02d}",
            device_id=f"device_{dev_idx:02d}",
        ))
        voice_counter += 1

    # Multi-voice devices
    lo, hi = topo["voices_per_multi_device"]
    for dev_idx in topo["multi_voice_device_indices"]:
        n_voices = int(rng.integers(lo, hi + 1))
        for _ in range(n_voices):
            assignments.append(VoiceAssignment(
                voice_id=f"voice_{voice_counter:02d}",
                device_id=f"device_{dev_idx:02d}",
            ))
            voice_counter += 1

    return assignments


# ── gaussian cluster sampling ───────────────────────────────────────

@dataclass
class GaussianParams:
    """Describes one (voice, model) cluster — single or mixture."""
    voice_id: str
    device_id: str
    model_name: str
    dim: int
    centroid: np.ndarray               # (dim,) — weighted mean of components
    is_mixture: bool = False
    component_means: list[np.ndarray] = field(default_factory=list)
    component_weights: np.ndarray | None = None


def build_voice_pools(
    cfg: dict,
    rng: np.random.Generator,
    model_names: list[str],
    topology: list[VoiceAssignment],
) -> dict[tuple[int, str], np.ndarray]:
    """Per-(shared_group, model) pool of voice directions.

    When enabled via ``device_geometry.voice_pool_size > 0``, voices
    across devices in the SAME shared group all sample their base
    direction from a small pool of K directions. This forces voice
    overlap across devices -> their class means stay similar -> per-
    device AUC drops substantially (target: 0.7-0.85 range).

    Returns {} when disabled.
    """
    dg = cfg.get("device_geometry") or {}
    pool_size = int(dg.get("voice_pool_size", 0))
    if pool_size <= 0:
        return {}
    mean_std = cfg["gaussians"]["between_cluster_mean_std"]
    n_groups = len(dg.get("shared_direction_groups", []) or [])
    out: dict[tuple[int, str], np.ndarray] = {}
    for gid in range(n_groups):
        for model_name in model_names:
            dim = parse_dim(model_name)
            out[(gid, model_name)] = rng.normal(
                scale=mean_std, size=(pool_size, dim)
            )
    return out


def build_device_means(
    cfg: dict,
    rng: np.random.Generator,
    model_names: list[str],
    topology: list[VoiceAssignment],
) -> dict[tuple[str, str], np.ndarray]:
    """Per-(device, model) base direction on the sphere.

    Devices listed in the same entry of
    ``device_geometry.shared_direction_groups`` share a single base
    direction — so after L2-normalize, their voices' sessions overlap
    in cosine space. Devices not in any group get an independent
    direction and stay separable.

    Returns an empty dict when ``device_geometry`` is absent from the
    config (preserves the old per-voice-independent behavior).
    """
    dg = cfg.get("device_geometry")
    if not dg:
        return {}

    shared_groups = dg.get("shared_direction_groups", []) or []
    group_dev_std = float(dg.get("shared_group_device_offset_std", 0.0))
    mean_std = cfg["gaussians"]["between_cluster_mean_std"]

    device_to_group: dict[str, int] = {}
    for gid, group in enumerate(shared_groups):
        for idx in group:
            device_to_group[f"device_{idx:02d}"] = gid

    all_devices = sorted({v.device_id for v in topology})
    out: dict[tuple[str, str], np.ndarray] = {}
    for model_name in model_names:
        dim = parse_dim(model_name)
        group_means: dict[int, np.ndarray] = {}
        for dev in all_devices:
            gid = device_to_group.get(dev)
            if gid is None:
                out[(dev, model_name)] = rng.normal(scale=mean_std, size=dim)
            else:
                if gid not in group_means:
                    group_means[gid] = rng.normal(scale=mean_std, size=dim)
                # Each device sits near its group direction but not ON it —
                # a small offset keeps them partially separable in cosine
                # space while still overlapping substantially.
                if group_dev_std > 0.0:
                    off = rng.normal(scale=mean_std * group_dev_std, size=dim)
                    out[(dev, model_name)] = group_means[gid] + off
                else:
                    out[(dev, model_name)] = group_means[gid]
    return out


def _draw_cluster(
    voice: VoiceAssignment,
    model_name: str,
    dim: int,
    cfg: dict,
    rng: np.random.Generator,
    device_mean: np.ndarray | None = None,
    voice_pool: np.ndarray | None = None,
) -> GaussianParams:
    """Draw a cluster's parameters — either single Gaussian or K-GMM.

    ``voice_pool`` (optional): shape (K, dim). When provided, the
    voice's base mean is ``pool[random_k] + N(0, mean_std *
    voice_offset_std * I_dim)``. Devices sharing the same pool draw
    voices from the same K directions, forcing cross-device overlap.

    ``device_mean`` (optional): used when ``voice_pool`` is None — the
    voice's base mean sits near a per-device shared direction.

    When both are None, falls back to the original independent draw.
    """
    gcfg = cfg["gaussians"]
    mean_std = gcfg["between_cluster_mean_std"]
    dg = cfg.get("device_geometry") or {}
    vo_std = float(dg.get("voice_offset_std", 1.0))

    if voice_pool is not None and len(voice_pool) > 0:
        k = int(rng.integers(0, len(voice_pool)))
        offset = rng.normal(scale=mean_std * vo_std, size=dim).astype(np.float64)
        base_mean = voice_pool[k].astype(np.float64) + offset
    elif device_mean is not None:
        offset = rng.normal(scale=mean_std * vo_std, size=dim).astype(np.float64)
        base_mean = device_mean.astype(np.float64) + offset
    else:
        base_mean = rng.normal(scale=mean_std, size=dim).astype(np.float64)

    is_mixture = rng.random() < gcfg["sample_from_mixture_prob"]

    if not is_mixture:
        return GaussianParams(
            voice_id=voice.voice_id,
            device_id=voice.device_id,
            model_name=model_name,
            dim=dim,
            centroid=base_mean,
        )

    K = gcfg["n_mixture_components"]
    sep = gcfg["mixture_component_separation"]
    alpha = gcfg["mixture_weight_dirichlet_alpha"]

    # Component means: first = base_mean, rest offset by sep * mean_std
    comp_means = [base_mean]
    for _ in range(K - 1):
        offset = rng.normal(scale=mean_std * sep, size=dim)
        comp_means.append(base_mean + offset)

    # Dirichlet weights
    weights = rng.dirichlet([alpha] * K)

    # Weighted centroid
    centroid = sum(w * m for w, m in zip(weights, comp_means))

    return GaussianParams(
        voice_id=voice.voice_id,
        device_id=voice.device_id,
        model_name=model_name,
        dim=dim,
        centroid=centroid,
        is_mixture=True,
        component_means=comp_means,
        component_weights=weights,
    )


def _sample_from_cluster(
    gp: GaussianParams,
    n_samples: int,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_samples vectors from a cluster (single or mixture)."""
    gcfg = cfg["gaussians"]
    lo, hi = gcfg["within_cluster_std_range"]
    cluster_std = rng.uniform(lo, hi)

    if not gp.is_mixture:
        return rng.normal(loc=gp.centroid, scale=cluster_std,
                          size=(n_samples, gp.dim))

    # Mixture: assign each sample to a component, then draw from that component
    K = len(gp.component_means)
    assignments = rng.choice(K, size=n_samples, p=gp.component_weights)
    samples = np.empty((n_samples, gp.dim), dtype=np.float64)
    for i in range(n_samples):
        k = assignments[i]
        samples[i] = rng.normal(loc=gp.component_means[k], scale=cluster_std,
                                size=gp.dim)
    return samples


# ── post-processing ─────────────────────────────────────────────────

def _post_process(vectors: np.ndarray, cfg: dict) -> np.ndarray:
    pp = cfg["post_processing"]
    dtype = np.dtype(pp["dtype"])
    if pp["l2_normalize"]:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, pp["l2_eps"])
    return vectors.astype(dtype)


# ── main generator ──────────────────────────────────────────────────

def load_and_enrich(
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate the demo DataFrame from ``generation.json``.

    Returns
    -------
    df : pd.DataFrame
        Columns: voice_id, session_id, device_id, model_name, centroid.
        ``centroid`` holds (1, D) numpy arrays.
    params : dict
        The parsed generation config (for reproducibility logging).
    """
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["seed"])

    model_names = build_model_names(cfg)
    topology = build_topology(cfg, rng)
    device_means = build_device_means(cfg, rng, model_names, topology)
    voice_pools = build_voice_pools(cfg, rng, model_names, topology)

    # device -> shared-group-id (for voice-pool lookup)
    dg = cfg.get("device_geometry") or {}
    device_to_group: dict[str, int] = {}
    for gid, group in enumerate(dg.get("shared_direction_groups", []) or []):
        for idx in group:
            device_to_group[f"device_{idx:02d}"] = gid

    sess_lo, sess_hi = cfg["sessions_per_voice_model"]
    pp = cfg["post_processing"]
    dtype = np.dtype(pp["dtype"])

    rows: list[dict] = []
    all_clusters: list[GaussianParams] = []

    for voice in topology:
        for model_name in model_names:
            dim = parse_dim(model_name)
            dmean = device_means.get((voice.device_id, model_name))
            gid = device_to_group.get(voice.device_id)
            pool = voice_pools.get((gid, model_name)) if gid is not None else None
            gp = _draw_cluster(
                voice, model_name, dim, cfg, rng,
                device_mean=dmean, voice_pool=pool,
            )
            all_clusters.append(gp)

            n_sessions = int(rng.integers(sess_lo, sess_hi + 1))
            samples = _sample_from_cluster(gp, n_sessions, cfg, rng)
            samples = _post_process(samples, cfg)

            for s_idx in range(n_sessions):
                session_id = (f"{voice.voice_id}_{model_name}"
                              f"_s{s_idx:03d}")
                rows.append({
                    "voice_id": voice.voice_id,
                    "session_id": session_id,
                    "device_id": voice.device_id,
                    "model_name": model_name,
                    "centroid": samples[s_idx].reshape(1, -1),
                })

    df = pd.DataFrame(rows)

    # Persist the flat CSV (without centroid vectors) for quick inspection
    csv_dir = DATA_DIR / cfg["paths"]["dfs_dir"]
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / cfg["paths"]["csv_name"]
    df[["voice_id", "session_id", "device_id", "model_name"]].to_csv(
        csv_path, index=False,
    )

    # Attach cluster metadata for downstream use (plot_dashboard, tests)
    cfg["_clusters"] = all_clusters
    cfg["_topology"] = topology
    cfg["_model_names"] = model_names

    return df, cfg


# ── diagnostic plot ─────────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, params: dict) -> None:
    """Render a 2×3 PCA scatter grid — one subplot per model_name.

    Each subplot shows the 2D PCA projection of all embeddings for that
    model, coloured by voice_id. Ground-truth centroids (from the
    generative Gaussians) are overlaid as large X markers.

    The figure is saved to ``data/{dashboard_name}`` and displayed
    inline in Jupyter.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    model_names = params.get("_model_names") or sorted(df["model_name"].unique())
    pcfg = params["plot"]
    figsize = tuple(pcfg["figsize"])
    n_models = len(model_names)
    ncols = min(3, n_models)
    nrows = max(1, (n_models + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    # Colour map: one colour per voice
    all_voices = sorted(df["voice_id"].unique())
    cmap = plt.cm.get_cmap("tab20", len(all_voices))
    voice_to_color = {v: cmap(i) for i, v in enumerate(all_voices)}

    # Cluster metadata (if available)
    clusters = params.get("_clusters", [])
    cluster_lookup: dict[tuple[str, str], GaussianParams] = {}
    for gp in clusters:
        cluster_lookup[(gp.voice_id, gp.model_name)] = gp

    for ax_idx, model_name in enumerate(model_names):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        sub = df[df["model_name"] == model_name].reset_index(drop=True)
        if sub.empty:
            ax.set_title(f"{model_name} (empty)")
            ax.axis("off")
            continue

        vectors = np.stack(
            [np.asarray(c).reshape(-1) for c in sub["centroid"]], axis=0,
        )
        dim = vectors.shape[1]

        pca = PCA(n_components=2, random_state=42)
        proj = pca.fit_transform(vectors)

        # Scatter samples coloured by voice
        for voice_id in all_voices:
            mask = (sub["voice_id"] == voice_id).values
            if not mask.any():
                continue
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                s=pcfg["sample_marker_size"],
                alpha=pcfg["sample_alpha"],
                color=voice_to_color[voice_id],
                label=voice_id,
                edgecolors="none",
            )

        # Overlay ground-truth centroids
        cent_list = []
        cent_voices = []
        for voice_id in all_voices:
            gp = cluster_lookup.get((voice_id, model_name))
            if gp is not None:
                cent_list.append(gp.centroid.reshape(-1)[:dim])
                cent_voices.append(voice_id)
        if cent_list:
            cent_arr = np.stack(cent_list, axis=0).astype(np.float64)
            # Project centroids through the SAME PCA
            cent_2d = pca.transform(cent_arr)
            for i, voice_id in enumerate(cent_voices):
                ax.scatter(
                    cent_2d[i, 0], cent_2d[i, 1],
                    s=pcfg["centroid_marker_size"],
                    marker="x",
                    color=voice_to_color[voice_id],
                    linewidths=2,
                    zorder=10,
                )

        ax.set_title(f"{model_name}  (n={len(sub)}, dim={dim})",
                     fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Hide unused axes
    for ax_idx in range(n_models, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].axis("off")

    # Shared legend (voices only, no duplicate entries)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(10, len(all_voices)), fontsize=8,
               frameon=True, framealpha=0.9)

    fig.suptitle(
        f"Vocal Embedding Dashboard — demo data "
        f"({len(df)} rows, {df['voice_id'].nunique()} voices, "
        f"{df['model_name'].nunique()} models)",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save to disk
    out_path = DATA_DIR / params["paths"]["dashboard_name"]
    fig.savefig(out_path, dpi=pcfg["dpi"], bbox_inches="tight")
    print(f"Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
    plt.show()
