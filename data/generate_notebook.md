# generate.ipynb

## Purpose
Standalone data-generation and inspection notebook. It calls `generate.load_and_enrich()` to build the synthetic speaker-embedding DataFrame and then renders the 2x3 PCA diagnostic grid via `generate.plot_dashboard()`. This notebook exists so a developer can inspect the generated data and its cluster structure without launching the full Dash dashboard.

## Motivation
Separating data generation/inspection from the dashboard launcher (`run.ipynb`) keeps concerns clean: `generate.ipynb` is for verifying that the synthetic data looks correct (cluster separation, voice count, dimension sanity), while `run.ipynb` is for launching the interactive GUI. This notebook is the first thing to run after changing `generation.json` to confirm the new parameters produce sensible clusters.

## Cells

### Cell 1 -- Imports + sys.path
- **What it does:** Loads the `autoreload` extension, adds the project root to `sys.path`, and imports the `generate` module.
- **Key imports:**
  - `generate` (from `data/`) -- contains `load_and_enrich()` and `plot_dashboard()`

### Cell 2 -- Generate + Plot
- **What it does:** Calls `generate.load_and_enrich()` to produce the DataFrame and config dict, prints the first 5 rows via `df.head()`, then calls `generate.plot_dashboard(df, params)` to render and save the 2x3 PCA scatter grid.
- **Outputs:**
  - Printed DataFrame head (voice_id, session_id, device_id, model_name, centroid columns)
  - Inline matplotlib figure: 2x3 grid of PCA scatter plots, one per model_name, coloured by voice_id, with ground-truth centroids as X markers
  - Saved file: `data/dashboard.png`

## Inputs & Outputs

**Inputs:**
- `data/generation.json` -- consumed indirectly via `generate.load_and_enrich()`

**Outputs:**
- `data/dfs/df.csv` -- flat CSV (side effect of `load_and_enrich()`)
- `data/dashboard.png` -- PCA diagnostic grid (side effect of `plot_dashboard()`)
- Inline display of the DataFrame head and the PCA figure

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal execution | DataFrame head printed, 2x3 PCA grid displayed and saved to `dashboard.png` | ~882 rows with default config |
| `generation.json` missing | `FileNotFoundError` in cell 2 | Generator refuses to run without config |
| Config seed changed | Different cluster layout in the PCA plot | All downstream draws change |
| Very high `within_cluster_std_range` | Overlapping clusters in PCA -- hard to distinguish voices | Indicates config needs tighter clusters |

## Where This Is Used
This is a standalone inspection notebook. It is not imported by any other module. Run it after editing `generation.json` to verify the data distribution before launching the dashboard.

## Known Gotchas & Non-Obvious Behavior

- **Same module, different notebook:** Both `generate.ipynb` and `run.ipynb` import and call `generate.load_and_enrich()`. If both are running in the same Jupyter server, they share the same `generation.json` and will produce identical data (same seed). They do not interfere with each other.
- **`%autoreload 2`:** Changes to `generate.py` take effect on re-run without a kernel restart.
- **`plot_dashboard` lazy imports:** `matplotlib` and `scikit-learn` are only imported when `plot_dashboard()` is called, not at module load time. If either is missing, the error appears here, not at import.

## Dependencies

**External:**
- `numpy`, `pandas` -- via `generate` module
- `matplotlib` -- PCA scatter grid rendering (lazy import inside `plot_dashboard`)
- `scikit-learn` -- `PCA(n_components=2)` for 2D projections (lazy import inside `plot_dashboard`)

**Internal:**
- `data/generate.py` -- `load_and_enrich()`, `plot_dashboard()`
- `data/generation.json` -- required config file
