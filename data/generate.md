# generate.py

## Purpose
Synthetic demo data generator for the Vocal Embedding Dashboard. It reads every tunable parameter from `generation.json` (same directory), builds per-(voice, model) Gaussian clusters (or K-component GMMs) in R^dim, optionally L2-normalises the samples, and returns a pandas DataFrame with one row per session embedding plus a diagnostic PCA plot. The single entry point is `load_and_enrich()`.

## Motivation
The dashboard needs realistic but fully reproducible speaker-embedding data to develop and demo against without requiring real audio extraction pipelines. Keeping all tunables in `generation.json` (and refusing to run without it) means no hardcoded defaults can silently change the data distribution. The generator also attaches cluster metadata so downstream diagnostic plots can overlay ground-truth centroids.

## Key Classes / Functions

### `load_config(path)`
- **What it does:** Reads and parses `generation.json`, stripping `// ...` line comments before JSON parsing.
- **Args:**
  - `path` (Path | None): Override config location. Defaults to `data/generation.json`.
- **Returns:** `dict` -- the parsed config.
- **Raises:** `FileNotFoundError` if the config file is missing.
- **Example call:**
  ```python
  cfg = load_config()
  cfg = load_config(Path("custom_generation.json"))
  ```

### `parse_dim(model_name)`
- **What it does:** Extracts the embedding dimension from a model name by parsing everything after the last `_` as an integer.
- **Args:**
  - `model_name` (str): e.g. `"PI_2_256"`, `"S2_1024"`.
- **Returns:** `int` -- the dimension (256, 1024, etc.).
- **Example call:**
  ```python
  parse_dim("PI_2_256")   # -> 256
  parse_dim("S2_1024")    # -> 1024
  ```

### `build_model_names(cfg)`
- **What it does:** Cartesian product of `cfg["models"]` x `cfg["extensions"]` to produce the full list of model names.
- **Args:**
  - `cfg` (dict): Parsed generation config.
- **Returns:** `list[str]` -- e.g. `["PI_2_256", "PI_2_1024", "PI_3_256", ...]`.
- **Example call:**
  ```python
  names = build_model_names(cfg)  # -> ['PI_2_256', 'PI_2_1024', 'PI_3_256', ...]
  ```

### `VoiceAssignment` (dataclass)
- **What it does:** Simple container holding a `voice_id` / `device_id` pair.
- **Fields:**
  - `voice_id` (str): e.g. `"voice_00"`
  - `device_id` (str): e.g. `"device_00"`

### `build_topology(cfg, rng)`
- **What it does:** Creates the list of (voice, device) assignments from the topology section of the config. Single-voice devices get exactly one voice; multi-voice devices draw a random count from the configured range.
- **Args:**
  - `cfg` (dict): Parsed generation config.
  - `rng` (np.random.Generator): Seeded RNG.
- **Returns:** `list[VoiceAssignment]`
- **Example call:**
  ```python
  rng = np.random.default_rng(42)
  topo = build_topology(cfg, rng)  # -> [VoiceAssignment('voice_00', 'device_00'), ...]
  ```

### `GaussianParams` (dataclass)
- **What it does:** Describes one (voice, model) cluster -- either a single Gaussian or a K-component GMM.
- **Fields:**
  - `voice_id` (str), `device_id` (str), `model_name` (str), `dim` (int)
  - `centroid` (np.ndarray): Shape `(dim,)`. For single Gaussians this is the mean; for mixtures it is the weighted mean of components.
  - `is_mixture` (bool): Whether this cluster uses multiple components.
  - `component_means` (list[np.ndarray]): Each shape `(dim,)`. Empty for single Gaussians.
  - `component_weights` (np.ndarray | None): Dirichlet-drawn weights summing to 1.0.

### `load_and_enrich(config_path)`
- **What it does:** Main entry point. Loads config, builds topology, generates all Gaussian clusters, samples embeddings, post-processes (L2-norm + dtype cast), and returns a DataFrame plus the enriched config dict.
- **Args:**
  - `config_path` (Path | None): Override config location.
- **Returns:** `tuple[pd.DataFrame, dict]`
  - `df` columns: `voice_id`, `session_id`, `device_id`, `model_name`, `centroid` (each centroid is a `(1, D)` ndarray).
  - `params` dict: The parsed config, enriched with `_clusters`, `_topology`, and `_model_names` keys for downstream use.
- **Side effects:** Writes a flat CSV (without centroid vectors) to `data/dfs/df.csv`.
- **Example call:**
  ```python
  df, params = load_and_enrich()
  df, params = load_and_enrich(Path("custom_generation.json"))
  ```

### `plot_dashboard(df, params)`
- **What it does:** Renders a 2x3 PCA scatter grid (one subplot per model_name), coloured by voice_id, with ground-truth centroids overlaid as large X markers. Saves the figure to `data/dashboard.png` and displays inline in Jupyter.
- **Args:**
  - `df` (pd.DataFrame): The DataFrame returned by `load_and_enrich()`.
  - `params` (dict): The enriched config dict returned by `load_and_enrich()`.
- **Returns:** None (displays plot and saves PNG).
- **Example call:**
  ```python
  df, params = load_and_enrich()
  plot_dashboard(df, params)
  ```

## Inputs & Outputs

**Inputs:**
- `data/generation.json` -- all tunables (seed, models, topology, Gaussian params, post-processing, paths, plot config). Required; generator refuses to run without it.

**Outputs:**
- `pd.DataFrame` with columns: `voice_id` (str), `session_id` (str), `device_id` (str), `model_name` (str), `centroid` (np.ndarray shape `(1, D)` where D is 256 or 1024).
- `data/dfs/df.csv` -- flat CSV with the four string columns (no centroid vectors).
- `data/dashboard.png` -- 2x3 PCA diagnostic grid (when `plot_dashboard` is called).

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal run with default config | ~882 rows, 9 voices, 6 models, all centroids L2-normalized | Exact count depends on RNG draws within configured ranges |
| `generation.json` missing | `FileNotFoundError` raised immediately | No fallback defaults; this is intentional |
| Seed changed in config | Completely different cluster positions and sample counts | Every downstream draw is affected |
| `l2_normalize` set to false | Raw Euclidean vectors; norms vary | Cosine-similarity assumptions in the dashboard may break |
| All voices on one device | Works fine; device_id column is just a label | No constraint on device uniqueness |
| Empty topology (no voices) | Returns an empty DataFrame with correct columns | No crash, but dashboard plot will show empty subplots |

## Where This Is Used

- `run.ipynb` (project root) -- calls `generate.load_and_enrich()` to build demo data, then passes the df to `SpeakerModel` and `Gui`.
- `data/generate.ipynb` -- calls `load_and_enrich()` and `plot_dashboard()` for standalone data inspection.
- `code/TESTS/tests/_helpers.py:70` -- imports `generate` to build test fixtures from the real generator.

## Known Gotchas & Non-Obvious Behavior

- **Comment stripping:** `generation.json` supports `// ...` line comments, which are stripped by `_strip_comments()` before JSON parsing. Standard JSON parsers will choke on the raw file.
- **Centroid shape is (1, D), not (D,):** Each `centroid` value in the DataFrame is reshaped to `(1, D)` at line 252. Downstream code must `.reshape(-1)` or index `[0]` to get a flat vector.
- **`_clusters` key on the config dict:** `load_and_enrich()` mutates the returned config dict by adding `_clusters`, `_topology`, and `_model_names` keys. These carry Python objects (dataclasses, numpy arrays) and are not JSON-serializable.
- **Inclusive integer ranges:** The config convention is `[lo, hi]` inclusive for integer ranges, implemented as `rng.integers(lo, hi + 1)`. This matches the docstring in `generation.json` but differs from Python's typical half-open convention.
- **Mixture centroid is a weighted mean:** For GMM clusters, `GaussianParams.centroid` is `sum(w_k * mean_k)`, not any single component mean. The individual component means live in `component_means`.
- **CSV lacks centroid vectors:** The CSV saved to `data/dfs/df.csv` intentionally omits the `centroid` column (numpy arrays don't serialize cleanly to CSV). The full data is only available via the returned DataFrame.

## Dependencies

**External:**
- `numpy` -- RNG, linear algebra, array operations
- `pandas` -- DataFrame construction and CSV export
- `matplotlib` (lazy import in `plot_dashboard` only) -- PCA scatter grid rendering
- `scikit-learn` (lazy import in `plot_dashboard` only) -- `PCA(n_components=2)` for 2D projections

**Internal:**
- `data/generation.json` -- required config file (no hardcoded defaults)
