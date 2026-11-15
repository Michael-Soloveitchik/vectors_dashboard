# speaker_model.py

## Purpose
Read-only registry that groups speaker embeddings by `model_name`. It wraps the enriched DataFrame returned by `generate.load_and_enrich()` and provides thin accessor methods for listing models, listing voices, filtering by model, and stacking centroid vectors into a numpy array. It does NOT mutate the dataframe and does NOT perform embedding fusion.

## Motivation
The dashboard launcher notebook (`run.ipynb`) needs a clean interface to query which models and voices exist and to extract per-model embedding matrices for plotting and similarity computation. Rather than scattering DataFrame filter logic across notebook cells and GUI callbacks, this class centralises it in one place with column validation at construction time. Embedding-level fusion (`concat`, `reinforce`, `sum`, `mean`) is a separate concern that lives in the Audio project at `modules_code/models/model_fused.py:FusedSpeakerModel` and runs at extraction time, not on the visualisation side.

## Key Classes / Functions

### `SpeakerModel`
- **What it does:** Thin read-only wrapper around the enriched DataFrame. Validates required columns on construction and exposes query methods.

#### `__init__(self, df)`
- **What it does:** Validates that the DataFrame contains all required columns and stores it.
- **Args:**
  - `df` (pd.DataFrame): Must contain columns: `voice_id`, `session_id`, `device_id`, `model_name`, `centroid`.
- **Raises:** `ValueError` if any required column is missing.
- **Example call:**
  ```python
  sm = SpeakerModel(df)
  ```

#### `list_models(self)`
- **What it does:** Returns sorted unique model names from the DataFrame.
- **Returns:** `list[str]` -- e.g. `['PI_2_1024', 'PI_2_256', 'PI_3_1024', ...]`
- **Example call:**
  ```python
  sm.list_models()  # -> ['PI_2_1024', 'PI_2_256', 'PI_3_1024', 'PI_3_256', 'S2_1024', 'S2_256']
  ```

#### `list_voices(self, model_name=None)`
- **What it does:** Returns sorted unique voice IDs, optionally filtered to a single model.
- **Args:**
  - `model_name` (str | None): If provided, only voices present in that model's subset are returned.
- **Returns:** `list[str]` -- e.g. `['voice_00', 'voice_01', ...]`
- **Example call:**
  ```python
  sm.list_voices()                    # all voices across all models
  sm.list_voices("PI_2_256")          # voices that have PI_2_256 embeddings
  ```

#### `filter_model(self, model_name)`
- **What it does:** Returns a copy of the DataFrame filtered to a single model_name, with the index reset.
- **Args:**
  - `model_name` (str): The model to filter on.
- **Returns:** `pd.DataFrame` -- same columns as the original, filtered rows only.
- **Example call:**
  ```python
  sub = sm.filter_model("S2_1024")  # DataFrame with only S2_1024 rows
  ```

#### `get_vectors(self, model_name)`
- **What it does:** Stacks all centroid vectors for a given model into a single 2D numpy array.
- **Args:**
  - `model_name` (str): The model to extract vectors for.
- **Returns:** `np.ndarray` shape `(N, D)` where N is the number of rows for that model and D is the embedding dimension (e.g. 256 or 1024).
- **Example call:**
  ```python
  vecs = sm.get_vectors("PI_2_256")  # shape (N, 256)
  ```

## Inputs & Outputs

**Inputs:**
- `pd.DataFrame` with required columns: `voice_id` (str), `session_id` (str), `device_id` (str), `model_name` (str), `centroid` (np.ndarray of shape `(1, D)`).

**Outputs:**
- `list_models()` -> `list[str]`
- `list_voices()` -> `list[str]`
- `filter_model()` -> `pd.DataFrame` (same schema, filtered rows)
- `get_vectors()` -> `np.ndarray` shape `(N, D)` where each row is a flattened centroid vector

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal construction with valid df | Stores df, no error | All 5 required columns present |
| Missing required column | `ValueError` listing which columns are absent | Checked at `__init__` time |
| `list_voices(model_name="nonexistent")` | Empty list `[]` | Filter returns empty subset; `.unique()` on empty is `[]` |
| `filter_model("nonexistent")` | Empty DataFrame with correct columns | `.reset_index(drop=True)` on zero rows |
| `get_vectors("nonexistent")` | `ValueError` from `np.stack` on empty sequence | No rows to stack; not explicitly handled |
| Centroid stored as `(1, D)` | `get_vectors` reshapes each to `(D,)` via `.reshape(-1)` | Flat vectors are stacked into `(N, D)` |

## Where This Is Used

- `run.ipynb` (project root) -- constructs `SpeakerModel(df)` after `generate.load_and_enrich()`, uses `sm.list_models()`, `sm.df`, and passes filtered data to `Gui`.

## Known Gotchas & Non-Obvious Behavior

- **Read-only by convention, not enforcement:** The `df` attribute is stored directly (not copied). External code could mutate `sm.df` and the `SpeakerModel` instance would reflect those changes. No defensive copy is made.
- **`get_vectors` reshapes centroids:** Each centroid in the DataFrame is `(1, D)` but `get_vectors` flattens them to `(D,)` before stacking. Code that expects the `(1, D)` shape from the raw DataFrame will get a different shape from `get_vectors`.
- **No fusion logic:** Despite the name "SpeakerModel", this class performs no model inference or embedding fusion. It is purely a data accessor. Fusion lives in a separate project (`modules_code/models/model_fused.py`).
- **Class attribute `REQUIRED`:** The required column list is a class-level constant, not instance-level. Subclasses or monkey-patching could change it for all instances.

## Dependencies

**External:**
- `numpy` -- array stacking and reshaping in `get_vectors`
- `pandas` -- DataFrame filtering and column access

**Internal:**
- None (standalone module; consumes whatever DataFrame is passed in)
