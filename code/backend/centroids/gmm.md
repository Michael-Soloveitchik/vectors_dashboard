# gmm.py

## Purpose
Fits one Gaussian Mixture Model (GMM) per unique label in a dataset and returns the resulting centroids, covariances, and mixture weights in the original high-dimensional vector space. This is the core of the "find_centroids" feature (spec point 42), which lets users discover cluster structure within each label by specifying how many GMM components to extract per label.

## Motivation
Centroids are fitted in the raw high-dimensional embedding space rather than in the post-reduction 2D coordinates. This ensures that the centroids have stable, projection-independent semantics: their meaning does not change when the user switches between PCA, t-SNE, or UMAP. The GMM approach (vs. simple mean) captures multi-modal label distributions where a single centroid per label would be misleading.

## Key Classes / Functions

### `CentroidSet`
- **What it does:** Dataclass holding per-label GMM fit results (means, covariances, weights) plus the session IDs that contributed to each fit.
- **Fields:**
  - `centroids` (dict[str, np.ndarray]): label -> (k, D) array of GMM means.
  - `covariances` (dict[str, np.ndarray]): label -> (k, D, D) array of full covariance matrices.
  - `weights` (dict[str, np.ndarray]): label -> (k,) array of mixture weights.
  - `member_session_ids` (dict[str, list[str]]): label -> list of session_ids used for fitting.
  - `n_per_label` (dict[str, int]): label -> actual number of components fitted (may be less than requested if samples were scarce).
- **Methods:**
  - `labels() -> list[str]`: sorted list of all label keys.
  - `total() -> int`: total number of centroids across all labels.

### `fit_centroids`
- **What it does:** Fits one `sklearn.mixture.GaussianMixture` per unique non-empty label value, collecting results into a `CentroidSet`.
- **Args:**
  - `df` (pd.DataFrame): the main dataframe containing vectors indexed by session_id.
  - `label_series` (pd.Series): per-row label assignments, same length as `df`.
  - `n_per_label` (dict[str, int] | int, default=1): number of GMM components per label. If int, applied uniformly. If dict, looked up per label (missing labels default to 1).
  - `covariance_type` (str, default="full"): passed to `GaussianMixture`. One of "full", "tied", "diag", "spherical".
  - `random_state` (int, default=42): random seed for GMM initialization.
  - `progress_callback` (callable | None): called as `callback(label_str, label_index)` after each label is fitted.
- **Returns:** `CentroidSet` with fitted centroids for all valid labels.
- **Raises:** No exceptions raised directly; labels with fewer than 2 samples or empty/None labels are silently skipped.
- **Example call:**
  ```python
  cset = fit_centroids(df, df["speaker"], n_per_label=3, covariance_type="full")
  ```

## Inputs & Outputs

**IN:**
- `df`: DataFrame with a `session_id` column and a `centroid` column containing high-dimensional vectors (D-dim each).
- `label_series`: Series of string labels aligned to `df` rows.

**OUT:**
- `CentroidSet` with GMM means of shape (k, D), covariances of shape (k, D, D), and weights of shape (k,) per label. All stored as float32.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Normal multi-sample labels | Full CentroidSet with all labels | Standard path |
| Label with only 1 sample | Label silently skipped | GMM requires >= 2 samples |
| Empty string or None label | Label silently skipped | Treated as missing |
| `n_per_label` > n_samples for a label | K clamped to n_samples | `max(1, min(k_target, n_samples))` |
| No valid labels at all | Empty CentroidSet (all dicts empty) | Returns successfully |
| `n_per_label` as dict with missing label | Defaults to 1 for that label | Via `.get(label, 1)` |

## Where This Is Used

- `backend/centroids/__init__.py` -- re-exported as public API.
- `backend/roc/curves.py:21` -- imports `CentroidSet` for ROC computation relative to centroids.
- `frontend/callbacks/dump.py:23` -- calls `fit_centroids` when the user clicks "find centroids".
- Multiple test files in `code/TESTS/tests/` exercise `fit_centroids` and `CentroidSet`.

## Known Gotchas & Non-Obvious Behavior

- `sklearn.mixture.GaussianMixture` is imported inside the function body (lazy import), so sklearn is only required when centroids are actually computed.
- `reg_covar=1e-4` is set explicitly to prevent singular covariance matrices on near-duplicate vectors. This is higher than sklearn's default of `1e-6`.
- All output arrays are cast to float32 regardless of the input dtype. This keeps downstream memory predictable but may lose precision from float64 GMM internals.
- The `progress_callback` receives the enumeration index from `groupby`, not a fraction -- the caller must know total label count to compute a percentage.

## Dependencies

- `numpy`, `pandas` -- array and dataframe operations.
- `sklearn.mixture.GaussianMixture` -- lazy import, only needed at fit time.
- `backend.common.data_access` -- `SESSION_COL`, `get_vectors_for_sessions`.
