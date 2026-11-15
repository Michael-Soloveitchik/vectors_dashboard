# __init__.py

## Purpose
Package initializer for `backend.centroids`. Re-exports the public API so consumers can import directly from the package without knowing the internal module layout.

## Public API Surface

| Symbol | Source module | Type |
|--------|-------------|------|
| `CentroidSet` | `gmm.py` | Dataclass |
| `fit_centroids` | `gmm.py` | Function |
| `project_with_centroids` | `projection.py` | Function |

## Example Import
```python
from backend.centroids import fit_centroids, project_with_centroids, CentroidSet
```

## Dependencies

- `.gmm` -- `CentroidSet`, `fit_centroids`
- `.projection` -- `project_with_centroids`
