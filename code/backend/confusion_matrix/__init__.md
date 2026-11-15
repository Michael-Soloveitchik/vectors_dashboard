# __init__.py

## Purpose
Package initializer for `backend.confusion_matrix`. Re-exports the public API so consumers can import directly from the package without knowing the internal module layout.

## Public API Surface

| Symbol | Source module | Type |
|--------|-------------|------|
| `ConfusionMatrix` | `matrix.py` | Dataclass |
| `build_matrix` | `matrix.py` | Function |
| `sort_matrix` | `sort.py` | Function |
| `available_sort_keys` | `sort.py` | Function |
| `apply_color_transform` | `transforms.py` | Function |

## Example Import
```python
from backend.confusion_matrix import build_matrix, sort_matrix, apply_color_transform
```

## Dependencies

- `.matrix` -- `ConfusionMatrix`, `build_matrix`
- `.sort` -- `sort_matrix`, `available_sort_keys`
- `.transforms` -- `apply_color_transform`
