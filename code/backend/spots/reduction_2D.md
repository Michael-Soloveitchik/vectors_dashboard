# reduction_2D.py

## Purpose
Thin re-export forwarder that satisfies the spec-mandated import path `backend/spots/reduction_2D.py`. The actual 2D dimensionality reduction algorithm lives in `backend/reduction_2d/`. This file re-exports `reduce_2d`, `ReductionResult`, and `ReductionCache` so the dashboard can import from the path specified in the brief without duplicating any logic.

## Motivation
The project brief requires the import path `backend.spots.reduction_2D`, but the actual implementation lives in `backend.reduction_2d` (lowercase, no underscore-capital convention) to keep filenames importable and consistent. This file bridges the gap: it is a pure forwarder with no logic of its own. Adding logic here would split the algorithm across two locations -- it must remain a one-line re-export.

## Re-exports
- `reduce_2d` -- the main reduction function (PCA / t-SNE / UMAP)
- `ReductionResult` -- dataclass holding projected coordinates and metadata
- `ReductionCache` -- cache object for method-aware caching across callbacks

All re-exported from `backend.reduction_2d`.

## Inputs & Outputs
Passthrough only. See `backend/reduction_2d/` for the actual API.

## Expected Behavior & Edge Cases

| Scenario | Expected output | Notes |
|----------|----------------|-------|
| Import from `backend.spots.reduction_2D` | Works identically to importing from `backend.reduction_2d` | Pure re-export |
| Import from `backend.spots` (the package) | Also works -- `__init__.py` re-exports the same symbols | Double forwarding layer |

## Where This Is Used
- `backend/spots/__init__.py` -- re-exports these symbols at the package level.

## Known Gotchas & Non-Obvious Behavior
- The filename uses `_2D` (capital D) to match the spec exactly. Python imports are case-sensitive, so `from backend.spots.reduction_2d` (lowercase d) will fail.
- This file must never contain logic beyond re-exports. If you need to modify the reduction algorithm, edit `backend/reduction_2d/` instead.

## Dependencies
- `backend.reduction_2d` (internal -- the actual implementation)
