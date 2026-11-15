# __init__.py

## Purpose
Package initializer for the `propagation` module. Re-exports `propagate_labels` as the single public function.

## Motivation
Provides a clean import path (`from backend.propagation import propagate_labels`) without exposing the internal `knn` module name.

## Re-exports
- `propagate_labels` from `.knn`

## Dependencies
- `.knn` (internal submodule)
