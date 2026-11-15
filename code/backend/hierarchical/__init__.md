# __init__.py

## Purpose
Package initializer for the `hierarchical` module. Re-exports the public API surface: `cluster`, `HierarchyResult`, `dendrogram_traces`, `node_xy_lookup`, and `HierarchyLabels`.

## Motivation
Provides a single import point (`from backend.hierarchical import cluster, dendrogram_traces, HierarchyLabels`) so callers do not need to know the internal file layout.

## Re-exports
- `HierarchyResult`, `cluster` from `.clustering`
- `dendrogram_traces`, `node_xy_lookup` from `.dendrogram`
- `HierarchyLabels` from `.labels`

## Dependencies
- `.clustering`, `.dendrogram`, `.labels` (internal submodules)
