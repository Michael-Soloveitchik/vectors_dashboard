# __init__.py

## Purpose
Package initializer for the `neighbors` module. Re-exports `find_nearest` as the single public function.

## Motivation
Provides a clean import path (`from backend.neighbors import find_nearest`) without exposing the internal `lookup` module name.

## Re-exports
- `find_nearest` from `.lookup`

## Dependencies
- `.lookup` (internal submodule)
