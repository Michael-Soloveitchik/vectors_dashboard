# __init__.py

## Purpose
Package initializer for the `spots` module. Re-exports the 2D reduction API from `reduction_2D.py`, which itself forwards from `backend.reduction_2d`.

## Motivation
Provides the spec-mandated import path `backend.spots` as a package entry point. The double forwarding layer (`spots/__init__.py` -> `spots/reduction_2D.py` -> `reduction_2d/`) exists because the brief specifies `backend/spots/reduction_2D.py` as the canonical path while the implementation lives in `backend/reduction_2d/` for clean naming.

## Re-exports
- `reduce_2d`, `ReductionResult`, `ReductionCache` from `.reduction_2D`

## Dependencies
- `.reduction_2D` (internal forwarder to `backend.reduction_2d`)
