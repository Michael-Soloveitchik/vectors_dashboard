[2026-05-07 00:00] Done: CM via `rep.score_XY(V,V)` matrix
Result: New `_cm_fig_from_score_matrix` helper bypasses toy V@V.T. Hover label "score:" not "cosine:". Selection slices both V and matrix in lockstep. Caches per (model, methodology, alpha, V_fp).
Files changed: frontend/figures.py (+helper, +score_matrix kwarg), frontend/callbacks/_render.py (wires `cached_score_xy_matrix`), frontend/callbacks/_fit.py (`_CM_SCORE_CACHE`)
Next: cosine-family methodologies still produce identical CM by definition (shared kernel)

[2026-05-07 00:05] Done: Selection-thr slider fan-out to all per-class keys
Result: Single `ctrl_pair_cos_thr` mirrors to every key in `store_thresholds`. Verified via DOM probe: 5 class rings shrink from full extent (~0.85) to 0.44–0.55 at thr=0.9.
Files changed: frontend/callbacks/_interaction.py (`_selection_thr_fans_out` callback)
Next: ROC op markers + dist vertical lines still consume RAW thr (frozen on clipped normalize_score)

[2026-05-07 00:25] Done: Fix AxisError on non-centroid methodologies (ACPAN, KNN, SRC, LabelPropagation, BinghamGeodesic)
Result: `get_cached_rep` refits inline on cache miss using df_m. ACPAN no longer crashes when selected via UI dropdown.
Files changed: frontend/callbacks/_fit.py (`get_cached_rep` accepts df_m, refits on miss), frontend/callbacks/_render.py (4 call sites pass df_m)
Next: server-side cycle of all 13 methodologies confirmed all fit + predict + score_xy + normalize_score work

[2026-05-07 00:28] Done: Plotly lasso CDP commit via `_fullData` emit
Result: `el.emit('plotly_selected', {points: [...]})` using `_fullData` to access expanded x/y arrays. Lasso 30 pts → CM shrinks to 30×30, ROC adds sel-vs-rest curves (AUC=0.913), dist gets selected overlay.
Files changed: scripts only (/tmp/lasso_emit4.py canary)
Next: integrate into `debug_dashboard` alias for repeatable lasso testing

[2026-05-07 00:30] Done: Threshold sync drift audit
Result: Identified 5 sync bugs across 4 panels. Plan written to `threshold_sync_plan.md`. P0 fix: push normalize_score INTO toy plots, accept norm thresholds directly, eliminate `_norm_to_raw_thresholds` round-trip degeneracy on clipped methodologies.
Files changed: discoveries.md (+4 entries), threshold_sync_plan.md (new), what_have_been_done.md (this)
Next: implement P0 — touch plot/roc.py, plot/dist.py, figures.py

[2026-05-07 00:45] Done: P0+P1+P4 — full thr-sync across 4 panels in normalized space
Result: Sweep thr=0.0/0.3/0.5/0.7/0.9/0.99 verified monotone:
  - PCA rings shrink (e.g. device_03 r: 0.812 → 0.455)
  - dist vertical lines exactly at slider value (0.0, 0.5, 0.7, 0.9, 0.99)
  - ROC op marker TPR decreases monotone (1.0 @ thr=0.7 → ~0.5 @ thr=0.99)
  - Op label shows norm thr (e.g. "device_00 @ thr=0.90") not raw cosine
  - PCA rings capped to data extent × 1.05 — no more off-canvas (>1.0) rings
  - Static pair-cos @ thr=0.50 and pair-norm @ thr=0.00 markers dropped
Files changed: plot/roc.py (op marker uses normalized scores), figures.py (drop _norm_to_raw_thresholds calls in dist_fig + roc_fig + pca path; add ring viewport cap)
Next: P2 (drift assertion) + P3 (CM tooltip docstring)

============================================================
Session: 2026-10-16
============================================================

[2026-10-16] Done: Promoted this fork to canonical name. Deleted the April `vectors_dashboard` (stale frontend, 174/1894 failing tests, synthetic data only) and renamed `vectors_dashboard_best` -> `vectors_dashboard`.
Result: Backend was byte-identical between the two; only frontend/plot/data differed. All in-code `vectors_dashboard_best` strings (gui title, grid header, server banner, real_data hint, architecture.md, run.ipynb) renamed.
Files changed: code/frontend/{gui.py,server.py,layout/_controls.py,layout/_grid.py}, data/real_data.py, summaries/architecture.md, run.ipynb
Next: no TESTS/ here — port the old TEST_ME.py skeleton if a regression suite is wanted.

============================================================
[NEW] Session: 2026-11-14 17:05
============================================================

[2026-11-14 17:05] Done: Frontend rewritten around one shared compute engine; new layout; WavLens inspector.
Result: All 13 methodologies render in all 4 panels (Playwright run, 0 browser console errors). Synthetic 325 vectors and real 943 / 45,967 vectors checked. Threshold tick server time: scatter ~44 ms, ROC ~24 ms, dist ~54 ms (Patch fast path). Notebook flow re-run twice in one process: server restarts on the same port, `gui.stop()` frees it.
Files changed: code/frontend/** (new: engine/, figures/, layout/, callbacks/, inspector.py, audio.py, theme.py, assets/style.css), _runserver.py, run.ipynb, data/real_data.py (wav_path), summaries/architecture.md. Old frontend moved to DELETE_ME_LATER/frontend_old/.
Next: delete DELETE_ME_LATER/ once the new UI is accepted; code/plot/ is no longer imported by the dashboard.

[2026-11-14 19:10] Done: Synchronization + concurrency hardening, friendlier API, notebook verified.
Result: Transaction layer (wanted vs rendered stamps, "in sync" indicator, reconciler). Sync audit: 4 seeds × ~45 random actions + no-wait bursts → 0 inconsistencies, 0 frozen panels, median transaction 0.3 s; 46k-vector model switch fully synced in 4.6 s. Concurrency test: single-flight, gate, latest-wins, determinism, cache caps, flat RSS / threads / fds. `run.ipynb` executed with its own kernel (nbconvert) → page served.
Files changed: code/frontend/engine/{concurrency,fit,state}.py (new), engine/{analysis,data,curves,matrix,project}.py, callbacks/{panels,thresholds,_util}.py (new), callbacks/{state,render,select,export}.py, assets/sync.js (new), gui.py, server.py, debug_log.py, layout/*, figures/*, run.ipynb, _runserver.py, summaries/architecture.md.
Next: not built from the original brief — dendrogram click-labelling, kNN label propagation, GMM centroids (backend modules exist, no UI).

[2026-11-14 22:20] Done: Second hardening pass + richer features + glass theme + notebook inline display.
Result: Sync audit (random macro actions incl. Ctrl+click, tree, combine modes, inspector actions, bursts): 0 persistent inconsistencies, 0 frozen panels, 0 console errors on the last 8 seeds. Mouse-role test passes (click / Ctrl+click / drag / Shift+drag / double-click / right-click / ⌂). Per-class slider drags continuously again; ◀▶ moves one sample. Page works behind a strip-prefix proxy; inline cell output finds the server from proxied and plain notebook origins. `run.ipynb` executed with its own kernel (ltx-2). Offline: test_engine (13 methodologies), test_concurrency, test_cases (awkward data + stale state through every figure builder) all pass.
New: clause-based selection with replace / add / intersect; class tree (click junction → select); colour-by any column or own-class score (continuous); selection composition + clickable list; inspector actions (select class / neighbours / speaker); min DCF metric + button; richer centroids.json; docked inspector; ⌂ / right-click zoom reset; Fluent glass look.
Files changed: code/frontend/callbacks/{selection.py (new), select.py, render.py, state.py, thresholds.py, panels.py, export.py}, engine/{state,matrix,curves,fit}.py, figures/{scatter,matrix,dist,roc}.py, layout/*, assets/{style.css,sync.js}, server.py, gui.py, inspector.py, theme.py, run.ipynb, summaries/architecture.md, DELETE_ME_LATER/{test_cases.py,audit_sync.py}.
Next: inline display inside the real code-server webview is untested (logic checked against a stand-in proxy); kNN label propagation and dendrogram labelling of single vectors remain unbuilt.

[2026-11-15 00:10] Done: Own labels, click-to-threshold, stable zoom, reset-all, theme polish, 100-play audit.
Result: Own labels usable as classes / colour / matrix blocks and persisted. Click on a ROC curve or distribution lane sets that class's threshold. Projection zoom: kept across every update incl. density toggle; ⌂ returns to the exact start; zoom-out bounded. Audit extended (26+ action kinds incl. roc pick, lane pick, labelling, wheel zoom, Shift+drag, double-click) with a multi-play mode.
Files changed: code/frontend/callbacks/{labels.py (new), select.py, thresholds.py, state.py, panels.py, render.py}, engine/{data,analysis}.py, figures/{common,scatter,roc,dist,matrix}.py, layout/*, assets/{style.css,sync.js}, gui.py, _runserver.py (--labels-dir), DELETE_ME_LATER/audit_sync.py (--plays), summaries/architecture.md.
Next: see the 100-play result in the session summary.

[2026-11-15 10:40] Done: Faded-blue Fluent theme; final verification before push.
Result: Blue bloom wallpaper, blue-tinted acrylic with grain, widget-style cards, softened class palette, blue heat-map ramp (replaces Viridis). 100-play audit: 100/100, 1,033 transactions, 0 persistent disagreements. test_engine / test_concurrency / test_cases pass; run.ipynb executes with its kernel; real-data smoke (WavLens, labelling, 46k model) passes.
Files changed: code/frontend/assets/style.css, theme.py, figures/{matrix,scatter}.py, .gitignore (new).
Next: push — no git remote exists for this project yet.
