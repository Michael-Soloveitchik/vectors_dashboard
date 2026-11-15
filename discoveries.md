[2026-05-07 00:30] Finding: ACPAN/non-centroid methodologies threw AxisError on render
Expected: All 13 methodologies render dist + ROC after methodology dropdown change
Implication: build_rep_from_state only rehydrates _centroids/_owner — ACPAN reads _proxies (None) → score_xy → l2_normalize axis=-1 dim=0 fails
Fix: get_cached_rep now refits inline on cache miss (df_m passed by render callbacks)
Seen in: backend/representations/ac_pan.py:116 → base.py:535 → math_utils.py:18

[2026-05-07 00:30] Finding: CM via `score_XY` collapses to identical signature for all cosine-family methodologies
Expected: Distinct CM per methodology (NCM, Cosine, KNN, CosFace, SRC, VMFMixture, MultiProxy)
Result: All 7 produce CM mean=0.097 min=-0.177 max=1.000
Implication: Mathematically correct — they share the cosine `_pair_kernel` so `score_xy(v,u)` collapses to V@V.T. Methodology distinction lives in `predict()` votes (NxK), not pair scores. Only Bingham, ArcCosine, LabelProp override _pair_kernel.
Seen in: backend/representations/base.py:402 _pair_kernel default

[2026-05-07 00:30] Finding: ROC operating-point markers FROZEN across slider sweep
Expected: Sliding selection-thr 0.0→0.99 should move (FPR, TPR) markers along ROC curves
Result: device_xx @ thr=0.07 stuck at (0.233, 1.0) for thr=0.0 AND thr=0.99
Implication: `_norm_to_raw_thresholds` translates norm→raw via argmin(|norm-target|). For ACPAN, normalize_score clips to [0,1] with most values at extremes → argmin returns same raw boundary regardless of target. Op marker position gets stuck because raw_thr unchanged.
Seen in: frontend/figures.py:122 _norm_to_raw_thresholds + plot/roc.py:103 _op_point

[2026-05-07 01:10] Finding: Fan-out callback returned empty during methodology transition
Expected: Selection-thr slider always fans out to all per-class keys
Result: For non-centroid methodologies (KNN/SRC/LabelProp/etc.), `prev` (state of store_thresholds) was momentarily empty while autofit re-seeded it. Fan-out's `if not prev: return prev or {}` guard returned empty → store stayed empty → rings drew at default thr=0 (cover all) → user saw "frozen" rings.
Fix: Added `Input(store_centroids)` to fan-out + read authoritative label list from `centroid_state.centroids.keys()` instead of `prev.keys()`. Always writes `{lbl: sel_thr}` for all current classes.
Seen in: frontend/callbacks/_interaction.py:331 _selection_thr_fans_out

[2026-05-07 01:15] Finding: Default NORM_ALPHA=0.5 collapses slider sensitivity for narrow-range methodologies
Expected: Slider thr=0.5→0.99 produces visible movement on PCA rings + ROC ops
Result: For ArcCosineKernel (norm[0.335, 1.000]), Bingham (norm[0.416, 1.000]), KNN k=1 (binary 0/1 votes), the minmax component of normalize_score compresses the bulk of mass above 0.5 → thr=0.5 vs thr=0.99 accept the SAME points → identical r/op.
Fix: Changed default `ctrl_alpha` slider value from 0.5 to 1.0 (pure ECDF rank-based). Now norm distribution is uniform [0, 1] for all methodologies → thr granularity uniformly visible. Marks updated to "0 (minmax) / 0.5 / 1 (ECDF)".
Seen in: frontend/layout/_controls.py:359 ctrl_alpha + backend/representations/base.py:460 NORM_ALPHA

[2026-05-07 00:30] Finding: Plotly lasso CDP commit requires `_fullData` not `data`
Expected: Input.dispatchMouseEvent on Lasso button + drag emits plotly_selected
Result: Drag draws polygon outline but selection event never fires (Plotly drag handler ignores synthetic mouseDown)
Workaround: emit plotly_selected directly via `el.emit('plotly_selected', {points: [...]})` using `el._fullData` to access expanded x/y arrays
Seen in: /tmp/lasso_emit4.py canary script

============================================================
[NEW] Session: 2026-11-14 17:05
============================================================

[2026-11-14 17:05] Finding: Centroid markers sat far outside their clusters in PCA.
Expected: A class prototype projects into the middle of its members.
Implication: A unit-norm prototype is ~1/mean-cos further from the origin than its members' projections (per-sample noise in high dim is orthogonal to the PC plane). Rescaling to the median vector norm does not fix it. Project `c * mean(cos(member, c))` instead.
Seen in: frontend/engine/project.py:centroid_xy (old: figures_pca.py "Bug D")

[2026-11-14 17:05] Finding: pair·cosine ROC collapsed to AUC 0.5 for ArcCosineSVM (0.68 for SRC).
Expected: Same embedding cosine AUC (0.955) for every methodology.
Implication: `rep.cosine_pairs` routes cosine through `normalize_score`, whose reference grid is in the methodology's own score units — cosine clips to a constant. The pair·cosine baseline now uses raw cosine.
Seen in: frontend/engine/curves.py:pair_curves

[2026-11-14 17:05] Finding: predictions.csv applied normalized thresholds in raw-score space.
Expected: Same accept / reject as the panels.
Implication: `rep.predict(V, thresholds=thr)` defaults to `normalized_thresholds=False`. Export now uses `Analysis.decide()` (the panels' rule).
Seen in: DELETE_ME_LATER/frontend_old/callbacks/_dump.py:83

[2026-11-14 17:05] Finding: A dash.Patch can arrive before the first full figure.
Expected: Initial render is triggered by the fit store.
Implication: Dash defers a callback until its Input stores settle, then may report only the LAST one as triggered (store_thr) → "Cannot patch undefined", blank panels. Guard every fast path with a structural signature echoed back as State.
Seen in: frontend/callbacks/render.py:_sig

[2026-11-14 17:05] Finding: Audio never loaded for real data.
Expected: `WAVS_DIR/<session_id>.wav`.
Implication: Wavs are in per-language subfolders (`wavs/<language>/<id>.wav`); index them once.
Seen in: frontend/engine/data.py:wav_path, data/real_data.py:wav_path

[2026-11-14 17:05] Finding: Old threshold panel re-created every slider on every store change.
Expected: Smooth drag.
Implication: Slider drag → store → panel re-render → new sliders fire → store … (callback storms, depth-5 nesting in the log). Sliders are now rebuilt only when the class list changes; the store has a single writer.
Seen in: frontend/callbacks/state.py

[2026-11-14 19:10] Finding: Partial figure updates went stale while every stamp said "in sync".
Expected: Patch(threshold) then Patch(focus) → both applied.
Implication: Dash drops the older of two overlapping responses of one callback. A Patch chosen from "what triggered me" then never repairs the dropped aspect. Decide from what the browser holds (echoed per-aspect hashes), patch every differing aspect.
Seen in: frontend/callbacks/_util.py:respond (found by DELETE_ME_LATER/audit_sync.py, seed 3)

[2026-11-14 19:10] Finding: Theme switch / refit silently cleared a lasso selection.
Expected: Selection survives a redraw.
Implication: Plotly emits an empty `selectedData` when the figure is redrawn; it is indistinguishable from a double-click deselect except by timing.
Seen in: frontend/callbacks/select.py:_selection (REDRAW_ECHO_S)

[2026-11-14 19:10] Finding: Each notebook re-run leaked one watchdog thread and one open log handle.
Expected: `del sys.modules[...]` reload is clean.
Implication: Module-level singletons with threads need a registry that outlives the module (`sys.__dict__`) and a close().
Seen in: frontend/debug_log.py

[2026-11-14 19:10] Finding: Per-alpha caches were unbounded (norm matrix + 400k-point pair ROC per alpha visited).
Expected: Memory flat while dragging the alpha slider.
Implication: 21 alpha steps × 46k×21 scores × 4 cached fits ≈ 1 GB worst case. Now LRU per kind and curves stored thinned.
Seen in: frontend/engine/analysis.py:MEMO_CAPS, engine/curves.py:STORE_POINTS

[2026-11-14 19:10] Finding: Plotly throws inside its own drag handler if the figure is replaced while a box-select drag is in progress.
Expected: —
Implication: `Cannot read properties of undefined (reading 'getAttribute')` in plotly.min.js moveFn; only reachable by a redraw landing mid-drag; state stays consistent afterwards.
Seen in: audit_sync.py burst phase

[2026-11-14 22:20] Finding: Per-class sliders snapped to 0 or 1 ("all or nothing").
Expected: Continuous drag with step 1e-7.
Implication: JavaScript prints numbers below 1e-6 in exponent form ("1e-7"); the slider derives its decimal count from that string, gets 0, and rounds to integers. Finest safe step is 0.000001.
Seen in: frontend/callbacks/thresholds.py:STEP

[2026-11-14 22:20] Finding: Server-sent selection was ignored after a user drag.
Expected: `selectedpoints` in the new figure / patch win.
Implication: Plotly preserves user-made selection state across `react` while `layout.selectionrevision` is unchanged (it falls back to `uirevision`). Every store-driven selection update needs a new revision. Plotly's persistent outline is likewise a second source of truth: it re-derives lit points from the box, counting centroid markers too.
Seen in: frontend/figures/scatter.py (selectionrevision), callbacks/render.py

[2026-11-14 22:20] Finding: A multi-source writer callback loses updates.
Expected: lasso → store.
Implication: When a no-op call of the same callback (echo of its own output, stale click) overlaps a real one, Dash keeps only the last response — the real update vanishes while Plotly has already dimmed the points. One callback per source writing an intent + a reducer that folds everything newer than the store's seq is loss-free.
Seen in: frontend/callbacks/selection.py

[2026-11-14 22:20] Finding: The same point could not be clicked twice; a callback that resets `clickData` starves other consumers.
Expected: Second click re-opens the inspector.
Implication: Dash fires only on prop change, so identical clickData is silent → reset it. But a callback whose Input is another pending callback's Output is deferred and then sees the reset value → exactly ONE callback may consume and reset clickData (click router).
Seen in: frontend/callbacks/select.py:_click

[2026-11-14 22:20] Finding: Plotly's drag handler throws if the plot is resized / redrawn mid-drag, and then emits no selection event.
Expected: —
Implication: The docked inspector closing (plots resize) right before a drag reproduced it. The browser now asks the server to re-assert the store's selection after every drag, so the picture cannot stay out of step.
Seen in: assets/sync.js (lasso-echo)

[2026-11-14 22:20] Finding: DOM `dblclick` never reaches a Plotly plot, and Plotly emits "deselect" only while it holds an outline.
Expected: double-click clears.
Implication: Plotly covers the page with its own layer while the button is down. Detect double-click from two quick mouse-ups.
Seen in: assets/sync.js

[2026-11-15 00:10] Finding: Toggling the density contour reset the projection zoom; ⌂ returned to a different view than the initial one.
Expected: `uirevision` keeps the user's zoom; "Reset axes" restores the first view.
Implication: A trace appearing / disappearing makes Plotly drop the preserved zoom — keep the trace and flip `visible`. Plotly's "Reset axes" goes to its own autorange, not to a given range — store the home ranges in `layout.meta` and relayout to them.
Seen in: frontend/figures/scatter.py, assets/sync.js:resetZoom
