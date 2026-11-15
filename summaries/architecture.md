# vectors_dashboard — architecture

Linked 4-panel explorer for embedding vectors (speaker / device embeddings).
Rewritten 2026-11-14; the previous frontend is parked in `DELETE_ME_LATER/frontend_old/`.

## Tree

```
vectors_dashboard/
├── run.ipynb              # paths → Gui.demo() / Gui.real() → gui.run(port)
├── _runserver.py          # same, outside Jupyter: [--real] [--model M] [--port P]
├── data/                  # generate.py (synthetic) · real_data.py (parquet + csv + wavs)
├── code/
│   ├── backend/           # representations, reduction_2d, ... (untouched)
│   ├── plot/              # old toy builders + ipywidgets dashboard — NOT used by frontend any more
│   └── frontend/
│       ├── gui.py         # Gui(df, …) · Gui.demo() · Gui.real() · Gui.from_arrays() · run/stop/url/analysis
│       ├── server.py      # werkzeug server in a thread, stop / stop_all, never kills foreign ports
│       ├── ids.py · theme.py · debug_log.py
│       ├── assets/        # style.css (layout, glass tokens) · theme.css (Windows-style polish, switch, full screen) · sync.js (sync indicator, mouse roles, zoom reset, ROC spot drag, audio dock, full screen)
│       ├── engine/        # compute core, no Dash
│       │   ├── concurrency.py  FlightCache (LRU + single-flight) · KindCache · HeavyGate · freeze
│       │   ├── data.py         DataHub / ModelFrame (unit-norm matrix per model, wav index)
│       │   ├── analysis.py     FitSpec → Analysis (scores, decisions, summary) · Engine
│       │   ├── fit.py          one fit; failures become cached error values
│       │   ├── state.py        store contracts: token / model scoping, selection + focus validation, neighbours
│       │   ├── curves.py       per-class OvR / pair / selection ROC, EER, operating points, DET
│       │   ├── project.py      2D projection cache, centroid placement, accepted hulls, covariance ellipses
│       │   ├── ptsne.py        parametric t-SNE (small network, trained on demand)
│       │   └── matrix.py       row picking + sorting, score_XY matrix, centroid matrix, confusion
│       ├── figures/       # pure builders: (Analysis, view state) → go.Figure
│       ├── layout/        # sidebar.py · cards.py · widgets.py
│       ├── callbacks/
│       │   ├── panels.py       transactions: wanted vs rendered stamps, reconciler, panel() wrapper
│       │   ├── state.py        controls → fit; theme; tab id
│       │   ├── thresholds.py   sliders → versioned threshold store
│       │   ├── render.py       projection · matrix · curves · distribution · summary
│       │   ├── selection.py    selection: one intent callback per source + reducer
│       │   ├── select.py       click router (inspect / Ctrl+click / tree / click-to-threshold) · inspector
│       │   ├── labels.py       the user's own labels (name the selection)
│       │   ├── ptsne.py        the p-t-SNE "train" button
│       │   └── export.py · _util.py
│       ├── inspector.py   # stats column content (vector / pair / class, neighbours)
│       └── audio.py       # /_wavlens route → WavLens standalone HTML for the audio dock
├── DELETE_ME_LATER/       # test_engine.py · test_concurrency.py · test_cases.py · audit_sync.py · frontend_old/
├── summaries/ · dumps/ · logs/
```

## Data flow

```
sidebar controls ──► _fit ──► store_fit {spec, labels, error}      (FitSpec = model, labels, methodology,
                               │                                     aggregator, n centroids, skip)
                               ▼
                     Engine.analysis(spec)  ── cached Analysis: rep, raw (N,K), norm(alpha) (N,K)
                               ▲
global slider / EER / reset / ◀▶ ─► per-class slider values ─► store_thr {label: thr}   (normalized [0,1])
lasso / class picker ─► store_sel [row, ...]        click ─► store_focus {rows | labels}
                               │
        4 render callbacks + summary chips read (store_fit, store_thr, alpha, store_sel, store_focus)
```

One rule everywhere: class `c` accepts a sample iff `norm[:, c] >= thr[c]`.
Scatter markers, distribution lines + FR/FA, ROC operating points, confusion matrix, header chips and the CSV export all come from `Analysis.decide()`.

## Synchronization — every action is a transaction

Stores carry what they belong to; every consumer checks it (`engine/state.py`):

| store | content | valid when |
|---|---|---|
| `store_fit` | spec, labels, error, `token = hash(spec)` | — |
| `store_thr` | `token`, `{label: thr}` | token == fit token |
| `store_sel` | `model`, lasso rows **or** class names | model == fit model (class names: same label columns) |
| `store_focus` | `model`, rows **or** labels | model == fit model |

Rules:

1. One writer per store. Thresholds: global slider / EER / min DCF / reset / ◀▶ → slider values → `store_thr`. Selection: see below.
2. Nothing is drawn against a half-updated state: a panel waits until `store_thr` is re-stamped for the current fit (one full render per fit, never two).
3. `callbacks/panels.py`: `store_want` = stamp each panel should show; `stamp-<panel>` = stamp it finished rendering (also on "nothing to redraw" and on Patch). `assets/sync.js` shows `● in sync` only when all stamps match **and** no callback request is in flight; otherwise `◌ updating <panels>`.
4. Reconciler: a panel late with nothing in flight (lost / dropped / reordered response) → the browser bumps `store_kick` (1 s, 2 s, 4 s … ≤ 60 s) and the stale callbacks rerun with current inputs.
5. Partial updates are relative to what the browser holds, not to "what triggered me": each figure's `sig-*` store echoes the structural signature plus one hash per aspect (thresholds / selection / focus); the answer is the full figure, one Patch covering every differing aspect, or nothing (`_util.respond`).
6. A failing panel shows its error and is stamped; the others keep working.
7. Selection = clauses per source, resolved by a strategy (after Mosaic): `store_sel = {model, clauses: {lasso | classes | rows}, mode: single | union | intersect, seq}`. Every source (drag, class picker, plot click, inspector buttons, clear / model / labels) has its OWN callback that only records an intent `{seq, …}`; one reducer folds all intents newer than the store's `seq`. Dash applies only the last of overlapping responses of one callback, so a multi-source writer loses real updates behind no-op calls; with intents nothing is lost.
8. The store owns the selection picture. Plotly dims points itself on every drag, re-derives lit points from its persistent outline (centroid markers included), and keeps the user's lit-points state while `selectionrevision` is unchanged. So: the outline is dropped once captured, `selectedpoints` come from the store, every store-driven update carries a new `selectionrevision`, non-vector marker traces are reset, and after ANY drag the browser asks the server to re-assert (`lasso-echo`).
9. One click router (`select._click`) consumes every plot click and clears `clickData`, so the same point can be clicked again.

Audit: `DELETE_ME_LATER/audit_sync.py <url> <n> <seed>` — random macro actions + a no-wait burst; after each transaction checks slider == distribution line == ROC operating point, hollow markers == Σ(1−TPR)·n, badge == chip == lit points, class sets, theme, focus.

## Mouse roles (the round “i” in each card header lists them)

| gesture | projection | vector matrix |
|---|---|---|
| click | inspect vector (centroid → class) | inspect the pair (classes / confusion cell → class pair) |
| Ctrl / Cmd + click | add / remove the vector from the selection (centroid → select class) | add / remove both vectors |
| drag | box-select | zoom |
| Shift + drag | add the box to the selection | — |
| double-click | clear selection | Plotly reset |
| right-click · ⌂ | reset zoom (exact home view) | reset zoom |
| wheel | zoom | zoom |

The projection's `click` switch swaps the first two rows: `inspect` (default) or `select` — in `select` a plain click adds / removes the point and Ctrl+click inspects.
ROC: drag a ● operating point along its curve → that class's threshold follows the cursor (`sync.js` → `thr-pick`, ~12 updates / s); click anywhere near a curve → the threshold jumps to that point; click a ● → inspect the class (`op-pick`). A click in a class's distribution lane moves its threshold there. All panels follow. `⌂ view` in the top bar: every zoom home, selection cleared, inspector closed.
Keys: `x` clear selection · `Esc` close inspector. Tree mode: click a junction → select its classes. ROC: click an operating point → inspect class (Ctrl → select it).
Modifier keys are recorded on mouse-down by `assets/sync.js` (`store-mods`); Plotly events do not carry them.

## Thresholds

Per class, in normalized score space. Ways to set them — all end in the same per-class slider values → `store_thr`:

| control | effect |
|---|---|
| per-class slider (sidebar) | that class |
| ◀ ▶ next to it | previous / next score boundary: exactly one sample changes side |
| "all classes" slider | every class to one value |
| EER · min DCF · reset | every class to its equal-error / minimum-cost threshold · 0 |
| click on a class's ROC curve | that class → the clicked point's threshold |
| click in a class's distribution lane | that class → the clicked score |

## Own labels

Name the current selection (Selection → "my labels"). Stored as one extra column `my_label` (`—` = unlabelled), saved to `dumps/my_labels_<model>.csv`, restored on start. Usable as class labels (the fit then includes the labels revision), projection colour, matrix sort key (colour strips / block labels follow the first column sort key), in the selection composition and in predictions.csv. Labelling does not refit an analysis that does not use `my_label`.

## Stable views

Every plot has a fixed home view (`layout.meta.home`), restored exactly by ⌂ / right-click. Projection axes no longer autorange (the picture used to rescale when traces changed); the density trace is always present (hidden when off) because a trace appearing made Plotly drop the user's zoom. Zoom-out is bounded (`minallowed` / `maxallowed`): about half a span beyond the data.

## Layout

Top bar · sidebar · the 2×2 panel grid. Two parts appear on demand and never cover the panels:

| part | where | content | closes |
|---|---|---|---|
| stats column | narrow, right of the grid | the inspected vector / pair / class: tags, metadata, score per class, neighbours | ✕ · Esc |
| audio dock | full width, under all panels | WavLens for the inspected vector(s) — only when the data has audio | with the stats column; ▾ folds it to its title bar |

Every card has a ▢ button: the panel becomes a big window over the page (Esc or the button again restores it).
WavLens follows the page theme (cookie `vd_theme` for a new frame, `postMessage` for an open one) and also has its own sun / moon button.

## Projection

PCA · t-SNE · UMAP · **p-t-SNE** (parametric t-SNE, `engine/ptsne.py`): a small network trained with the t-SNE objective on the displayed rows. `▶ train` runs 300 gradient steps (CPU, ≤ 2 threads, ~5 s); press again to refine. The model is kept per (model, perplexity, max points); until the first round the panel says so.
Overlays: accepted envelope · covariance ellipses (σ selectable) · density. The clicked vector is ringed in dark blue (`ring` token) — a colour no class has.

## Score distribution

Views: histogram · box (quartiles, whiskers 5–95 %) · samples. Scale: `normalized` (the 0 … 1 scale thresholds are set on; sidebar "score scale": linear ↔ rank) or `raw` (the methodology's own scores; each threshold line is drawn where it falls among them).

## Matrix

`sort by` takes several keys; they are numbered in the control and in the subtitle (1. class → 2. hierarchical): the first key makes the blocks, later keys order rows inside them. Keys: `class`, `hierarchical`, `random`, any label column, `my_label`.
`skip from fit` accepts classes and `column=value` filters (e.g. `gender=female`).

## Serving and notebook display

- `gui.run(mode="inline")` shows the dashboard in the notebook AND prints the URL(s) for a browser tab; `external` prints only; `tab` opens a local tab.
- The page is relocatable (`server.make_relocatable`): asset URLs are relative and the Dash path prefix is taken from the browser's address, so it also works behind `/proxy/<port>/` (code-server, jupyter-server-proxy). Checked through a strip-prefix proxy.
- The inline cell output probes, in the browser, `/proxy/<port>/` on the notebook's origin(s), then `host:<port>`, then the printed URLs (`/_alive`, CORS-open), and embeds the first that answers; otherwise it shows links.

## Look

Windows-11-style widgets: translucent blurred panels (`backdrop-filter`), 12 px corners, lit top edge, soft header fade, quiet wallpaper glow; one accent blue; nothing animates by itself. Light / dark follow the OS, the sun / moon switch (top right) overrides and is remembered. Tokens live in `assets/style.css` (`--glass`, `--raised`, `--accent`, …) and are mirrored for Plotly in `theme.py`.

## What comparable tools do (survey, 2026-11-14)

| tool | taken from it |
|---|---|
| Mosaic (UW IDL), used by Apple Embedding Atlas | selection as clauses per source with single / union / intersect resolution; empty selection = everything |
| Embedding Atlas, Renumics Spotlight | nearest neighbours, density, colour-by, list of selected items → click to inspect / listen |
| FiftyOne embeddings panel | two-way lasso sync, categorical + continuous colour-by, `x` / double-click to clear |
| BOSARIS, llreval | min DCF next to EER (here P_target 0.01), threshold at min DCF |
| Dash community threads | only the last of overlapping responses is applied; completion can be out of order → stamps, held-state patches, reconciler |

Not taken: text search, automatic cluster labels, saved views, Cllr / calibration plots.

## Concurrency and resources

One thread per request (werkzeug). No thread is spawned by the dashboard except one log watchdog.

| mechanism | where | guarantees |
|---|---|---|
| single-flight LRU (`FlightCache`) | fits (4), projections (12), frames (3), per-analysis memo | N concurrent callers of one key → one computation; different keys in parallel; no lock held while computing; failures not cached |
| heavy gate (`HeavyGate`, default 2 slots) | fit, t-SNE / UMAP | bounded CPU on a shared box (`Gui(heavy_slots=…, blas_threads=…)`) |
| latest-wins | queued fit / projection per browser tab | a queued job no tab wants any more is skipped; a caller that still wants it is served |
| `rep_lock` per analysis | every call into a backend representation | backend classes need not be thread-safe |
| read-only arrays | Vn, raw, norm, true_idx, projection xy | one request cannot mutate what another reads |
| immutable `Analysis` | after fit | no shared mutable knob (alpha is an argument, not `NORM_ALPHA`) |

Bounds (nothing grows with session length):

| resource | bound |
|---|---|
| fitted analyses | 4 (LRU) — each holds its rep, N×K scores |
| per-analysis memo | norm 3 alphas · class curves 3 · pair curves 2 · pair matrix 3 · hierarchical order 3 |
| ROC curves | stored thinned to 2000 points (AUC / EER from the full curve) |
| model frames | 3 (one unit-norm float64 matrix each; no second copy) |
| projections | 12 |
| class-picker selection | stored as class names, not row lists |
| WavLens pages | 24 (LRU) |
| tab table (latest-wins) | 256 |
| log | 5 MB, one rotation; one watchdog thread and one handle per process (closed on module reload) |
| servers | one per port; `gui.stop()` / `server.stop_all()` |

Checks: `DELETE_ME_LATER/test_concurrency.py` — 16 callers → 1 fit; peak concurrency ≤ slots; latest-wins; 252 concurrent mixed reads equal the single-thread reference; cache caps; RSS flat between identical rounds; threads / fds flat over 6 reload + run + stop cycles.

## Panels

1. **Projection** — PCA / t-SNE / UMAP of unit-norm vectors (cached per model + params, independent of methodology).
   Per point: ● accepted · ○ below own-class threshold · ✕ another class wins · ◇ skipped in fit. Dashed hull = accepted envelope per class.
   Drag = select, click = inspect.
2. **Matrix** — pair scores (`rep.score_XY`, ≤ 500 rows stratified, class strips, sort keys incl. hierarchical / random) · centroid scores · confusion (true × predicted + rejected, follows thresholds and selection).
3. **Detection curves** — per-class one-vs-rest + pair·cosine + pair·normalized + selection-vs-rest; ROC or DET; log axes; operating points move with thresholds. AUC and EER in the legend.
4. **Score distribution** — one lane per class: own samples vs everyone else, threshold line, FR / FA per lane; histogram or per-sample strip coloured by true class.

Inspector drawer: scores per class vs thresholds, metadata, pair score; WavLens (waveform / spectrogram / playback, two clips zero-padded onto one timeline) when `wav_dir` has `**/<session_id>.wav`.

## Scale

Metrics always use all vectors. Drawing is capped: scatter `max points` (default 4000), matrix 500 rows, pair ROC 400k sampled pairs above 1200 vectors. Verified on 45,967 × 192.
