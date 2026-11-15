"""Transactions: every user action ends when all panels agree.

Each panel declares the inputs it is drawn from (``PANELS``). From those:

* ``store_want``  {panel: stamp}  — what every panel SHOULD be showing for
                  the current controls / stores (one server callback);
* ``stamp-<p>``   what panel ``p`` actually finished rendering — written
                  by the panel's own callback together with its figure,
                  also when it answers "nothing to redraw" or with a
                  ``Patch``;
* the browser compares the two (``store_late``); ``assets/sync.js`` shows
  "● in sync" only when nothing is late AND no callback request is in
  flight — the end of the transaction — else "◌ updating projection · …";
* the reconciler: if a panel stays behind (response lost, dropped,
  superseded, reordered — whatever the cause) the browser bumps
  ``store_kick`` with exponential back-off and the stale callbacks run
  again with the CURRENT inputs. The UI converges instead of freezing.

A stamp is ``None`` while the action is still settling (theme unknown,
fit missing, thresholds not yet re-stamped for the new fit).
"""
from __future__ import annotations

import functools
import traceback

from dash import Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from .. import figures as F
from ..debug_log import dlog, trace
from ..engine import MULTI_CENTROID_ARG, state
from ..engine.concurrency import Superseded
from ..engine.data import MY_LABEL
from ..ids import IDS

SHARED = [("fit", IDS.store_fit, "data"), ("thr", IDS.store_thr, "data"),
          ("alpha", IDS.ctrl_alpha, "value")]
SEL = ("sel", IDS.store_sel, "data")
FOCUS = ("focus", IDS.store_focus, "data")
THEME = ("theme", IDS.store_theme, "data")
FIT_CONTROLS = [("model", IDS.ctrl_model, "value"),
                ("label_keys", IDS.ctrl_labels, "value"),
                ("methodology", IDS.ctrl_methodology, "value"),
                ("aggregator", IDS.ctrl_aggregator, "value"),
                ("skip", IDS.ctrl_skip, "value"),
                ("n_global", IDS.ctrl_n_centroids, "value"),
                ("pcn", IDS.store_per_class_n, "data"),
                ("labels", IDS.store_labels, "data")]
LABELS = ("labels", IDS.store_labels, "data")   # the user's own labels

PANELS: dict[str, list[tuple[str, str, str]]] = {
    "projection": SHARED + [SEL, FOCUS,
                            ("reduction", IDS.ctrl_reduction, "value"),
                            ("perplexity", IDS.ctrl_perplexity, "value"),
                            ("n_nbrs", IDS.ctrl_n_neighbors, "value"),
                            ("min_dist", IDS.ctrl_min_dist, "value"),
                            ("max_pts", IDS.ctrl_max_points, "value"),
                            ("opts", IDS.ctrl_scatter_opts, "value"),
                            ("color_by", IDS.ctrl_color_by, "value"),
                            ("echo", IDS.lasso_echo, "data"), LABELS, THEME],
    "matrix": SHARED + [SEL, FOCUS,
                        ("mode", IDS.ctrl_matrix_mode, "value"),
                        ("sort", IDS.ctrl_matrix_sort, "value"),
                        ("color", IDS.ctrl_color_auto, "value"),
                        ("power", IDS.ctrl_color_power, "value"),
                        ("crange", IDS.ctrl_color_range, "value"), LABELS,
                        THEME],
    "curves": SHARED + [SEL, FOCUS, ("curve", IDS.ctrl_curve, "value"),
                        ("show", IDS.ctrl_roc_show, "value"),
                        ("axes", IDS.ctrl_roc_axes, "value"), THEME],
    "distribution": SHARED + [SEL, FOCUS,
                              ("view", IDS.ctrl_dist_view, "value"), THEME],
    "summary": SHARED + [SEL, LABELS],
    "inspector": SHARED + [FOCUS],
}
NAMES = list(PANELS)


def spec_from_controls(model, label_keys, methodology, aggregator, skip,
                       n_global, pcn, labels=None) -> dict | None:
    """The fit spec the current sidebar asks for (``None`` = incomplete).
    Shared by ``_fit`` and the wanted-state callback so they cannot drift."""
    if not (model and methodology):
        return None
    multi = methodology in MULTI_CENTROID_ARG
    return {"model": str(model), "label_keys": list(label_keys or []),
            "methodology": str(methodology),
            "aggregator": str(aggregator or "mean"),
            "n_global": int(n_global or 1) if multi else 1,
            "per_class_n": dict(pcn or {}) if multi else {},
            "skip": list(skip or []),
            # own labels are part of the fit only when they ARE the classes
            "labels_rev": int((labels or {}).get("rev") or 0)
            if MY_LABEL in (label_keys or []) else 0}


def stamp(panel: str, v: dict) -> str | None:
    fit = v.get("fit")
    if not fit or ("theme" in v and v["theme"] is None):
        return None
    if not fit.get("error") and not state.thr_synced(fit, v.get("thr")):
        return None
    rest = {k: x for k, x in v.items() if k != "fit"}
    return F.sig(panel, fit.get("token"), rest)


def panel(app, name: str, outputs: list, states: list | tuple = (),
          on_error=None):
    """Register ``fn(v, *states) -> outputs`` as panel ``name``.

    The wrapper adds the kick input and the stamp output, skips the call
    while the action is still settling, and turns an exception into
    ``on_error(exc, v)`` (logged) so one failing panel never blocks the
    others — it is stamped too, i.e. it does not count as frozen."""
    spec = PANELS[name]
    keys = [k for k, _, _ in spec]

    def deco(fn):
        @functools.wraps(fn)
        def run(*args):
            v = dict(zip(keys, args[:len(keys)]))
            st = args[len(keys) + 1:]              # after the kick input
            mark = stamp(name, v)
            if mark is None:
                raise PreventUpdate
            try:
                out = fn(v, *st)
            except Superseded:                     # a newer request owns it
                raise PreventUpdate
            except PreventUpdate:
                out = tuple([no_update] * len(outputs))
            except Exception as exc:
                dlog.event("panel_error", panel=name,
                           err=f"{type(exc).__name__}: {exc}",
                           tb=traceback.format_exc(limit=6))
                if on_error is None:
                    raise
                out = on_error(exc, v)
            out = out if isinstance(out, tuple) else (out,)
            return (*out, mark)
        run.__name__ = f"_panel_{name}"
        return app.callback(
            *outputs, Output(f"stamp-{name}", "data"),
            *[Input(i, p) for _, i, p in spec], Input(IDS.store_kick, "data"),
            *states)(trace(run))
    return deco


LATE_JS = """function(want, fit, a, b, c, d, e, f) {
    const names = %s;
    const got = [a, b, c, d, e, f];
    if (!want || !want.fit) { window.__vd_late = null; return null; }
    const late = names.filter((n, i) => got[i] !== want[n]);
    if ((fit || {}).token !== want.fit) { late.unshift('fit'); }
    window.__vd_late = late;          // read by assets/sync.js
    return late;
}""" % NAMES

# Re-run stale callbacks when a panel stays behind with NOTHING in flight
# (a lost / dropped / superseded response). Back-off: 1 s, 2 s, 4 s, … ≤ 60 s.
RECONCILE_JS = """function(n, kick) {
    const st = window.__vd_rec || (window.__vd_rec = {since: 0, next: 0, tries: 0});
    const late = window.__vd_late;
    const now = Date.now();
    const idle = (window.__vd_pending || 0) === 0;
    if ((late && late.length === 0) || !idle) {
        st.since = 0; st.tries = 0; return window.dash_clientside.no_update;
    }
    if (!st.since) { st.since = now; st.next = now + 1000; return window.dash_clientside.no_update; }
    if (now < st.next) { return window.dash_clientside.no_update; }
    st.tries += 1;
    st.next = now + Math.min(60000, 1000 * Math.pow(2, st.tries));
    return (kick || 0) + 1;
}"""


def register(app) -> None:
    union: list[tuple[str, str, str]] = list(FIT_CONTROLS)
    for spec in PANELS.values():
        union += [t for t in spec if t not in union]
    keys = [k for k, _, _ in union]

    @app.callback(Output(IDS.store_want, "data"),
                  *[Input(i, p) for _, i, p in union],
                  Input(IDS.store_kick, "data"))
    def _want(*args):
        v = dict(zip(keys, args))
        spec = spec_from_controls(*[v[k] for k, _, _ in FIT_CONTROLS])
        out = {"fit": state.token(spec) if spec else None}
        for name, pspec in PANELS.items():
            out[name] = stamp(name, {k: v[k] for k, _, _ in pspec})
        return out

    app.clientside_callback(
        LATE_JS, Output(IDS.store_late, "data"),
        Input(IDS.store_want, "data"), Input(IDS.store_fit, "data"),
        *[Input(f"stamp-{n}", "data") for n in NAMES])
    app.clientside_callback(
        RECONCILE_JS, Output(IDS.store_kick, "data"),
        Input(IDS.tick, "n_intervals"), State(IDS.store_kick, "data"))
