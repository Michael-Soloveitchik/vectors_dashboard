"""Panel renders (projection, matrix, curves, distribution, summary).

Registered through ``panels.panel``: a render runs only once the action
has settled (theme known, thresholds re-stamped for the current fit),
its result is stamped, and a failure becomes a message in that panel.
Figures answer through ``_util.respond``: full figure, one ``dash.Patch``
covering every aspect (thresholds / selection / focus) that differs from
what the browser holds, or nothing — decided from the echoed ``sig-*``
store, never from "what triggered me".
"""
from __future__ import annotations

import time

from dash import Output, State, no_update

from .. import figures as F
from ..engine import Engine, state
from ..engine import curves as cv
from ..engine import matrix as mx
from ..engine.project import norm_method, project
from ..ids import IDS
from ..inspector import composition
from ._util import chip, respond
from .panels import panel


def register(app, engine: Engine) -> None:

    def analysis(v):
        """→ (analysis, None) or (None, message figure)."""
        fit, theme = v["fit"], v.get("theme")
        if fit.get("error"):
            return None, F.message_fig(fit["error"], theme)
        an = engine.analysis(fit["spec"])
        if an is None or not an.ok:
            return None, F.message_fig((an and an.error) or "no data", theme)
        return an, None

    def aspects(v) -> dict[str, str]:
        """One hash per dynamic aspect of the shared state."""
        return {F.THR: F.sig(state.thr_values(v["thr"])),
                # + the store's seq and the ignored-gesture counter: Plotly
                # dims points by itself on every drag, so the selection must
                # be re-asserted even when its meaning did not change.
                F.SEL: F.sig(state.sel_key(v["sel"]),
                             (v["sel"] or {}).get("seq"), v.get("echo")),
                F.FOC: F.sig(v["focus"])}

    def failed(n_out):
        def on_error(exc, v):
            fig = F.message_fig(f"{type(exc).__name__}: {exc}", v.get("theme"))
            tail = [None] if n_out == 3 else [None, no_update]   # sig (+ drawn)
            return (fig, no_update, *tail)
        return on_error

    @panel(app, "projection",
           [Output(IDS.fig_scatter, "figure"),
            Output(IDS.sub_scatter, "children"), Output(IDS.sig_scatter, "data"),
            Output(IDS.drawn_scatter, "data")],
           [State(IDS.sig_scatter, "data"), State(IDS.store_client, "data")],
           on_error=failed(4))
    def _projection(v, held, client):
        an, msg = analysis(v)
        if an is None:
            return msg, "", None, no_update
        m = norm_method(v["reduction"])
        params = ({"perplexity": float(v["perplexity"] or 30)} if m == "tsne"
                  else {"n_neighbors": int(v["n_nbrs"] or 15),
                        "min_dist": float(0.1 if v["min_dist"] is None
                                          else v["min_dist"])}
                  if m == "umap" else {})
        proj = project(engine, an.frame, v["reduction"] or "PCA", params,
                       v["max_pts"] or 4000, client=client)
        rows, labels = state.focus_parts(an, v["focus"])
        nbrs = ([r for r, _ in state.neighbours(an, rows[0])]
                if len(rows) == 1 else [])
        opts, asp = v["opts"] or [], aspects(v)
        fig = F.scatter_fig(
            an, proj, an.decide(state.thr_values(v["thr"]), v["alpha"]),
            theme=v["theme"], alpha=v["alpha"],
            sel_rows=state.sel_rows(an, v["sel"]), sel_rev=asp[F.SEL],
            focus_rows=rows, focus_labels=labels, neighbour_rows=nbrs,
            show_hull="hull" in opts, show_density="density" in opts,
            color_by=None if v["color_by"] in (None, "class") else v["color_by"])
        sig = F.sig(v["fit"]["token"], v["alpha"], m, params, proj.rows.size,
                    opts, v["color_by"], v["labels"], v["theme"])
        out, held, changed = respond(fig, sig, asp, held)
        if changed is not None:                    # patch or nothing
            if F.SEL in changed:               # the store owns the selection
                out["layout"]["selections"] = []
                out["layout"]["selectionrevision"] = asp[F.SEL]
            return out, no_update, held, no_update
        sub = (f"{proj.rows.size:,} of {an.frame.n:,} vectors"
               if proj.rows.size < an.frame.n else f"{an.frame.n:,} vectors")
        return fig, f"{sub} · {an.spec.methodology}", held, time.time()

    @panel(app, "matrix",
           [Output(IDS.fig_matrix, "figure"),
            Output(IDS.sub_matrix, "children"), Output(IDS.sig_matrix, "data")],
           [State(IDS.sig_matrix, "data")], on_error=failed(3))
    def _matrix(v, prev_sig):
        an, msg = analysis(v)
        if an is None:
            return msg, "", None
        mode, theme, color = v["mode"], v["theme"], v["color"] or []
        rows_sel = state.sel_rows(an, v["sel"])
        frows, _ = state.focus_parts(an, v["focus"])
        tv = state.thr_values(v["thr"])
        # Only what this mode depends on enters the signature: thresholds /
        # alpha move the confusion matrix but never the scores.
        sk = state.sel_key(v["sel"])
        live = {"confusion": (tv, v["alpha"], sk), "classes": (),
                "vectors": (sk, frows)}.get(mode, ())
        sig = F.sig(v["fit"]["token"], mode, v["sort"], color, v["power"],
                    v["crange"], theme, live, v["labels"])
        if sig == prev_sig:
            return no_update, no_update, no_update
        lo, hi = (list(v["crange"] or []) + [-1.0, 1.0])[:2]
        kw = dict(theme=theme, abs_on="abs" in color,
                  power=float(v["power"] or 1.0), auto_range="auto" in color,
                  lo=float(lo), hi=float(hi))
        if mode == "confusion":
            n = len(rows_sel) if rows_sel else int((an.true_idx >= 0).sum())
            return (F.confusion_fig(an, an.decide(tv, v["alpha"]), rows_sel,
                                    theme=theme),
                    f"true × predicted at current thresholds · {n:,} vectors"
                    + (" (selection)" if rows_sel else ""), sig)
        if mode == "tree":
            return (F.tree_fig(an, theme=theme),
                    "class tree · click a junction to select its classes", sig)
        if mode == "classes":
            return (F.classes_fig(an, **kw),
                    f"{an.spec.methodology} score between class prototypes",
                    sig)
        rows = mx.order_rows(an, mx.pick_rows(an, rows_sel), v["sort"])
        total = len(rows_sel) if rows_sel else an.frame.n
        sub = (f"{rows.size} × {rows.size}"
               + (f" · sampled from {total:,}" if rows.size < total else "")
               + (" · selection" if rows_sel else "")
               + f" · {an.spec.methodology} score_XY")
        by = next((k for k in (v["sort"] or []) if k in an.frame.df.columns
                   and k != "session_id"), None)
        return (F.vectors_fig(an, rows, focus_rows=frows, strip_by=by, **kw),
                sub + (f" · blocks: {by}" if by else ""), sig)

    @panel(app, "curves",
           [Output(IDS.fig_roc, "figure"), Output(IDS.sub_roc, "children"),
            Output(IDS.sig_roc, "data")],
           [State(IDS.sig_roc, "data")], on_error=failed(3))
    def _curves(v, held):
        an, msg = analysis(v)
        if an is None:
            return msg, "", None
        _, labels = state.focus_parts(an, v["focus"])
        axes = v["axes"] or []
        fig = F.roc_fig(an, alpha=v["alpha"], thr=state.thr_values(v["thr"]),
                        theme=v["theme"], sel_rows=state.sel_rows(an, v["sel"]),
                        det=v["curve"] == "det", show=v["show"] or "all",
                        log_x="x" in axes, log_y="y" in axes,
                        focus_labels=labels)
        sig = F.sig(v["fit"]["token"], v["alpha"], state.sel_key(v["sel"]),
                    v["curve"],
                    v["show"], axes, labels, v["theme"])
        out, held, changed = respond(fig, sig, {F.THR: aspects(v)[F.THR]}, held)
        if changed is not None:
            return out, no_update, held
        return fig, "one-vs-rest · ● operating point at threshold", held

    def dist_failed(exc, v):
        fig = F.message_fig(f"{type(exc).__name__}: {exc}", v.get("theme"))
        return fig, no_update, no_update, None

    @panel(app, "distribution",
           [Output(IDS.fig_dist, "figure"), Output(IDS.fig_dist, "style"),
            Output(IDS.sub_dist, "children"), Output(IDS.sig_dist, "data")],
           [State(IDS.sig_dist, "data")], on_error=dist_failed)
    def _distribution(v, held):
        an, msg = analysis(v)
        if an is None:
            return msg, {"minHeight": "100%"}, "", None
        view = v["view"] or "hist"
        rows, _ = state.focus_parts(an, v["focus"])
        fig = F.dist_fig(an, alpha=v["alpha"], thr=state.thr_values(v["thr"]),
                         theme=v["theme"], view=view,
                         sel_rows=state.sel_rows(an, v["sel"]), focus_rows=rows)
        sig = F.sig(v["fit"]["token"], v["alpha"], view, v["theme"])
        out, held, changed = respond(fig, sig, aspects(v), held)
        if changed is not None:
            return out, no_update, no_update, held
        style = {"minHeight": f"{F.lane_height(an.k, view)}px"}
        return fig, style, "own samples (colour) vs everyone else (grey)", held

    @panel(app, "summary",
           [Output(IDS.summary, "children"), Output(IDS.sel_summary, "children")],
           on_error=lambda exc, v: (
               [chip("!", f"{type(exc).__name__}: {exc}", "warn")], []))
    def _summary(v):
        fit = v["fit"]
        if fit.get("error"):
            return [chip("!", fit["error"], "warn")], []
        an = engine.analysis(fit["spec"])
        s = an.summary(state.thr_values(v["thr"]), v["alpha"])
        if not s:
            return [], []
        cc = cv.class_curves(an, v["alpha"])
        auc = sum(c.auc for c in cc) / max(len(cc), 1)
        eer = sum(c.eer for c in cc) / max(len(cc), 1)
        chips = [
            chip(f"{s['n']:,}", "vectors"), chip(str(s["k"]), "classes"),
            chip(str(s["dim"]), "dim"), chip(f"{s['top1']:.1%}", "top-1"),
            chip(f"{auc:.3f}", "macro AUC"), chip(f"{eer:.1%}", "mean EER"),
            chip(f"{sum(c.dcf for c in cc) / max(len(cc), 1):.3f}", "min DCF"),
            chip(f"{s['correct']:.1%}", "correct @ thr"),
            chip(f"{s['fr']:.1%}", "false reject"),
            chip(f"{s['fa']:.1%}", "false accept"),
        ]
        rows = state.sel_rows(an, v["sel"])
        if rows:
            chips.append(chip(f"{len(rows):,}", "selected", "gold"))
        return chips, composition(engine.hub, an, rows)
