"""Sandbox test: awkward data and state cases through the engine AND every
figure builder / inspector view (no browser).

    OMP_NUM_THREADS=2 nice python DELETE_ME_LATER/test_cases.py
"""
import os, sys, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
os.environ.setdefault("DLOG_PRINT", "0")
import numpy as np, pandas as pd
from frontend import Gui, figures as F, inspector as insp
from frontend.engine import curves, matrix, project, state

rng = np.random.default_rng(0)
fails = []


def blobs(n_per, d=24, sep=3.0):
    X, y = [], []
    for k, n in enumerate(n_per):
        c = rng.normal(size=d) * sep
        X.append(c + rng.normal(size=(n, d))); y += [f"c{k}"] * n
    return np.vstack(X), np.array(y)


def render_all(gui, an, thr, sel=None, focus=None, tag=""):
    """Every panel, both themes, the views with the most branches."""
    eng = gui.engine
    rows = state.sel_rows(an, sel)
    frows, flabels = state.focus_parts(an, focus)
    d = an.decide(thr, 0.7)
    for theme in ("dark", "light"):
        for red, prm in (("PCA", {}), ("t-SNE", {"perplexity": 5.0})):
            pr = project.project(eng, an.frame, red, prm, 300)
            F.scatter_fig(an, pr, d, theme=theme, alpha=0.7, sel_rows=rows,
                          focus_rows=frows, focus_labels=flabels,
                          neighbour_rows=[r for r, _ in state.neighbours(an, frows[0])] if len(frows) == 1 else [],
                          show_hull=True, show_density=True).to_plotly_json()
        kw = dict(theme=theme, abs_on=True, power=0.5, auto_range=True, lo=-1, hi=1)
        for keys in (["class"], ["hierarchical"], ["random", "class"], ["session_id"], ["nope"], []):
            F.vectors_fig(an, matrix.order_rows(an, matrix.pick_rows(an, rows, cap=60), keys), focus_rows=frows, **kw).to_plotly_json()
        F.classes_fig(an, **kw).to_plotly_json(); F.confusion_fig(an, d, rows, theme=theme).to_plotly_json()
        F.tree_fig(an, theme=theme).to_plotly_json()
        for col in [c_ for c_ in an.frame.df.columns if c_ not in ("session_id",)][:2] + ["nope", F.OWN_SCORE]:
            F.scatter_fig(an, project.project(eng, an.frame, "PCA", {}, 300), d, theme=theme, alpha=0.7,
                          color_by=col).to_plotly_json()
        insp.composition(gui.hub, an, rows)
        for det in (False, True):
            for show in ("all", "class", "pair"):
                F.roc_fig(an, alpha=0.7, thr=thr, theme=theme, sel_rows=rows, det=det, show=show,
                          log_x=True, log_y=True, focus_labels=flabels).to_plotly_json()
        for view in ("hist", "strip"):
            F.dist_fig(an, alpha=0.7, thr=thr, theme=theme, view=view, sel_rows=rows, focus_rows=frows).to_plotly_json()
    an.summary(thr, 0.7); curves.eer_thresholds(an, 0.7)
    if frows: insp.rows_view(an, frows, thr, 0.7); insp.audio_view(gui.hub, an, frows)
    if flabels: insp.labels_view(an, flabels, thr, 0.7)


def case(name, fn):
    try:
        out = fn(); print(f"ok   {name}" + (f"  → {out}" if out else ""))
    except Exception as exc:
        fails.append(name); print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)


def full(X, y, methods=("NCM", "KNN", "VMFMixture", "ACPAN", "SRC", "LabelPropagation", "ArcCosineSVM"), **kw):
    gui = Gui.from_arrays(X, {"label": y}, **kw)
    m0 = gui.models[0]; notes = []
    for m in methods:
        an = gui.analysis(methodology=m)
        if not an.ok:
            notes.append(f"{m}: {an.error[:60]}"); continue
        thr = {l: 0.6 for l in an.labels}; thr["ghost"] = 0.9; thr[an.labels[0]] = None
        render_all(gui, an, thr, tag=m)
        render_all(gui, an, {}, sel={"model": m0, "mode": "single", "seq": 1, "clauses": {"lasso": {"rows": [0, 1, 2, 10**9, -4, "x"]}}},
                   focus={"model": m0, "rows": [1]})
        render_all(gui, an, thr, sel={"model": m0, "mode": "intersect", "seq": 2, "clauses": {"classes": {"classes": an.labels[:1] + ["ghost"], "label_keys": ["label"]}, "rows": {"rows": list(range(40))}}},
                   focus={"model": m0, "rows": [0, 2]})
        render_all(gui, an, thr, sel={"model": "other-model", "mode": "union", "clauses": {"lasso": {"rows": [0]}}},
                   focus={"model": m0, "labels": an.labels[:2] + ["ghost"]})
    return "; ".join(notes)


X, y = blobs([30, 25, 20])
case("3 balanced classes, all methods", lambda: full(X, y))
X2, y2 = blobs([40, 1, 2, 12])
case("classes with 1 and 2 members", lambda: full(X2, y2))
case("exactly 2 classes", lambda: full(*blobs([15, 15]), methods=("NCM", "KNN", "Cosine")))
Xz = X.copy(); Xz[3] = 0.0
case("a zero vector", lambda: full(Xz, y, methods=("NCM", "KNN")))
case("integer labels", lambda: full(X, np.repeat([0, 1, 2], [30, 25, 20]), methods=("NCM",)))
case("labels with separators / html", lambda: full(X, np.where(y == "c0", "a  ·  b #2 <b>x</b>", np.where(y == "c1", "α | β", "plain")), methods=("NCM", "KNN")))
case("tiny dim (D=2)", lambda: full(*blobs([20, 20, 20], d=2), methods=("NCM", "KNN")))
case("12 small classes, 4 centroids each", lambda: (lambda g: [render_all(g, g.analysis(n_centroids=4), {})] and None)(
    Gui.from_arrays(*[(a, {"label": b}) for a, b in [blobs([6] * 12)]][0])))


def errors():
    out = []
    g = Gui.from_arrays(X, {"label": y, "one": np.full(len(y), "same")})
    for kw in (dict(labels=["one"]), dict(labels=["missing"]), dict(methodology="Nope"), dict(skip=["c0", "c1"])):
        an = g.analysis(**kw); assert not an.ok and an.error, kw; out.append(an.error[:34])
    an = g.analysis(skip=["c2"]); assert an.ok and an.labels == ["c0", "c1"]
    render_all(g, an, {"c0": 0.5}, focus={"model": g.models[0], "rows": [70]})      # focus on a skipped-class row
    for bad in (dict(vectors=np.zeros((3,)), labels=[1, 2, 3]), dict(vectors=X, labels=y[:5])):
        try: Gui.from_arrays(**bad); raise AssertionError("accepted bad input")
        except ValueError: pass
    Xn = X.copy(); Xn[0, 0] = np.nan
    try: Gui.from_arrays(Xn, y); raise AssertionError("NaN vectors accepted")
    except ValueError as e: out.append(str(e)[:44])
    try: Gui(pd.DataFrame({"a": [1]})); raise AssertionError
    except ValueError as e: out.append(str(e)[:30])
    return " | ".join(out)
case("error states are values, not crashes", errors)


def multi():
    df = pd.DataFrame({"a": np.repeat(["x", "y"], 30), "b": np.tile(["p", "q", "r"], 20), "num": np.arange(60) % 4})
    Xm, _ = blobs([60], d=16); df["centroid"] = list(Xm)
    g = Gui(df, labels=["a", "b"])
    an = g.analysis(); assert an.ok and an.k == 6, (an.error, an.k)
    render_all(g, an, {l: 0.5 for l in an.labels})
    assert g.hub.label_columns(g.models[0]) == ["a", "b", "num"], g.hub.label_columns(g.models[0])
    return f"{an.k} joint classes"
case("joint labels from two columns, no ids / model column", multi)

print("\nFAILS:", fails or 0)
sys.exit(1 if fails else 0)
