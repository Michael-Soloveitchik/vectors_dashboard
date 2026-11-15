"""Sandbox test: engine on synthetic (all methodologies) + real (scale)."""
import sys, time, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
import numpy as np
# import engine without frontend/__init__ (gui not needed here)
import types
pkg = types.ModuleType("frontend"); pkg.__path__ = [str(ROOT / "code/frontend")]
sys.modules["frontend"] = pkg
from frontend.engine import DataHub, Engine, FitSpec
from frontend.engine import curves, matrix, project
from backend.representations import METHOD_REGISTRY
from data.generate import load_and_enrich

df, _ = load_and_enrich()
hub = DataHub(df); eng = Engine(hub, max_entries=20)
model = hub.models[0]
print("models", hub.models, "labels", hub.label_columns(model), hub.default_labels(model))
fails = 0
for m in sorted(METHOD_REGISTRY):
    t = time.time()
    an = eng.analysis(FitSpec(model, ("device_id",), m, "mean"))
    if not an.ok:
        print(f"FAIL {m}: {an.error}"); fails += 1; continue
    P = an.norm(1.0)
    assert P.shape == an.raw.shape and np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1
    # monotone: raising thresholds never increases accepted count
    acc = [an.decide({l: t_ for l in an.labels}, 1.0).accept.sum() for t_ in (0, .3, .6, .9, 1.0)]
    assert all(a >= b for a, b in zip(acc, acc[1:])), acc
    d0 = an.decide({}, 1.0)
    assert (d0.pred >= 0).all(), "thr=0 must accept everything"
    assert (d0.pred == an.argmax()).all()
    cc = curves.class_curves(an, 1.0); pc = curves.pair_curves(an, 1.0)
    ops = curves.operating_points(an, {l: 0.5 for l in an.labels}, 1.0)
    for c in cc:   # op point must sit on the curve's hull (fpr/tpr in range)
        f, tp, _ = ops[c.key]; assert 0 <= f <= 1 and 0 <= tp <= 1
    s = an.summary({}, 1.0)
    print(f"ok  {m:18s} K={an.k} top1={s['top1']:.3f} "
          f"AUC={np.mean([c.auc for c in cc]):.3f} EER={np.mean([c.eer for c in cc]):.3f} "
          f"pair={[round(c.auc,3) for c in pc]} cents={sum(v.shape[0] for v in an.centroids.values())} "
          f"{time.time()-t:.2f}s")
an = eng.analysis(FitSpec(model, ("device_id",), "NCM", "mean", n_global=2))
pr = project.project(eng, an.frame, "PCA", {}, 5000)
cxy = project.centroid_xy(an, pr)
# centroid markers must land inside their own cluster's bbox
for lbl, j, x, y in cxy:
    pts = pr.xy[an.true_idx[pr.rows] == an.labels.index(lbl)]
    lo, hi = pts.min(0), pts.max(0)
    assert lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1], (lbl, j, x, y, lo, hi)
print("centroids inside clusters:", len(cxy))
h = project.accepted_hulls(an, pr, an.decide({}, 1.0).own_ok); print("hulls", len(h))
rows = matrix.order_rows(an, matrix.pick_rows(an, None), ["class", "hierarchical"])
M = matrix.pair_matrix(an, rows); assert M.shape == (len(rows),) * 2
assert np.allclose(M, M.T, atol=1e-8)
Mc, names = matrix.centroid_matrix(an); print("centroid matrix", Mc.shape, names[:3])
cnt, r, c = matrix.confusion(an, an.decide({l: .9 for l in an.labels}, 1.0)); print("confusion\n", cnt)
assert cnt.sum() == (an.true_idx >= 0).sum()
print("skip test:", eng.analysis(FitSpec(model, ("device_id",), "NCM", "mean", skip=("device_00",))).labels)
print("FAILS", fails)
if "--real" in sys.argv:
    from data.real_data import load_real, available_models, WAVS_DIR
    df = load_real("idrnd_redimnet_b3")
    hub = DataHub(df, loader=load_real, extra_models=available_models(), wav_dir=WAVS_DIR)
    eng = Engine(hub)
    print(hub.models)
    for mdl in ["idrnd_redimnet_b3", "nvidia_titanet_small"]:
        t = time.time(); fr = hub.frame(mdl); print(mdl, fr.V.shape, f"load {time.time()-t:.1f}s", hub.label_columns(mdl))
        t = time.time(); an = eng.analysis(FitSpec(mdl, tuple(hub.default_labels(mdl)), "NCM", "mean")); print(" fit", an.error, an.k, f"{time.time()-t:.1f}s")
        t = time.time(); curves.class_curves(an, 1.0); print(f" class curves {time.time()-t:.1f}s")
        t = time.time(); curves.pair_curves(an, 1.0); print(f" pair curves {time.time()-t:.1f}s")
        t = time.time(); pr = project.project(eng, fr, "PCA", {}, 4000); print(f" pca {time.time()-t:.1f}s", pr.xy.shape)
        t = time.time(); rows = matrix.order_rows(an, matrix.pick_rows(an, None), ["class"]); M = matrix.pair_matrix(an, rows); print(f" matrix {time.time()-t:.1f}s", M.shape)
        t = time.time(); print(an.summary({l: .5 for l in an.labels}, 1.0), f"{time.time()-t:.2f}s")
    t = time.time(); print("wav", hub.wav_path(hub.frame("idrnd_redimnet_b3").sids[0]), f"{time.time()-t:.1f}s")
