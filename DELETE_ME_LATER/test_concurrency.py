"""Sandbox test: threading model of the engine — single-flight, gate,
latest-wins, determinism under load, cache bounds, thread / fd leaks.

    OMP_NUM_THREADS=2 nice python DELETE_ME_LATER/test_concurrency.py
"""
import os, sys, threading, time, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
os.environ.setdefault("DLOG_PRINT", "0")
import numpy as np
import psutil
from frontend import Gui
from frontend.engine import Engine, FitSpec, DataHub
from frontend.engine import analysis as A, curves, matrix, project, state
from frontend.engine.concurrency import Superseded, HeavyGate, FlightCache
from backend.representations import METHOD_REGISTRY
from data.generate import load_and_enrich

proc = psutil.Process()
rss = lambda: proc.memory_info().rss / 1e6
fds = lambda: len(os.listdir("/proc/self/fd"))
df, _ = load_and_enrich()
gui = Gui(df); hub = gui.hub
M = hub.models[0]
spec = lambda m="NCM", n=1, model=M: FitSpec(model, ("device_id",), m, "mean", n_global=n)

# count real fits + peak concurrency inside the heavy section
real_fit = A.fit_analysis; calls = []; live = [0]; peak = [0]; mu = threading.Lock()
def counted(s, f):
    with mu: live[0] += 1; peak[0] = max(peak[0], live[0]); calls.append(s)
    try:
        time.sleep(0.05); return real_fit(s, f)
    finally:
        with mu: live[0] -= 1
A.fit_analysis = counted

# 1 ── single-flight: 16 threads, one new spec → exactly one fit, one object
eng = Engine(hub, max_entries=4, heavy_slots=2)
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(lambda _: eng.analysis(spec()), range(16)))
assert len(calls) == 1 and all(r is res[0] for r in res), (len(calls), eng.fits.stats)
print("1 single-flight      fits=1 for 16 callers ·", eng.fits.stats)

# 2 ── different specs run in parallel, bounded by the gate
calls.clear(); peak[0] = 0
specs = [spec(m) for m in ("Cosine", "KNN", "CosFace", "ArcCosineKernel")]
t = time.time()
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda i: eng.analysis(specs[i % 4]), range(16)))
assert len(calls) == 4 and peak[0] <= 2, (len(calls), peak[0])
print(f"2 parallel + gate     4 fits, peak concurrency {peak[0]} ≤ 2 slots · {time.time()-t:.2f}s")

# 3 ── latest-wins: queued fits nobody wants are skipped; a joiner still gets its result
eng1 = Engine(hub, max_entries=8, heavy_slots=1); calls.clear()
hold = threading.Event(); started = threading.Event()
def blocker(s, f):
    started.set(); hold.wait(); return real_fit(s, f)
A.fit_analysis = blocker
th0 = threading.Thread(target=lambda: eng1.analysis(spec("Cosine"))); th0.start(); started.wait()
A.fit_analysis = counted
out = {}
def ask(name, s, client):
    try: out[name] = eng1.analysis(s, client=client)
    except Superseded: out[name] = "superseded"
ths = []
for name, s, cl in [("A", spec("NCM", 2), "tab1"), ("B", spec("NCM", 3), "tab1"), ("C", spec("NCM", 4), "tab1"),
                    ("A-anon", spec("NCM", 2), None)]:
    th = threading.Thread(target=ask, args=(name, s, cl)); th.start(); ths.append(th); time.sleep(0.05)
hold.set(); th0.join(); [th.join() for th in ths]
assert out["A"] == "superseded" and out["B"] == "superseded", out
assert out["C"].spec.n_global == 4 and out["A-anon"].spec.n_global == 2, out
print("3 latest-wins         A,B dropped · C fitted · anonymous joiner of A still served ·", eng1.gate.stats)

# 4 ── determinism under load: 12 threads × mixed reads == single-thread reference
A.fit_analysis = real_fit
eng = Engine(hub, max_entries=4, heavy_slots=2)
sp = [spec("NCM"), spec("KNN"), spec("VMFMixture", 2)]
thr = lambda an, v: {l: v for l in an.labels}
def probe(i):
    an = eng.analysis(sp[i % 3]); a = [1.0, 0.5, 0.0][i % 3]; rows = np.arange(i % 7, 200, 3)
    pr = project.project(eng, an.frame, "PCA", {}, 4000)
    return (an.norm(a).sum(), an.decide(thr(an, .6), a).pred.sum(), an.summary(thr(an, .6), a)["fa"],
            sum(c.auc for c in curves.class_curves(an, a)), sum(c.auc for c in curves.pair_curves(an, a)),
            matrix.pair_matrix(an, matrix.order_rows(an, rows, ["class", "hierarchical"])).sum(), pr.xy.sum(),
            tuple(r for r, _ in state.neighbours(an, i % 300)))
ref = [probe(i) for i in range(21)]
with ThreadPoolExecutor(12) as ex:
    got = list(ex.map(probe, [i % 21 for i in range(252)]))
assert all(np.allclose(g[:7], ref[i % 21][:7]) and g[7] == ref[i % 21][7] for i, g in enumerate(got))
print("4 determinism         252 concurrent mixed reads == reference (no cross-talk)")

# 5 ── shared arrays are read-only
an = eng.analysis(sp[0])
for name, arr in [("raw", an.raw), ("norm", an.norm(1.0)), ("Vn", an.frame.Vn), ("true_idx", an.true_idx)]:
    try: arr[0] = 0; raise SystemExit(f"{name} is writable!")
    except ValueError: pass
print("5 immutability        raw / norm / Vn / true_idx are read-only")

# 6 ── cache bounds + memory: cycle every methodology × 21 alphas × models
r0 = rss(); eng = gui.engine
for rnd in range(2):
    for model in hub.models:
        for m in sorted(METHOD_REGISTRY):
            an = eng.analysis(FitSpec(model, ("device_id",), m, "mean"))
            if not an.ok: continue
            for a in np.linspace(0, 1, 21):
                an.norm(a); curves.class_curves(an, a); curves.pair_curves(an, a)
            for k in range(6):
                matrix.pair_matrix(an, matrix.order_rows(an, np.arange(k, min(300, an.frame.n), 2), ["hierarchical"]))
            sizes = an.memo.sizes()
            assert all(v <= A.MEMO_CAPS.get(k, 3) for k, v in sizes.items()), sizes
    if rnd == 0: r1 = rss()
st = eng.stats()
assert st["fits"]["size"] <= 4 and st["frames"] <= 3, st
print(f"6 bounds              fits={st['fits']['size']}≤4 frames={st['frames']}≤3 memo={sizes} · "
      f"RSS start {r0:.0f}MB → round1 {r1:.0f}MB → round2 {rss():.0f}MB (Δround2 {rss()-r1:+.0f}MB)")
assert rss() - r1 < 60, "memory keeps growing between identical rounds"

# 7 ── server lifecycle: 6 × (new Gui → run → stop), threads and fds stay flat
import importlib, urllib.request
t0, f0 = threading.active_count(), fds()
for i in range(6):
    for m in [m for m in sys.modules if m.split(".")[0] == "frontend"]: del sys.modules[m]   # notebook reload
    from frontend import Gui as G
    g = G(df); g.run(port=8079, mode="external")
    assert urllib.request.urlopen(g.url).status == 200
    g.stop()
time.sleep(1.5)
print(f"7 lifecycle           threads {t0} → {threading.active_count()} · fds {f0} → {fds()} after 6 reload+run+stop cycles")
assert threading.active_count() <= t0 + 1 and fds() <= f0 + 3
print("ALL OK")
