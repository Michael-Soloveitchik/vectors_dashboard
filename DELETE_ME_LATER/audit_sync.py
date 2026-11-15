"""Sync audit (Playwright). Every macro action must end with the top-bar
indicator "in sync" (else the late panels are reported as FROZEN), and
then all panels must agree:

  thresholds   slider == distribution line == ROC operating point
  rejects      Σ(1-TPR_k)·n_k == hollow markers on the projection
  selection    badge == header chip == selected points on the projection
  classes      slider rows == distribution lanes == ROC markers
  theme        every figure's font colour matches the page theme
  focus        inspector open  ⇔  focus ring / inspected markers drawn

    python _runserver.py --port 8071 &
    python DELETE_ME_LATER/audit_sync.py http://127.0.0.1:8071/ [n_random=40] [seed=0]
"""
import random, re, sys, time
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8070/"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
problems, transients, errs, timings = [], [], [], []

STATE_JS = """() => {
  const g = id => { const e = document.querySelector('#' + id + ' .js-plotly-plot'); return e && e._fullData ? {data: e._fullData, layout: e.layout} : null; };
  const sc = g('fig-scatter'), roc = g('fig-roc'), di = g('fig-dist'), mx = g('fig-matrix');
  const out = {sync: document.querySelector('#sync').innerText, theme: document.documentElement.dataset.theme,
               badge: document.querySelector('#badge-selection').innerText,
               chips: [...document.querySelectorAll('#summary .chip')].map(c => c.innerText.replace(/\\n/g, ' ')),
               sliders: [...document.querySelectorAll('.thr-row')].map(r => [r.querySelector('.thr-name').title,
                         parseFloat(r.querySelector('[role=slider]').getAttribute('aria-valuenow'))]),
               drawer: document.querySelector('#inspector').className.includes('open'),
               colourBy: document.querySelector('#ctrl-color-by').innerText.trim(),
               composition: document.querySelector('#sel-summary').children.length,
               status: document.querySelector('#fit-status').innerText,
               fonts: [sc, roc, di, mx].map(p => p ? p.layout.font.color : null)};
  if (sc) {
    const cls = sc.data.filter(t => t.meta && t.meta.dyn && (t.meta.dyn.thr || []).includes('marker.symbol'));
    out.shown = cls.reduce((a, t) => a + t.x.length, 0);
    out.hollow = cls.reduce((a, t) => a + [].concat(t.marker.symbol).filter(s => s === 'circle-open').length, 0);
    out.selected = cls.reduce((a, t) => a + (t.selectedpoints ? t.selectedpoints.length : t.x.length), 0);
    out.counts = Object.fromEntries(sc.data.filter(t => t.mode === 'markers' && t.x.length === 1 && !(t.meta && t.meta.dyn) && /[(][0-9][0-9,]*[)]/.test(t.name))
                   .map(t => { const m = t.name.match(/^(.*?)\\s+\\(([\\d,]+)\\)/); return [m[1], parseInt(m[2].replace(/,/g, ''))]; }));
    const f = sc.data.find(t => t.name === 'focus'); out.focusPts = f ? f.x.length : 0;
    out.rawDomain = /raw/.test(((document.querySelector('#ctrl-score-domain label.selected') || {}).innerText) || '');
    const sg = document.querySelector('#fig-scatter .js-plotly-plot'), hm = sg.layout.meta && sg.layout.meta.home;
    if (hm) { const r = sg._fullLayout.xaxis.range, h = hm['xaxis.range']; out.zoom = (r[1] - r[0]) / (h[1] - h[0]); }
  }
  if (roc) { const op = roc.data.find(t => t.name === 'operating points'); out.ops = op ? op.customdata : null; }
  if (di && di.layout.yaxis.ticktext) {
    const th = di.data.find(t => t.name === 'threshold'); out.lanes = di.layout.yaxis.ticktext;
    out.lines = th ? th.x.filter((_, i) => i % 3 === 0) : null;
    const ins = di.data.find(t => t.name === 'inspected'); out.inspected = ins ? ins.x.length : 0;
  }
  return out; }"""


BUSY_JS = ("() => (window.__vd_pending || 0) > 0 || "
           "!document.getElementById('sync').className.includes('ok')")


def settle(pg, what, timeout=90):
    """End of the transaction: the action's requests were seen starting
    (or 0.8 s passed with none), then nothing is in flight and every
    panel is stamped."""
    t = time.time(); saw = False
    while time.time() - t < timeout:
        if pg.evaluate(BUSY_JS):
            saw = True
        elif saw or time.time() - t > 0.8:
            timings.append((time.time() - t, what)); return True
        time.sleep(0.03)
    problems.append(f"FROZEN after [{what}]: {pg.inner_text('#sync')}")
    return False


def check(pg, what):
    """A disagreement only counts if it PERSISTS: it is re-read after a
    pause, so a probe racing the last DOM update is not reported."""
    found = _check(pg, what)
    if found[1]:
        time.sleep(1.5); settle(pg, what + " (recheck)")
        again = _check(pg, what)
        problems.extend(again[1])
        transients.extend(m for m in found[1] if m not in again[1])
        return again[0]
    return found[0]


def _check(pg, what):
    s = pg.evaluate(STATE_JS)
    out = []
    bad = lambda msg: out.append(f"[{what}] {msg}")
    if "failed" in s["status"] or s["sliders"] == []:
        return s, out
    sl = dict(s["sliders"])
    if s.get("lanes") is not None and list(sl) != s["lanes"]:
        bad(f"classes: sliders {list(sl)[:3]}… vs lanes {s['lanes'][:3]}…")
    # (raw-score view: the lines sit where the thresholds fall among raw scores)
    if s.get("lines") and not s.get("rawDomain") and any(abs(a - b) > 1e-6 for a, b in zip(sl.values(), s["lines"])):
        bad(f"thr: sliders {list(sl.values())[:3]} vs dist lines {s['lines'][:3]}")
    if s.get("ops"):
        for k, thr, fpr, tpr in s["ops"]:
            if abs(sl.get(k, -1) - thr) > 1e-6:
                bad(f"thr: slider {k}={sl.get(k)} vs ROC op {thr}")
        if s["colourBy"] == "class" and s["counts"] and s.get("shown") == sum(s["counts"].values()):   # every vector drawn
            want = sum(round((1 - tpr) * s["counts"][k]) for k, _, _, tpr in s["ops"])
            if want != s["hollow"]:
                bad(f"rejects: ROC says {want} hollow, projection draws {s['hollow']}")
    n_badge = int(s["badge"].split()[0].replace(",", ""))
    chip = next((c for c in s["chips"] if "SELECTED" in c.upper()), None)
    n_chip = int(chip.split()[0].replace(",", "")) if chip else 0
    if n_badge != n_chip:
        bad(f"selection: badge {n_badge} vs header chip {n_chip}")
    if n_badge and s.get("counts") and s.get("shown") == sum(s["counts"].values()) and s["selected"] != n_badge:
        bad(f"selection: badge {n_badge} vs {s['selected']} selected points drawn")
    if bool(n_badge) != bool(s["composition"]):
        bad(f"selection: badge {n_badge} but composition rows {s['composition']}")
    if not n_badge and s.get("selected") != s.get("shown"):
        bad(f"selection: nothing selected but only {s['selected']}/{s['shown']} points lit")
    want_font = {"dark": "#e6e9ef", "light": "#1c2330"}[s["theme"]]
    if any(f not in (None, want_font) for f in s["fonts"]):
        bad(f"theme {s['theme']}: figure fonts {s['fonts']}")
    if s.get("zoom") and s["zoom"] > 2.2:
        bad(f"zoom: projection zoomed out to {s['zoom']:.2f}× the home view")
    if expect.get("thr"):                       # a ROC-curve click just happened
        lbl, want_thr = expect.pop("thr")
        if lbl in sl and abs(sl[lbl] - want_thr) > 2e-6:
            bad(f"roc pick: clicked thr {want_thr:.6f} of {lbl}, slider shows {sl[lbl]:.6f}")
    if s["drawer"] is False and (s.get("focusPts") or s.get("inspected")):
        bad("focus: inspector closed but focus markers still drawn")
    return s, out


import audit_actions as A                      # noqa: E402
from audit_actions import (ACTIONS, close_drawer, expect, lasso,  # noqa: E402,F401
                           pill, choose)

A.settle = settle
rng = A.rng                                    # one RNG for actions and plays
rng.seed(SEED)


PLAYS = int(sys.argv[sys.argv.index("--plays") + 1]) if "--plays" in sys.argv else 0
LEN = int(sys.argv[sys.argv.index("--len") + 1]) if "--len" in sys.argv else 10


def play(pg, k, seed):
    """One short session of a pretend user: LEN random actions, each a
    transaction that must settle and leave every panel in agreement."""
    rng.seed(seed)
    before = len(problems)
    names = [rng.choice(list(ACTIONS)) for _ in range(LEN)]
    for i, name in enumerate(names):
        if name not in ("close", "theme", "EER", "reset thr", "slider +", "slider -", "method", "matrix mode", "min DCF", "home view"):
            if "open" in pg.get_attribute("#inspector", "class"):
                close_drawer(pg); settle(pg, "close drawer")
        expect.clear()
        try:
            ACTIONS[name](pg)
        except Exception:
            continue                                   # not applicable in this state
        if settle(pg, f"play {k} · {i}:{name}"):
            check(pg, f"play {k} · {i}:{name}")
        else:
            pg.reload(wait_until="networkidle"); settle(pg, "reload")
    if k % 4 == 3:                                     # every 4th play ends in a burst
        close_drawer(pg); settle(pg, "pre-burst")
        for name in [rng.choice(["EER", "slider +", "method", "matrix mode", "curve", "theme", "centroids", "dist view", "lasso", "clear", "combine", "lane pick", "wheel zoom"]) for _ in range(10)]:
            try: ACTIONS[name](pg)
            except Exception: pass
        expect.clear()
        if settle(pg, f"play {k} · burst", timeout=150): check(pg, f"play {k} · burst")
    ok = len(problems) == before
    print(f"play {k:3d} seed {seed:4d} {'ok  ' if ok else 'FAIL'} {' → '.join(names)}"[:230], flush=True)
    return ok


with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1700, "height": 1000})
    pg.set_default_timeout(2500)          # an action that does not apply fails fast
    pg.on("console", lambda m: errs.append(f"{m.text[:160]} @ {m.location.get('url', '')[-60:]}:{m.location.get('lineNumber')}") if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append("pageerror " + (e.stack or str(e))[:400]))
    pg.goto(URL, wait_until="networkidle")
    settle(pg, "load"); check(pg, "load")
    if PLAYS:
        seed0 = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 1000
        passed = sum(play(pg, k, seed0 + k) for k in range(PLAYS))
        print(f"\nPLAYS: {passed}/{PLAYS} passed · {len(timings)} transactions · median {sorted(t for t, _ in timings)[len(timings)//2]:.2f}s")
        print("console errors:", len(errs), errs[:3])
        print(f"PROBLEMS: {len(problems)}   (transient probe races, gone on re-read: {len(transients)})")
        [print("  ", x) for x in problems[:30]]
        b.close(); sys.exit(1 if problems or errs else 0)
    for i in range(N):                                  # one action = one transaction
        name = rng.choice(list(ACTIONS))
        if name not in ("close", "theme", "EER", "reset thr", "slider +", "slider -", "method", "matrix mode"):
            if "open" in pg.get_attribute("#inspector", "class"):
                close_drawer(pg); settle(pg, "close drawer")
        try:
            ACTIONS[name](pg)
        except Exception as exc:
            print(f"  (action {name} not applicable: {type(exc).__name__})"); continue
        if settle(pg, f"{i}:{name}"): check(pg, f"{i}:{name}")
    close_drawer(pg); settle(pg, "pre-burst")
    burst = [rng.choice(["EER", "slider +", "method", "matrix mode", "curve", "theme", "centroids", "dist view", "lasso", "clear"]) for _ in range(14)]
    for name in burst:                                  # no waiting in between
        try: ACTIONS[name](pg)
        except Exception: pass
    if settle(pg, "burst " + ",".join(burst), timeout=150): check(pg, "burst")
    last = check(pg, "final")
    print("sample state:", {k: (v if not isinstance(v, (list, dict)) else (len(v), str(v)[:70]))
                            for k, v in last.items() if k in ("sync", "theme", "badge", "shown", "hollow", "selected", "ops", "lines", "sliders", "status")})
    slow = sorted(timings, reverse=True)[:3]
    print(f"{len(timings)} transactions settled · median {sorted(t for t, _ in timings)[len(timings)//2]:.2f}s · "
          f"slowest {[(round(t, 1), w) for t, w in slow]}")
    print("console errors:", len(errs), errs[:3])
    print(f"PROBLEMS: {len(problems)}   (transient probe races, gone on re-read: {len(transients)})")
    [print("  ", x) for x in problems[:25]]
    b.close()
sys.exit(1 if problems or errs else 0)
