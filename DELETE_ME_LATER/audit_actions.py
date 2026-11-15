"""Macro actions of a pretend researcher (random, seeded) for audit_sync.py.

``rng`` is reseeded by the audit per play; ``settle`` is injected by it
(the end-of-transaction wait); ``expect`` carries what an action expects
the next check to find (e.g. the threshold a ROC click must produce).
"""
import random
import time

rng = random.Random(0)
settle = None


def choose(pg, dd, text):
    pg.click(f"#{dd}"); time.sleep(.25)
    pg.locator(".dash-dropdown-option, .dash-options-list-option", has_text=text).last.click()
    time.sleep(.15); pg.keyboard.press("Escape")


def pill(pg, group, text):
    pg.locator(f"#{group} label", has_text=text).first.click()


def lasso(pg):
    b = pg.locator("#fig-scatter .nsewdrag").bounding_box()
    x, y = b["x"] + b["width"] * rng.uniform(.05, .5), b["y"] + b["height"] * rng.uniform(.05, .5)
    pg.mouse.move(x, y); pg.mouse.down()
    pg.mouse.move(x + b["width"] * .4, y + b["height"] * .45, steps=6); pg.mouse.up()


def click_point(pg):
    pts = pg.locator("#fig-scatter .scatterlayer .trace .points path")
    bb = pts.nth(rng.randrange(min(pts.count(), 200))).bounding_box()
    if bb: pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)


def tree_click(pg):
    pill(pg, "ctrl-matrix-mode", "tree"); settle(pg, "tree mode")
    pts = pg.locator("#fig-matrix .scatterlayer .trace").nth(1).locator(".points path")
    bb = pts.nth(rng.randrange(pts.count())).bounding_box()
    pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)


def inspect_select(pg):
    click_point(pg); settle(pg, "inspect")
    btn = pg.locator(".insp-details button", has_text=rng.choice(["select its class", "select neighbours"]))
    if btn.count(): btn.first.click()


def ctrl_click(pg):
    pts = pg.locator("#fig-scatter .scatterlayer .trace .points path")
    bb = pts.nth(rng.randrange(min(pts.count(), 200))).bounding_box()
    if bb:
        pg.keyboard.down("Control"); pg.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2); pg.keyboard.up("Control")


PICK_JS = """(seed) => {
  const gd = document.querySelector('#fig-roc .js-plotly-plot');
  const cur = gd.data.map((t, i) => [t, i]).filter(([t]) => t.mode === 'lines' && t.customdata && t.customdata.length > 5
      && !/^(pair|selection)/.test(t.customdata[0][0]));
  if (!cur.length) return null;
  const [t] = cur[seed % cur.length], i = 2 + (seed * 7) % (t.customdata.length - 4);
  const fl = gd._fullLayout, r = gd.getBoundingClientRect();
  return {label: t.customdata[i][0], thr: t.customdata[i][1],
          x: r.left + fl.xaxis._offset + fl.xaxis.c2p(t.x[i]), y: r.top + fl.yaxis._offset + fl.yaxis.c2p(t.y[i])};
}"""
expect = {}


def roc_pick(pg):
    """Click a point ON a class's ROC curve: its threshold must move there."""
    if "ROC" not in pg.locator("#ctrl-curve label.selected").inner_text():
        pill(pg, "ctrl-curve", "ROC"); settle(pg, "to ROC")
    t = pg.evaluate(PICK_JS, rng.randrange(10 ** 6))
    ok = t and all(isinstance(t[k], (int, float)) and t[k] == t[k] and abs(t[k]) < 1e5 for k in ("x", "y"))
    if ok:                                   # (NaN on a log axis at FPR 0)
        pg.mouse.click(t["x"], t["y"]); expect["thr"] = (t["label"], t["thr"])


OPS_JS = "() => { const gd = document.querySelector('#fig-roc .js-plotly-plot'); const t = gd._fullData.find(t => t.name === 'operating points'); if (!t || !t.x) return []; const fl = gd._fullLayout, r = gd.getBoundingClientRect(); return Array.from(t.x).map((x, i) => [r.left + fl.xaxis._offset + fl.xaxis.c2p(x), r.top + fl.yaxis._offset + fl.yaxis.c2p(t.y[i])]); }"


def roc_drag(pg):
    """Drag an operating point along its curve (sync.js → thr-pick)."""
    ops = [o for o in pg.evaluate(OPS_JS) if all(v == v and abs(v) < 1e5 for v in o)]
    if not ops:
        return
    x, y = rng.choice(ops)
    pg.mouse.move(x, y); pg.mouse.down()
    for k in range(1, rng.randint(3, 8)):
        pg.mouse.move(x + rng.uniform(2, 9) * k, y + rng.uniform(4, 16) * k); time.sleep(.1)
    pg.mouse.up()


def lane_pick(pg):
    b = pg.locator("#fig-dist .nsewdrag").bounding_box()
    pg.mouse.click(b["x"] + b["width"] * rng.uniform(.05, .7), b["y"] + b["height"] * rng.uniform(.1, .9))


def label_selection(pg):
    lasso(pg); settle(pg, "lasso for label")
    pg.fill("#ctrl-label-name", rng.choice(["good", "noisy", "outliers", "set A", ""]))
    pg.click("#btn-label-apply")


def wheel(pg):
    b = pg.locator("#fig-scatter .nsewdrag").bounding_box()
    pg.mouse.move(b["x"] + b["width"] * rng.uniform(.2, .8), b["y"] + b["height"] * rng.uniform(.2, .8))
    for _ in range(rng.randint(1, 12)):
        pg.mouse.wheel(0, rng.choice([-400, 500, 900])); time.sleep(.04)
    time.sleep(.4)


def slide(pg, n):
    pg.locator("#ctrl-thr-all [role=slider]").focus()
    for _ in range(abs(n)): pg.keyboard.press("ArrowRight" if n > 0 else "ArrowLeft")


def close_drawer(pg):
    if "open" in pg.get_attribute("#inspector", "class"): pg.click("#btn-inspector-close")


ACTIONS = {
    "EER": lambda pg: pg.click("#btn-thr-eer"),
    "reset thr": lambda pg: pg.click("#btn-thr-reset"),
    "slider +": lambda pg: slide(pg, rng.randint(5, 60)),
    "slider -": lambda pg: slide(pg, -rng.randint(5, 40)),
    "step": lambda pg: pg.locator(".step").nth(rng.randrange(pg.locator(".step").count())).click(),
    "method": lambda pg: choose(pg, "ctrl-methodology", rng.choice(["NCM", "KNN", "Cosine", "ACPAN", "VMFMixture", "ArcCosineSVM"])),
    "model": lambda pg: choose(pg, "ctrl-model", rng.choice(["PI_2_1024", "PI_2_256", "S2_256"])),
    "lasso": lasso,
    "clear": lambda pg: pg.click("#btn-sel-clear"),
    "click point": click_point,
    "close": close_drawer,
    "matrix mode": lambda pg: pill(pg, "ctrl-matrix-mode", rng.choice(["pair scores", "centroids", "confusion", "tree"])),
    "tree click": tree_click,
    "combine": lambda pg: pill(pg, "ctrl-sel-mode", rng.choice(["replace", "add", "intersect"])),
    "list click": lambda pg: pg.locator(".sel-item").nth(0).click(timeout=1500),
    "min DCF": lambda pg: pg.click("#btn-thr-dcf"),
    "key x": lambda pg: (pg.locator("#sync").click(), pg.keyboard.press("x")),
    "ctrl click": ctrl_click,
    "roc pick": roc_pick, "lane pick": lane_pick, "label selection": label_selection,
    "unlabel all": lambda pg: pg.click("#btn-label-clear"),
    "home view": lambda pg: pg.click("#btn-home-all"),
    "wheel zoom": wheel,
    "sort by my_label": lambda pg: choose(pg, "ctrl-matrix-sort", "my_label"),
    "colour my_label": lambda pg: choose(pg, "ctrl-color-by", "my_label"),
    "classes = my_label": lambda pg: (pg.locator("#ctrl-labels .dash-dropdown-clear").click(), time.sleep(.3), choose(pg, "ctrl-labels", rng.choice(["my_label", "device_id"]))),
    "shift drag": lambda pg: (pg.keyboard.down("Shift"), lasso(pg), pg.keyboard.up("Shift")),
    "dbl click": lambda pg: (lambda b: pg.mouse.dblclick(b["x"] + b["width"] * .97, b["y"] + b["height"] * .97))(pg.locator("#fig-scatter .nsewdrag").bounding_box()),
    "home": lambda pg: pg.locator(".card-scatter .home-btn").click(),
    "colour by": lambda pg: choose(pg, "ctrl-color-by", rng.choice(["class", "speaker_id", "device_id", "own-class score"])),
    "inspect+select": inspect_select,
    "labels": lambda pg: (pg.locator("#ctrl-labels .dash-dropdown-clear").click(), time.sleep(.3), choose(pg, "ctrl-labels", rng.choice(["device_id", "speaker_id"]))),
    "curve": lambda pg: pill(pg, "ctrl-curve", rng.choice(["ROC", "DET"])),
    "dist view": lambda pg: pill(pg, "ctrl-dist-view", rng.choice(["histogram", "samples", "box"])),
    "score domain": lambda pg: pill(pg, "ctrl-score-domain", rng.choice(["normalized", "raw"])),
    "reduction": lambda pg: pill(pg, "ctrl-reduction", rng.choice(["PCA", "PCA", "t-SNE"])),
    "p-t-SNE train": lambda pg: (pill(pg, "ctrl-reduction", "p-t-SNE"), time.sleep(.5), pg.click("#btn-ptsne-train")),
    "click mode": lambda pg: pill(pg, "ctrl-click-mode", rng.choice(["inspect", "select"])),
    "ellipses": lambda pg: pg.locator("#ctrl-scatter-opts label", has_text="ellipses").first.click(),
    "roc drag": lambda pg: roc_drag(pg),
    "full screen": lambda pg: (lambda b: (b.click(), time.sleep(.7), pg.keyboard.press("Escape") if rng.random() < .5 else b.click()))(pg.locator(".max-btn").nth(rng.randrange(4))),
    "theme": lambda pg: pg.click("#btn-theme"),
    "centroids": lambda pg: (pg.fill("#ctrl-n-centroids", str(rng.choice([1, 2]))), pg.keyboard.press("Enter")),
}
