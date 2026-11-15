/* Transaction indicator. "in sync" means BOTH:
 *   - no Dash callback request is in flight, and
 *   - every panel's rendered stamp equals the wanted stamp
 *     (window.__vd_late, maintained by callbacks/panels.py),
 * for two consecutive ticks. Owned here (not a Dash output) so it can
 * follow request activity, which Dash does not expose as a prop. */
(function () {
  const pending = new Set();
  let seq = 0, calm = 0;
  const orig = window.fetch;
  window.fetch = function (input) {
    const url = (typeof input === 'string') ? input : ((input && input.url) || '');
    if (url.indexOf('_dash-update-component') === -1) { return orig.apply(this, arguments); }
    const k = ++seq;
    pending.add(k);
    const done = function () { pending.delete(k); };
    return orig.apply(this, arguments).then(
      function (r) { done(); return r; },
      function (e) { done(); throw e; });
  };
  document.addEventListener('keydown', function (e) {   // Esc: close inspector · x: clear selection
    const d = document.getElementById('inspector');
    if (e.key === 'Escape' && d && d.className.indexOf('open') !== -1) {
      const b = document.getElementById('btn-inspector-close');
      if (b) { b.click(); }
    }
    const tag = (e.target && e.target.tagName) || '';
    const typing = tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable);
    if (e.key === 'x' && !typing && !e.ctrlKey && !e.metaKey && !e.altKey) {
      const c = document.getElementById('btn-sel-clear');     // x = clear selection
      if (c) { c.click(); }
    }
  });
  // Reset zoom: the ⌂ button in a card header, or right-click on a plot.
  // (In select mode Plotly's double-click clears the selection instead.)
  function resetZoom(card) {
    const gd = card && card.querySelector('.js-plotly-plot');
    if (!gd) { return; }
    const meta = gd.layout && gd.layout.meta;          // the figure's own home view
    if (meta && meta.home && window.Plotly) { window.Plotly.relayout(gd, meta.home); return; }
    const btn = gd.querySelector('.modebar-btn[data-attr="zoom"][data-val="reset"]');
    if (btn) { btn.click(); return; }
    if (window.Plotly) { window.Plotly.relayout(gd, {'xaxis.autorange': true, 'yaxis.autorange': true}); }
  }
  document.addEventListener('click', function (e) {
    const b = e.target.closest && e.target.closest('.home-btn');
    if (b) { resetZoom(b.closest('.card')); }
    if (e.target.closest && e.target.closest('#btn-home-all')) {
      // "Back to start" for the VIEW: every zoom, the selection, the
      // inspector. (Thresholds and the method are analysis state — they
      // have their own reset buttons.)
      document.querySelectorAll('.card').forEach(resetZoom);
      const d = document.getElementById('inspector');
      if (d && d.className.indexOf('open') !== -1) { document.getElementById('btn-inspector-close').click(); }
      const c = document.getElementById('btn-sel-clear');
      if (c) { c.click(); }
    }
  });
  document.addEventListener('contextmenu', function (e) {
    const g = e.target.closest && e.target.closest('.card .js-plotly-plot');
    if (g) { e.preventDefault(); resetZoom(g.closest('.card')); }
  });

  // Mouse roles need the modifier keys, which Plotly's click / selection
  // events do not carry: record them on mouse-down (before those events).
  // After ANY drag on the projection, also ask the server to re-assert the
  // store's selection: Plotly dims points on its own during a drag, and if
  // its handler is interrupted (figure redrawn mid-drag) no selection event
  // follows — the picture would stay out of step with the store.
  const setProps = function (id, props) {
    if (window.dash_clientside && window.dash_clientside.set_props) {
      window.dash_clientside.set_props(id, props);
    }
  };
  let down = null, lastUp = 0;
  document.addEventListener('mousedown', function (e) {
    const plot = e.target.closest && e.target.closest('.js-plotly-plot');
    if (plot && e.button === 0) {
      setProps('store-mods', {data: {ctrl: e.ctrlKey || e.metaKey, shift: e.shiftKey, t: Date.now()}});
    }
    down = (e.button === 0 && e.target.closest && e.target.closest('#fig-scatter')) ? [e.clientX, e.clientY] : null;
  }, true);
  document.addEventListener('mouseup', function (e) {
    if (!down) { return; }
    const moved = Math.abs(e.clientX - down[0]) + Math.abs(e.clientY - down[1]) > 4;
    down = null;
    if (moved) { setTimeout(function () { setProps('lasso-echo', {data: Date.now()}); }, 350); return; }
    // Double-click on the projection clears the selection. Detected here:
    // Plotly covers the page with its own layer while the button is down
    // (a DOM "dblclick" never reaches the plot) and only emits "deselect"
    // while it holds a selection outline, which the store-owned selection
    // never leaves behind.
    const now = Date.now();
    if (now - lastUp < 380) {
      const c = document.getElementById('btn-sel-clear');
      if (c) { c.click(); }
      lastUp = 0;
    } else { lastUp = now; }
  }, true);

  setInterval(function () {
    window.__vd_pending = pending.size;
    // Plotly keeps the index of the "active" selection outline; when the
    // outline is removed programmatically (class picker, clear) the index
    // goes stale and Plotly's own drag handler throws on the next drag.
    const gd = document.querySelector('#fig-scatter .js-plotly-plot');
    const fl = gd && gd._fullLayout;
    if (fl && fl._activeSelectionIndex !== undefined &&
        fl._activeSelectionIndex >= (fl.selections || []).length) {
      delete fl._activeSelectionIndex;
    }
    const el = document.getElementById('sync');
    if (!el) { return; }
    const late = window.__vd_late;
    const busy = pending.size > 0 || !late || late.length > 0;
    calm = busy ? 0 : calm + 1;
    const ok = calm >= 2;
    const text = ok ? '● in sync'
      : (!late ? '◌ loading'
         : (late.length ? '◌ updating ' + late.join(' · ') : '◌ updating'));
    if (el.textContent !== text) { el.textContent = text; }
    const cls = 'sync ' + (ok ? 'ok' : 'busy');
    if (el.className !== cls) { el.className = cls; }
  }, 100);
})();

/* Audio dock: fold / unfold, and WavLens follows the page theme (a cookie
 * for the next frame the server renders, postMessage for the open ones). */
(function () {
  document.addEventListener('click', function (e) {
    const b = e.target.closest && e.target.closest('.dock-toggle');
    if (!b) { return; }
    const d = b.closest('.dock');
    const folded = d.classList.toggle('folded');
    b.textContent = folded ? '▴' : '▾';
  });
  function pushTheme() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t !== 'light' && t !== 'dark') { return; }
    try { document.cookie = 'vd_theme=' + t + ';path=/;max-age=31536000;samesite=lax'; } catch (e) {}
    document.querySelectorAll('iframe.wavlens').forEach(function (f) {
      try { f.contentWindow.postMessage({wavlens_theme: t}, '*'); } catch (e) {}
    });
  }
  new MutationObserver(pushTheme).observe(document.documentElement,
    {attributes: true, attributeFilter: ['data-theme']});
  document.addEventListener('load', function (e) {     // a new audio frame
    if (e.target && e.target.classList && e.target.classList.contains('wavlens')) { pushTheme(); }
  }, true);
})();

/* ROC: drag an operating point along its curve — that class's threshold
 * follows (through the thr-pick store, like a click on the curve), and with
 * it every panel. A press without movement inspects that class. */
(function () {
  let drag = null, last = 0;
  const plot = function () { return document.querySelector('#fig-roc .js-plotly-plot'); };
  function near(gd, t, e, maxd) {          // index of the trace point closest to the cursor
    const fl = gd._fullLayout, r = gd.getBoundingClientRect();
    let best = -1, bd = maxd;
    for (let i = 0; i < t.x.length; i++) {
      const d = Math.hypot(r.left + fl.xaxis._offset + fl.xaxis.c2p(t.x[i]) - e.clientX,
                           r.top + fl.yaxis._offset + fl.yaxis.c2p(t.y[i]) - e.clientY);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }
  document.addEventListener('mousedown', function (e) {
    if (e.button !== 0 || e.ctrlKey || e.metaKey || e.shiftKey) { return; }
    const gd = plot();
    if (!gd || !gd._fullData || !gd.contains(e.target)) { return; }
    const ops = gd._fullData.find(function (t) { return t.name === 'operating points'; });
    const i = ops && ops.x ? near(gd, ops, e, 14) : -1;
    if (i < 0) { return; }
    drag = {label: ops.customdata[i][0], moved: false, x: e.clientX, y: e.clientY};
    e.stopPropagation(); e.preventDefault();
  }, true);
  document.addEventListener('mousemove', function (e) {
    if (!drag) { return; }
    if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 4) { return; }
    drag.moved = true; document.body.classList.add('vd-dragging');
    const now = Date.now();
    if (now - last < 80) { return; }
    const gd = plot();
    const curve = gd._fullData.find(function (t) {
      return t.mode === 'lines' && t.customdata && t.customdata.length &&
             t.customdata[0].length === 2 && t.customdata[0][0] === drag.label; });
    const i = curve ? near(gd, curve, e, Infinity) : -1;
    if (i < 0) { return; }
    last = now;
    window.dash_clientside.set_props('thr-pick', {data: {label: drag.label, thr: curve.customdata[i][1], seq: now}});
  }, true);
  document.addEventListener('mouseup', function (e) {
    if (!drag) { return; }
    const d = drag; drag = null;
    document.body.classList.remove('vd-dragging');
    e.stopPropagation();
    if (!d.moved) {                         // a plain click on the spot
      window.dash_clientside.set_props('op-pick', {data: {label: d.label, seq: Date.now()}});
    }
  }, true);
})();

/* Full screen: the ▢ button turns one panel into a big window; Esc (or the
 * button again) puts it back. Plotly follows the new size by itself. */
(function () {
  function restore() {
    document.querySelectorAll('.card.max').forEach(function (c) { c.classList.remove('max'); });
    document.body.classList.remove('vd-max');
  }
  document.addEventListener('click', function (e) {
    const b = e.target.closest && e.target.closest('.max-btn');
    if (!b) { return; }
    const card = b.closest('.card'), on = !card.classList.contains('max');
    restore();
    if (on) { card.classList.add('max'); document.body.classList.add('vd-max'); }
    window.dispatchEvent(new Event('resize'));
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && document.querySelector('.card.max')) {
      restore(); window.dispatchEvent(new Event('resize'));
      e.stopPropagation();                    // Esc leaves full screen first
    }
  }, true);
})();
