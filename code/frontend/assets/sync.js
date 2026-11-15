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
