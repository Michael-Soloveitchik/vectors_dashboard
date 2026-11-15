"""Dash launcher — background werkzeug server with a real stop handle.

Re-running the notebook cell stops OUR previous server on that port and
rebinds. A port held by a different process is never killed (the old
launcher SIGKILLed whatever owned the port — on a shared box that can be
someone else's job): we move to the next free port and say so.

``jupyter_mode``: external (print URL) · tab (open browser tab) ·
inline / jupyterlab (IFrame).
"""
from __future__ import annotations

import socket
import sys
import threading

JUPYTER_MODES = {"inline", "external", "jupyterlab", "tab"}
# Lives on `sys` so it survives the notebook's `del sys.modules[...]` reload.
_REGISTRY: dict[int, object] = sys.__dict__.setdefault(
    "_vectors_dashboard_servers", {})


# Injected right after Dash's config block: the page works from ANY URL
# prefix (code-server / jupyter-server-proxy serve it at /proxy/<port>/).
_RELOCATE = """<script>(function(){var e=document.getElementById('_dash-config');
if(!e){return;}var c=JSON.parse(e.textContent),p=window.location.pathname;
p=p.substring(0,p.lastIndexOf('/')+1);c.requests_pathname_prefix=p;
c.routes_pathname_prefix=p;c.url_base_pathname=null;
e.textContent=JSON.stringify(c);})();</script>"""


def make_relocatable(server) -> None:
    """Relative asset URLs + path prefix taken from the browser's address,
    and a CORS-open ``/_alive`` so a notebook can find a reachable URL."""
    from flask import Response, request

    @server.after_request
    def _relative(resp):
        if resp.mimetype != "text/html" or resp.direct_passthrough:
            return resp
        html = resp.get_data(as_text=True)
        at = html.find('id="_dash-config"')
        if at < 0:
            return resp
        end = html.find("</script>", at) + len("</script>")
        html = html[:end] + _RELOCATE + html[end:]
        resp.set_data(html.replace('src="/', 'src="').replace('href="/', 'href="'))
        return resp

    @server.route("/_alive")
    def _alive():
        return Response("ok", headers={"Access-Control-Allow-Origin": "*"})


def _inline_html(port: int, urls: list[str], height: int) -> str:
    """An iframe that finds, in the BROWSER, a URL that reaches the server:
    same-origin proxy path first (remote / https notebooks), then the
    notebook's host, the LAN address, localhost."""
    uid = f"vd-{port}-{id(urls) % 100000}"
    return f"""
<div id="{uid}-msg" style="font:12px sans-serif;color:#888">connecting to the dashboard…</div>
<iframe id="{uid}" style="width:100%;height:{int(height)}px;border:0;border-radius:12px;display:none"></iframe>
<script>(async function() {{
  const frame = document.getElementById("{uid}"), msg = document.getElementById("{uid}-msg");
  const loc = window.location, list = [];
  const origins = [loc.origin].concat(Array.from(loc.ancestorOrigins || []));
  origins.filter(o => o && o.indexOf("http") === 0).forEach(o => {{
    list.push(o + "/proxy/{port}/");                 // code-server, jupyter-server-proxy
    list.push(o.replace(/:\\d+$/, "") + ":{port}/");   // same host, dashboard port
  }});
  {urls!r}.forEach(u => list.push(u));
  for (const u of list) {{
    try {{
      const r = await fetch(u + "_alive", {{mode: "cors"}});
      if (r.ok && (await r.text()) === "ok") {{
        frame.src = u; frame.style.display = "block"; msg.innerHTML =
          'dashboard: <a href="' + u + '" target="_blank">' + u + '</a>'; return;
      }}
    }} catch (e) {{}}
  }}
  msg.innerHTML = "could not reach the dashboard from this page — open one of: " +
    {urls!r}.map(u => '<a href="' + u + '" target="_blank">' + u + '</a>').join(" · ");
}})();</script>"""


def _busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def stop(port: int) -> bool:
    srv = _REGISTRY.pop(port, None)
    if srv is None:
        return False
    try:
        srv.shutdown()
        srv.server_close()
    except Exception:
        pass
    return True


def stop_all() -> int:
    """Stop every dashboard server this process started; returns how many."""
    return sum(stop(p) for p in list(_REGISTRY))


def _urls(port: int) -> list[str]:
    urls = [f"http://127.0.0.1:{port}/"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        if ip and ip != "127.0.0.1":
            urls.append(f"http://{ip}:{port}/")
    except OSError:
        pass
    return urls


def run_dash(app, host: str = "0.0.0.0", port: int = 8070,
             jupyter_mode: str | None = None, jupyter_height: int = 900,
             in_jupyter: bool = False) -> tuple[str, int]:
    if jupyter_mode is not None and jupyter_mode not in JUPYTER_MODES:
        raise ValueError(
            f"jupyter_mode must be one of {sorted(JUPYTER_MODES)} or None")
    import logging
    from werkzeug.serving import make_server
    logging.getLogger("werkzeug").setLevel(logging.ERROR)  # no per-request spam
    stop(port)
    want = port
    while _busy(port) and port < want + 20:
        port += 1
    if port != want:
        print(f"[note] port {want} is held by another process — using {port}")
    srv = make_server(host, port, app.server, threaded=True)
    _REGISTRY[port] = srv
    threading.Thread(target=srv.serve_forever, daemon=True,
                     name=f"vectors-dashboard:{port}").start()
    urls = _urls(port)
    print("vectors dashboard →  " + "   ".join(urls)
          + ("" if host in ("127.0.0.1", "localhost") else
             "   (all interfaces; host='127.0.0.1' to keep it local)"))
    mode = (jupyter_mode or ("inline" if in_jupyter else "external")).lower()
    if mode == "tab":
        try:
            import webbrowser
            webbrowser.open_new_tab(urls[0])
        except Exception as exc:
            print(f"[warn] could not open a browser tab ({exc})")
    elif mode in ("inline", "jupyterlab"):
        try:
            from IPython.display import HTML, display
            display(HTML(_inline_html(port, urls, jupyter_height)))
        except Exception as exc:
            print(f"[warn] inline display unavailable ({exc}); open {urls[0]}")
    return urls[0], port
