"""Dash launcher with 4-mode Jupyter dispatch.

Always runs the dev server in a daemon thread so the call returns
immediately. ``jupyter_mode`` selects how to surface the URL:

  - external   : print URL only, no IFrame.
  - tab        : open a new browser tab (openclaw → webbrowser fallback).
  - inline     : render IPython.display.IFrame at jupyter_height.
  - jupyterlab : same iframe (renders in JupyterLab side panel).

Outside Jupyter every mode prints the URL and returns.
"""
from __future__ import annotations

import socket
import threading

JUPYTER_MODES = {"inline", "external", "jupyterlab", "tab"}


def _free_port(port: int) -> bool:
    import os
    import signal
    import subprocess
    import time
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return True
    try:
        pids = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], text=True
        ).strip().split()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    time.sleep(0.3)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _wait_until_ready(host: str, port: int, timeout: float = 8.0) -> bool:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                return True
        time.sleep(0.1)
    return False


def _display_inline(url: str, height: int):
    try:
        from IPython.display import IFrame, display
        display(IFrame(src=url, width="100%", height=height))
    except Exception as exc:
        print(f"[warn] IPython.display.IFrame unavailable ({exc}). Open: {url}")


def _open_in_tab(url: str) -> None:
    import json
    import os
    import shutil
    import subprocess
    token = os.path.expanduser("~/.openclaw/openclaw.json")
    if os.path.exists(token) and shutil.which("curl"):
        try:
            with open(token) as fh:
                tok = json.load(fh)["gateway"]["auth"]["token"]
            subprocess.run(
                ["curl", "-s", "--max-time", "10",
                 "http://127.0.0.1:18791/tabs/open",
                 "-H", f"Authorization: Bearer {tok}",
                 "-H", "Content-Type: application/json",
                 "-X", "POST", "-d", json.dumps({"url": url})],
                check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass
    try:
        import webbrowser
        webbrowser.open_new_tab(url)
    except Exception as exc:
        print(f"[warn] could not open browser tab ({exc}). Open: {url}")


def run_dash(app, host: str = "127.0.0.1", port: int = 8050,
             jupyter_mode: str | None = None, jupyter_height: int = 820,
             in_jupyter: bool = False) -> None:
    if jupyter_mode is not None and jupyter_mode not in JUPYTER_MODES:
        raise ValueError(
            f"jupyter_mode must be one of {sorted(JUPYTER_MODES)} or None")
    if not _free_port(port):
        port += 1
        if not _free_port(port):
            print(f"[error] port unavailable, aborting.")
            return
    # Always print 127.0.0.1 + every reachable LAN IP so the user picks
    # whichever URL their browser can hit (Windows host vs Mac vs SSH).
    urls = [f"http://127.0.0.1:{port}/"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as _s:
            _s.connect(("8.8.8.8", 80))
            lan_ip = _s.getsockname()[0]
        if lan_ip and lan_ip != "127.0.0.1":
            urls.append(f"http://{lan_ip}:{port}/")
    except Exception:
        pass
    url = urls[0]
    print(f"\n{'=' * 60}\n  vectors_dashboard")
    for u in urls:
        print(f"    {u}")
    print(f"{'=' * 60}\n")

    def _serve():
        try:
            app.run(host=host, port=port, debug=False, use_reloader=False)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"[warn] server stopped: {exc}")

    threading.Thread(target=_serve, daemon=True).start()
    if not _wait_until_ready(host, port, timeout=8.0):
        print(f"[warn] server slow on {url}; open manually if blank.")
    print(f"running on {url}")

    mode = (jupyter_mode or ("inline" if in_jupyter else "external")).lower()
    if mode == "external":
        return
    if mode == "tab":
        _open_in_tab(url)
        return
    if mode in ("inline", "jupyterlab"):
        _display_inline(url, jupyter_height)
