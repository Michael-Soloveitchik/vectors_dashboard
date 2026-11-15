"""JSONL debug logger w/ visible bracket-balance.

Every wrapped callback prints an OPEN line on enter and a CLOSE line on
exit. The shared depth counter is rendered as a bracket string so an
unfinished callback is impossible to miss in the terminal:

    ( fit                                  depth=1   open=1   "("
    ((render_pca                           depth=2   open=2   "(("
    ((( render_cm                          depth=3   open=3   "((("
    ((  →render_cm 12.4ms                  depth=2   open=2   "(("
    (   →render_pca 87.1ms                 depth=1   open=1   "("
        →fit 240.0ms                       depth=0   open=0   ""

If a callback raises, the close line is tagged ``ERR`` and the bracket
string keeps the unmatched ``(`` so the next OPEN visually inherits the
imbalance until a separate cleanup is called.

Background watchdog thread prints a `STILL OPEN` warning when any frame
sits in the stack longer than ``WATCHDOG_S`` (default 5 s).

Action ID:
    Flask `before_request` hook reads ``X-Action-Id`` (sent by the Dash
    front-end via the `_dash-update-component` POST) and sets a context-
    var. Every log line is tagged with that id so a single user click is
    one search-string in the file.

JSONL file: logs/dashboard_events.jsonl (one record per OPEN/CLOSE).
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

LOG_PATH = Path(__file__).resolve().parents[2] / "logs" / "dashboard_events.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

WATCHDOG_S = float(os.environ.get("DLOG_WATCHDOG_S", "5.0"))
PRINT = os.environ.get("DLOG_PRINT", "1") != "0"
MAX_BRACKET = 40
_action_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_action_id", default="-")


class _DebugLog:
    def __init__(self) -> None:
        self._stack: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._fh = LOG_PATH.open("a", buffering=1)
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    # ── public API ──────────────────────────────────────────────
    def enter(self, name: str, **extra: Any) -> dict[str, Any]:
        frame = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "t0": time.monotonic(),
            "action": _action_id.get(),
            **extra,
        }
        with self._lock:
            self._stack.append(frame)
            depth = len(self._stack)
        self._write({"event": "open", **frame, "depth": depth})
        if PRINT:
            print(f"{self._brackets(depth)}OPEN  {name}  "
                  f"depth={depth} action={frame['action']}", flush=True)
        return frame

    def exit(self, frame: dict[str, Any], *, ok: bool = True,
             err: str | None = None, **extra: Any) -> None:
        dt_ms = (time.monotonic() - frame["t0"]) * 1000.0
        with self._lock:
            try:
                self._stack.remove(frame)
            except ValueError:
                pass
            depth = len(self._stack)
        rec = {"event": "close", "ok": ok, "ms": round(dt_ms, 2),
               "err": err, "depth_after": depth, **frame, **extra}
        self._write(rec)
        if PRINT:
            tag = "OK " if ok else "ERR"
            print(f"{self._brackets(depth)}{tag} →{frame['name']} "
                  f"{dt_ms:.1f}ms  depth={depth} action={frame['action']}"
                  + (f"  err={err}" if err else ""), flush=True)

    @contextmanager
    def span(self, name: str, **extra: Any):
        f = self.enter(name, **extra)
        try:
            yield f
        except Exception as e:
            self.exit(f, ok=False, err=f"{type(e).__name__}: {e}")
            raise
        else:
            self.exit(f, ok=True)

    def event(self, kind: str, **extra: Any) -> None:
        self._write({"event": "tick", "kind": kind,
                     "action": _action_id.get(),
                     "depth": len(self._stack), **extra})

    # ── helpers ─────────────────────────────────────────────────
    def _brackets(self, depth: int) -> str:
        d = max(0, min(depth, MAX_BRACKET))
        return "(" * d + " " * (MAX_BRACKET - d) + " "

    def _write(self, rec: dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def _watchdog(self) -> None:
        while True:
            time.sleep(WATCHDOG_S / 2.0)
            now = time.monotonic()
            with self._lock:
                stale = [f for f in self._stack
                         if now - f["t0"] > WATCHDOG_S]
            for f in stale:
                age = now - f["t0"]
                if PRINT:
                    print(f"!!! STILL OPEN {age:.1f}s  {f['name']}  "
                          f"action={f['action']}  id={f['id']}",
                          flush=True)
                self._write({"event": "stale", "age_s": round(age, 2), **f})


dlog = _DebugLog()


# ── decorator for Dash callbacks ────────────────────────────────
def trace(fn: Callable) -> Callable:
    """Wrap a callback to emit OPEN/CLOSE log records w/ bracket balance."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        f = dlog.enter(fn.__name__)
        try:
            r = fn(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            dlog.exit(f, ok=False, err=f"{type(e).__name__}: {e}", tb=tb)
            raise
        dlog.exit(f, ok=True)
        return r
    return wrapped


# ── Flask hook ──────────────────────────────────────────────────
def install_action_hook(server) -> None:
    """Bind ``X-Action-Id`` request header → contextvar so log records
    can be filtered per user click."""
    from flask import request

    @server.before_request
    def _before():  # noqa: ANN202
        aid = request.headers.get("X-Action-Id") or request.headers.get(
            "x-action-id") or "-"
        _action_id.set(aid)

    @server.after_request
    def _after(resp):  # noqa: ANN001 ANN202
        try:
            resp.headers["X-Server-Action-Id"] = _action_id.get()
        except Exception:
            pass
        return resp


__all__ = ["dlog", "trace", "install_action_hook"]
