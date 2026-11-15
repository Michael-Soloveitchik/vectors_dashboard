"""Parametric t-SNE: a small network learns the t-SNE map.

Unlike t-SNE the result is a function — training can continue (every
press of "train" runs ``STEPS`` more gradient steps) and the same map
places any vector. One model per (model, perplexity, max_points), trained
on the displayed rows, kept for the process life (``MAX_MODELS``).
CPU only and at most two threads: the box is shared.
"""
from __future__ import annotations

import math
import threading

import numpy as np

from .concurrency import freeze
from .data import ModelFrame

STEPS = 300            # gradient steps per training round
BATCH = 512
MAX_MODELS = 3
_LOCK = threading.Lock()         # one training / embedding at a time


class NotTrained(RuntimeError):
    """No p-t-SNE model for these settings yet."""


def _torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError("p-t-SNE needs PyTorch (pip install torch)") from exc


def _sqdist(torch, A):
    sq = (A * A).sum(1)
    return (sq[:, None] + sq[None, :] - 2.0 * A @ A.T).clamp_min(0.0)


def _affinities(torch, X, perplexity: float):
    """Symmetric t-SNE affinities of one batch: per row, binary search of
    the Gaussian width that gives the requested perplexity."""
    n = X.shape[0]
    eye = torch.eye(n, dtype=torch.bool)
    d2 = _sqdist(torch, X)
    d2 = d2 - d2.masked_fill(eye, float("inf")).min(1, keepdim=True).values
    target = math.log(max(2.0, min(perplexity, (n - 1) / 3.0)))
    beta = torch.ones(n, 1, dtype=X.dtype)
    lo, hi = torch.zeros_like(beta), torch.full_like(beta, float("inf"))
    for _ in range(40):
        p = torch.exp(-d2 * beta).masked_fill(eye, 0.0)
        s = p.sum(1, keepdim=True).clamp_min(1e-12)
        h = torch.log(s) + beta * (d2 * p).sum(1, keepdim=True) / s
        up = h > target                       # too flat → narrower Gaussian
        lo, hi = torch.where(up, beta, lo), torch.where(up, hi, beta)
        beta = torch.where(up, torch.where(torch.isinf(hi), beta * 2,
                                           (beta + hi) / 2), (beta + lo) / 2)
    p = torch.exp(-d2 * beta).masked_fill(eye, 0.0)
    p = p / p.sum(1, keepdim=True).clamp_min(1e-12)
    return ((p + p.T) / (2 * n)).clamp_min(1e-12)


class Model:
    def __init__(self, dim: int, perplexity: float, seed: int = 42) -> None:
        torch = _torch()
        self.perplexity, self.rounds, self.loss = float(perplexity), 0, math.nan
        self.gen = torch.Generator().manual_seed(seed)
        nn = torch.nn
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(),
                                 nn.Linear(128, 2))
        with torch.no_grad():                 # seeded, global RNG untouched
            for layer in self.net:
                if isinstance(layer, nn.Linear):
                    w = torch.randn(layer.weight.shape, generator=self.gen)
                    layer.weight.copy_(w / math.sqrt(layer.in_features))
                    layer.bias.zero_()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def train(self, X: np.ndarray, steps: int = STEPS) -> None:
        torch = _torch()
        threads = torch.get_num_threads()
        torch.set_num_threads(min(threads, 2))
        try:
            Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            n = Xt.shape[0]
            b = min(n, BATCH)
            eye = torch.eye(b, dtype=torch.bool)
            loss = None
            for _ in range(steps):
                xb = Xt[torch.randperm(n, generator=self.gen)[:b]]
                with torch.no_grad():
                    P = _affinities(torch, xb, self.perplexity)
                num = (1.0 / (1.0 + _sqdist(torch, self.net(xb)))
                       ).masked_fill(eye, 0.0)
                Q = (num / num.sum()).clamp_min(1e-12)
                loss = (P * (torch.log(P) - torch.log(Q))).sum()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.loss, self.rounds = float(loss.detach()), self.rounds + 1
        finally:
            torch.set_num_threads(threads)

    def transform(self, X: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            return np.concatenate([self.net(Xt[i:i + 8192]).numpy()
                                   for i in range(0, Xt.shape[0], 8192)]
                                  ).astype(float)


def _key(frame: ModelFrame, perplexity, max_points) -> tuple:
    return (frame.name, round(float(perplexity), 3), int(max_points))


def info(engine, frame: ModelFrame, perplexity, max_points) -> dict:
    m = engine.ptsne.get(_key(frame, perplexity, max_points))
    return {"rounds": m.rounds if m else 0, "steps": (m.rounds if m else 0) * STEPS,
            "loss": m.loss if m else None}


def train(engine, frame: ModelFrame, perplexity, max_points) -> dict:
    """One more round on the displayed rows → ``info``."""
    from .project import display_rows
    key = _key(frame, perplexity, max_points)
    with _LOCK:
        m = engine.ptsne.get(key)
        if m is None:
            m = engine.ptsne[key] = Model(frame.dim, perplexity)
            while len(engine.ptsne) > MAX_MODELS:
                engine.ptsne.popitem(last=False)
        engine.ptsne.move_to_end(key)
        m.train(frame.Vn[display_rows(frame.n, int(max_points))])
    return info(engine, frame, perplexity, max_points)


def embed(engine, frame: ModelFrame, perplexity, max_points):
    from .project import Projection, display_rows
    with _LOCK:
        m = engine.ptsne.get(_key(frame, perplexity, max_points))
        if m is None or not m.rounds:
            raise NotTrained("p-t-SNE is not trained for these settings yet "
                             "— press  ▶ train")
        rows = display_rows(frame.n, int(max_points))
        xy = m.transform(frame.Vn[rows])
    xy = xy - np.median(xy, axis=0, keepdims=True)
    return Projection(rows=rows, xy=freeze(xy), method="ptsne")
