"""Synthetic moving-disc clips, in the M3 batch layout (B, T, 3, S, S).

A stand-in batch source (data.source: synthetic). The real one is M2's episode loader
(data/dataset.py). Use this for smoke runs and benchmarks ONLY. It shares the shapes and rough look of the real task (a flat-shaded ball over a static 1-D
terrain profile, seen side-on), but not its physics: the disc flies ballistically and bounces off
the frame edges, ignoring the terrain. Nothing measured on it is evidence about the real task.
"""

from __future__ import annotations

import math

import torch

SKY = (0.55, 0.75, 0.95)
GROUND = (0.35, 0.28, 0.20)
BALL = (0.95, 0.25, 0.20)


def _fold(p: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Reflect p into [lo, hi] (elastic bounces off both walls)."""
    span = hi - lo
    q = torch.remainder(p - lo, 2 * span)
    return lo + torch.where(q > span, 2 * span - q, q)


def moving_disc_batch(
    batch: int, frames: int, size: int, *, generator: torch.Generator | None = None
) -> torch.Tensor:
    """(batch, frames, 3, size, size) float32 in [0, 1]. Deterministic given `generator` (CPU)."""

    def u(*shape: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(*shape, generator=generator)

    s = float(size)
    col = torch.arange(size, dtype=torch.float32) + 0.5  # pixel centres
    row = col.view(-1, 1)

    # terrain: per clip, a sum of 3 sines; surface row index per column (rows grow downward)
    freq = u(batch, 3, 1, lo=0.5, hi=3.0) * 2 * math.pi / s
    phase = u(batch, 3, 1, hi=2 * math.pi)
    amp = u(batch, 3, 1, hi=0.06 * s)
    surface = 0.75 * s + (amp * torch.sin(freq * col + phase)).sum(1)  # (B, S)
    ground = (row.unsqueeze(0) >= surface.unsqueeze(1)).float()  # (B, S, S)

    # disc: radius, start, velocity (px/frame), gravity (px/frame^2)
    r = u(batch, lo=0.05 * s, hi=0.1 * s)
    x0, y0 = u(batch, lo=0.2 * s, hi=0.8 * s), u(batch, lo=0.15 * s, hi=0.5 * s)
    vx, vy = u(batch, lo=-0.06 * s, hi=0.06 * s), u(batch, lo=-0.05 * s, hi=0.02 * s)
    g = 0.004 * s
    t = torch.arange(frames, dtype=torch.float32)
    x = _fold(x0[:, None] + vx[:, None] * t, r[:, None], s - r[:, None])  # (B, T)
    y = _fold(y0[:, None] + vy[:, None] * t + 0.5 * g * t**2, r[:, None], s - r[:, None])  # (B, T)
    d2 = (col.view(1, 1, 1, -1) - x[..., None, None]) ** 2 + (
        row.view(1, 1, -1, 1) - y[..., None, None]
    ) ** 2
    disc = (d2 <= r.view(-1, 1, 1, 1) ** 2).float()  # (B, T, S, S)

    sky, gnd, ball = (torch.tensor(c).view(1, 1, 3, 1, 1) for c in (SKY, GROUND, BALL))
    background = sky + (gnd - sky) * ground[:, None, None]  # (B, 1, 3, S, S)
    img = background + (ball - background) * disc[:, :, None]
    return img.clamp_(0.0, 1.0).contiguous()
