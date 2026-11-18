"""Goal head (SPEC §4.6, §5.2): pooled encoder levels -> a diagonal 2-D Gaussian over the ball's
displacement from time t, one per horizon (docs/decisions/2026-11-17-goal-target-displacement.md).

Standalone: it reads no config and does not depend on the JEPA core. See models/DESIGN-aux.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from jepa_world.models.unet import check_groups

LOG_2PI = math.log(2 * math.pi)


def displacement_targets(
    ball_pos: torch.Tensor, context_frames: int, horizons: Sequence[int]
) -> torch.Tensor:
    """ball_pos (B, T, 2) world (x, z), on the batch's frame axis (index C-1 is time t)
    -> (B, K, 2) float32: ball_pos[:, C-1+k] - ball_pos[:, C-1], in `horizons` order.

    The subtraction runs at the input's precision (M2 stores float64), then casts to fp32."""
    if ball_pos.ndim != 3 or ball_pos.shape[-1] != 2:
        raise ValueError(f"ball_pos must be (B, T, 2), got {tuple(ball_pos.shape)}")
    if not horizons or min(horizons) < 0:
        # A negative index would silently wrap to the end of the episode.
        raise ValueError(f"horizons must be non-empty and >= 0, got {list(horizons)}")
    if context_frames < 1:
        raise ValueError(f"context_frames must be >= 1, got {context_frames}")
    need = context_frames + max(horizons)
    if ball_pos.shape[1] < need:
        raise ValueError(
            f"ball_pos has {ball_pos.shape[1]} frames; horizons {list(horizons)} with "
            f"{context_frames} context frames need {need}"
        )
    t = context_frames - 1
    idx = torch.tensor([t + k for k in horizons], device=ball_pos.device)
    return (ball_pos[:, idx] - ball_pos[:, t : t + 1]).to(torch.float32)


def goal_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Diagonal-Gaussian NLL of sg(target), (B, K, 2) each -> (K,), one value per horizon:

        mean_b sum_{x,z} 0.5 * [(t - mu)^2 * exp(-logvar) + logvar + log(2 pi)]

    The same formula as the far heads' gaussian_nll (models/DESIGN.md), so it is a true NLL."""
    shapes = {"mean": mean.shape, "logvar": logvar.shape, "target": target.shape}
    if len(set(shapes.values())) != 1 or mean.ndim != 3 or mean.shape[-1] != 2:
        raise ValueError(f"shape mismatch, expected (B, K, 2) each: {shapes}")
    err2 = (target.detach() - mean).square()
    return (0.5 * (err2 * torch.exp(-logvar) + logvar + LOG_2PI)).sum(dim=-1).mean(dim=0)


class GoalHead(nn.Module):
    """l3 (B, l3_ch, h, w) [, l2 (B, l2_ch, h', w')] -> (mean, logvar), each (B, K, 2).

    Both levels are global-average-pooled and concatenated. Each horizon appends its own learned
    k-vector, and one shared MLP (Linear -> GroupNorm -> SiLU, twice, then Linear) maps that to a
    mean and a raw log-variance, bounded as logvar = b * tanh(raw / b), like the far heads.
    """

    def __init__(
        self,
        l3_ch: int,
        *,
        horizons: Sequence[int],
        hidden: int,
        embed_dim: int,
        norm_groups: int,
        logvar_bound: float,
        l2_ch: int = 0,
    ):
        super().__init__()
        horizons = tuple(int(k) for k in horizons)
        if not horizons or len(set(horizons)) != len(horizons) or min(horizons) < 0:
            raise ValueError(f"horizons must be non-empty, distinct and >= 0, got {list(horizons)}")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1 (it tells horizons apart), got {embed_dim}")
        if logvar_bound <= 0:
            raise ValueError(f"logvar_bound must be > 0, got {logvar_bound}")
        if hidden // check_groups(norm_groups, hidden) < 2:
            # On a vector, a 1-channel group normalises to its bias: the input would be cut off.
            raise ValueError(
                f"GroupNorm on a vector needs >= 2 channels per group: hidden={hidden}, "
                f"norm_groups={norm_groups}"
            )
        self.horizons = horizons
        self.l3_ch, self.l2_ch = l3_ch, l2_ch
        self.logvar_bound = float(logvar_bound)
        self.k_embed = nn.Parameter(torch.randn(len(horizons), embed_dim))
        self.mlp = nn.Sequential(
            nn.Linear(l3_ch + l2_ch + embed_dim, hidden),
            nn.GroupNorm(norm_groups, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.GroupNorm(norm_groups, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    @staticmethod
    def _pool(name: str, x: torch.Tensor, ch: int) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != ch:
            raise ValueError(f"{name} must be (B, {ch}, H, W), got {tuple(x.shape)}")
        return x.mean(dim=(2, 3))

    def forward(
        self, l3: torch.Tensor, l2: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = [self._pool("l3", l3, self.l3_ch)]
        if self.l2_ch:
            if l2 is None:
                raise ValueError(f"this head reads l2 ({self.l2_ch} channels), but none was given")
            if l2.shape[0] != l3.shape[0]:
                raise ValueError(f"l2 batch {l2.shape[0]} != l3 batch {l3.shape[0]}")
            pooled.append(self._pool("l2", l2, self.l2_ch))
        elif l2 is not None:
            raise ValueError("an l2 was given, but this head was built with l2_ch=0")
        g = torch.cat(pooled, dim=1)  # (B, C)
        b, k = g.shape[0], len(self.horizons)
        x = torch.cat([g.unsqueeze(1).expand(b, k, -1), self.k_embed.expand(b, -1, -1)], dim=2)
        mean, raw = self.mlp(x.reshape(b * k, -1)).view(b, k, 4).split(2, dim=2)
        bound = self.logvar_bound
        return mean, bound * torch.tanh(raw / bound)
