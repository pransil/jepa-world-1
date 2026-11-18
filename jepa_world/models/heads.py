"""Predictor heads, one per horizon k (SPEC §4.5).

HorizonHead: near heads (k = 1..5, on L1), a deterministic mean.
GaussianHead: far heads (k = 10 on L2, k = 20 on L3), a mean and a bounded log-variance.
TopDownHead: M7 top-down branches (off by default). A far head's coarse level, upsampled, predicts
the next-finer target at the same horizon (models/DESIGN.md, "Top-down prediction").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from jepa_world.models.unet import ConvBlock, check_groups


class HorizonHead(nn.Module):
    """feat (B, in_ch, H, W) [+ action (B, action_dim)] -> mean prediction (B, out_ch, H, W).

    The input is the level's context features, concatenated with this head's learned k-vector and
    the action slot, each broadcast over positions. Then two k x k ConvBlocks and a 1x1 output
    conv follow.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        hidden: int,
        kernel: int,
        embed_dim: int,
        action_dim: int,
        norm_groups: int,
    ):
        super().__init__()
        check_groups(norm_groups, hidden)
        self.action_dim = action_dim
        self.k_embed = nn.Parameter(torch.randn(embed_dim))
        c_in = in_ch + embed_dim + action_dim
        self.body = nn.Sequential(
            ConvBlock(c_in, hidden, groups=norm_groups, kernel=kernel),
            ConvBlock(hidden, hidden, groups=norm_groups, kernel=kernel),
            nn.Conv2d(hidden, out_ch, 1),
        )

    def forward(self, feat: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        b, _, h, w = feat.shape
        parts = [feat, self.k_embed.view(1, -1, 1, 1).expand(b, -1, h, w)]
        if self.action_dim:
            if action is None:
                action = feat.new_zeros(b, self.action_dim)
            if action.shape != (b, self.action_dim):
                raise ValueError(
                    f"action must be ({b}, {self.action_dim}), got {tuple(action.shape)}"
                )
            parts.append(action.view(b, -1, 1, 1).expand(-1, -1, h, w))
        elif action is not None:
            raise ValueError("an action was given, but model.action_dim is 0")
        return self.body(torch.cat(parts, dim=1))


class GaussianHead(HorizonHead):
    """Far head: the same body, with a 2 * out_ch output conv split into (mean, raw logvar).

    logvar = b * tanh(raw / b), with b = logvar_bound: always in (-b, b), so the variance is
    positive and finite. It is the identity near 0 and keeps a gradient everywhere, unlike a hard
    clamp (models/DESIGN.md). Returns (mean, logvar), each (B, out_ch, H, W).
    """

    def __init__(self, in_ch: int, out_ch: int, *, logvar_bound: float, **kw):
        if logvar_bound <= 0:
            raise ValueError(f"logvar_bound must be > 0, got {logvar_bound}")
        super().__init__(in_ch, 2 * out_ch, **kw)
        self.logvar_bound = float(logvar_bound)

    def forward(  # type: ignore[override]
        self, feat: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw = super().forward(feat, action).chunk(2, dim=1)
        b = self.logvar_bound
        return mean, b * torch.tanh(raw / b)


class TopDownHead(nn.Module):
    """coarse feat (B, in_ch, s, s) -> bilinear x`scale` -> GaussianHead -> (mean, logvar), each
    (B, out_ch, s*scale, s*scale). It reads only the coarse level, so the finer target's error flows
    back into it (docs/decisions/2026-11-17-top-down-targets.md). Bilinear, not nearest (blocky
    tiles) or a transposed conv (checkerboard artefacts, more parameters)."""

    def __init__(self, in_ch: int, out_ch: int, *, scale: int, **gaussian_kw):
        super().__init__()
        if int(scale) != scale or scale < 2:
            raise ValueError(f"scale must be an integer >= 2, got {scale}")
        self.scale = int(scale)
        self.head = GaussianHead(in_ch, out_ch, **gaussian_kw)

    def forward(
        self, feat: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        up = F.interpolate(feat, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return self.head(up, action)
