"""Predictor heads, one per horizon k (SPEC §4.5). M3: deterministic-mean heads on L1.

M4 adds the far heads (mean + log-variance on L2 / L3).
"""

from __future__ import annotations

import torch
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
