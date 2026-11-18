"""Pixel decoder (SPEC §4.7, §5.3): the U-Net decoder path over the encoder levels -> the RGB frame
at t+1, in [0, 1]. Auxiliary grounding at low weight, separate from the JEPA loss.

Standalone: it reads no config and does not depend on the JEPA core. See models/DESIGN-aux.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from jepa_world.models.unet import ConvBlock


def pixel_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean |pred - sg(target)| over every element: mean absolute error in [0, 1] units (SPEC §5.3)."""
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    return (pred - target.detach()).abs().mean()


class PixelDecoder(nn.Module):
    """[L1, L2, ..., Ln] (fine -> coarse) -> (B, out_ch, S1, S1) in [0, 1].

    levels: (channels, size) per encoder level, fine -> coarse, e.g. [(64, 64), (128, 32), (256, 8)].
    widths: one per decoder stage, i.e. per finer level, in decoding order (coarse -> fine).

    Decoder path: x = Ln, then for each finer level: nearest-upsample to its size, concatenate that
    level's encoder features (the skip), two 3x3 ConvBlocks. The last stage therefore reads L1 plus
    the decoder path (SPEC §4.7). A 1x1 conv and a sigmoid give the frame.

    skips=False is the M7 ablation (docs/decisions/2026-11-17-skip-connections-meaning.md): no
    concatenation, so the frame is decoded from Ln alone. forward still takes (and checks) every
    level, so callers don't branch, but only Ln is consumed.
    """

    def __init__(
        self,
        levels: Sequence[tuple[int, int]],
        *,
        widths: Sequence[int],
        norm_groups: int,
        out_ch: int = 3,
        skips: bool = True,
    ):
        super().__init__()
        levels = [(int(c), int(s)) for c, s in levels]
        if len(levels) < 2:
            raise ValueError(f"need at least two levels to decode, got {levels}")
        if len(widths) != len(levels) - 1:
            raise ValueError(
                f"need one width per decoder stage ({len(levels) - 1}), got widths={list(widths)}"
            )
        for (_, fine), (_, coarse) in zip(levels, levels[1:], strict=False):
            if coarse >= fine or fine % coarse:
                raise ValueError(
                    f"level sizes must shrink by an integer factor fine -> coarse, got {levels}"
                )
        self.levels = levels
        self.skips = bool(skips)
        self.stages = nn.ModuleList()
        c_prev = levels[-1][0]
        for (c_skip, _), w in zip(reversed(levels[:-1]), widths, strict=True):
            self.stages.append(
                nn.Sequential(
                    ConvBlock(c_prev + (c_skip if self.skips else 0), w, groups=norm_groups),
                    ConvBlock(w, w, groups=norm_groups),
                )
            )
            c_prev = w
        self.out = nn.Conv2d(c_prev, out_ch, 1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.levels):
            raise ValueError(
                f"expected {len(self.levels)} levels (fine -> coarse), got {len(feats)}"
            )
        b = feats[0].shape[0] if feats[0].ndim == 4 else None
        for i, (f, (c, s)) in enumerate(zip(feats, self.levels, strict=True)):
            if b is None or f.ndim != 4 or f.shape[0] != b or f.shape[1:] != (c, s, s):
                raise ValueError(f"level {i}: expected (B, {c}, {s}, {s}), got {tuple(f.shape)}")
        x = feats[-1]
        for skip, (_, s), stage in zip(
            reversed(feats[:-1]), reversed(self.levels[:-1]), self.stages, strict=True
        ):
            x = F.interpolate(x, size=(s, s), mode="nearest")
            x = stage(torch.cat([x, skip], dim=1) if self.skips else x)
        return torch.sigmoid(self.out(x))
