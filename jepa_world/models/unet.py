"""U-Net encoder: the context encoder, and (as an EMA copy) the target encoder. SPEC §4.2-4.3.

Only the contracting path lives here. The decoder path exists only for the pixel decoder (M5).
See models/DESIGN.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def check_groups(groups: int, channels: int) -> int:
    if channels % groups:
        raise ValueError(f"{groups} GroupNorm groups do not divide {channels} channels")
    return groups


class ConvBlock(nn.Sequential):
    """conv (k x k, given stride) -> GroupNorm -> SiLU."""

    def __init__(self, c_in: int, c_out: int, *, groups: int, kernel: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=kernel // 2),
            nn.GroupNorm(check_groups(groups, c_out), c_out),
            nn.SiLU(),
        )


class UNetEncoder(nn.Module):
    """x (B, in_ch, S, S) -> {level name: (B, ch, size, size)}, in level order.

    Each level first downsamples with one 3x3 stride-2 conv per 2x step (so 32 -> 8 is two steps),
    then applies two 3x3 ConvBlocks. GroupNorm throughout, so the module has no buffers.
    """

    def __init__(
        self,
        in_ch: int,
        levels: Sequence[tuple[str, int, int]],
        *,
        input_size: int,
        downsample: str,
        norm_groups: int,
    ):
        super().__init__()
        if downsample != "stride2":
            raise ValueError(
                f"downsample={downsample!r}: only 'stride2' is implemented "
                "(docs/decisions/2026-11-17-downsample-stride2.md)"
            )
        self.in_ch, self.input_size = in_ch, input_size
        self.names: list[str] = []
        self.stages = nn.ModuleList()
        c_prev, s_prev = in_ch, input_size
        for name, ch, size in levels:
            ratio, rem = divmod(s_prev, size)
            if size <= 0 or rem or ratio < 1 or ratio & (ratio - 1):
                raise ValueError(
                    f"level {name}: resolution {s_prev} -> {size} is not a power of 2 reduction"
                )
            steps = ratio.bit_length() - 1
            layers: list[nn.Module] = [
                ConvBlock(c_prev, c_prev, groups=norm_groups, stride=2) for _ in range(steps)
            ]
            layers += [
                ConvBlock(c_prev, ch, groups=norm_groups),
                ConvBlock(ch, ch, groups=norm_groups),
            ]
            self.names.append(name)
            self.stages.append(nn.Sequential(*layers))
            c_prev, s_prev = ch, size

    def forward(self, x: torch.Tensor, upto: str | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_ch or x.shape[-2:] != (self.input_size,) * 2:
            raise ValueError(
                f"expected (B, {self.in_ch}, {self.input_size}, {self.input_size}), "
                f"got {tuple(x.shape)}"
            )
        if upto is not None and upto not in self.names:
            raise ValueError(f"upto={upto!r}: no such level (have {self.names})")
        out: dict[str, torch.Tensor] = {}
        for name, stage in zip(self.names, self.stages, strict=True):
            x = stage(x)
            out[name] = x
            if name == upto:
                break
        return out


def encoder_from_cfg(cfg: dict) -> UNetEncoder:
    m = cfg["model"]
    return UNetEncoder(
        3 * m["context_frames"],
        [(name, lv["channels"], lv["size"]) for name, lv in m["levels"].items()],
        input_size=cfg["sim"]["frame_size"],
        downsample=m["downsample"],
        norm_groups=m["norm_groups"],
    )
