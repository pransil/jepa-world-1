"""Tiny configs for fast CPU tests: base.yaml with the shapes shrunk but the SPEC ratios kept.

L1 8ch @ 16, L2 16ch @ 8 (2x down), L3 32ch @ 2 (4x down), like SPEC's 64 -> 32 -> 8. Lives here,
not in conftest.py, so other lanes editing conftest can't collide with it.
"""

from __future__ import annotations

from pathlib import Path

import torch

from jepa_world.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BASE = CONFIGS / "base.yaml"

TINY = [
    "sim.frame_size=16",
    "model.levels={L1: {channels: 8, size: 16}, L2: {channels: 16, size: 8},"
    " L3: {channels: 32, size: 2}}",
    "model.norm_groups=2",
    "model.l1_head_hidden=4",
    "model.horizon_embed_dim=2",
    "train.device=cpu",
    "train.batch_size=2",
    "train.stages=[near]",
    "train.steps=3",
    "train.log_every=2",
    "train.ckpt_every=100",
    "data.source=synthetic",
]

NEAR = [1, 2, 3, 4, 5]


def tiny_cfg(*overrides: str) -> dict:
    return load_config(BASE, [*TINY, *overrides])


def spec_cfg(*overrides: str) -> dict:
    """SPEC §4.2 shapes (64x64, 64/128/256 channels), near stage only."""
    return load_config(BASE, ["train.stages=[near]", "data.source=synthetic", *overrides])


def random_frames(cfg: dict, batch: int = 2, seed: int = 0) -> torch.Tensor:
    """(B, C + 5, 3, S, S) uniform noise. Enough for shape, gradient and leakage tests."""
    g = torch.Generator().manual_seed(seed)
    c = cfg["model"]["context_frames"]
    s = cfg["sim"]["frame_size"]
    return torch.rand(batch, c + max(NEAR), 3, s, s, generator=g)
