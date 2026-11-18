"""Tiny configs for fast CPU tests: base.yaml with the shapes shrunk but the SPEC ratios kept.

L1 8ch @ 16, L2 16ch @ 8 (2x down), L3 32ch @ 2 (4x down), like SPEC's 64 -> 32 -> 8. Lives here,
not in conftest.py, so other lanes editing conftest can't collide with it.

tiny_cfg() is near-only (the M3 shape). tiny_far_cfg() adds the far stage, active from step 0.
tiny_aux_cfg() is near + goal + pixel (M5), every stage active from step 0.
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
    "model.l2_head_hidden=4",
    "model.l3_head_hidden=4",
    "model.horizon_embed_dim=2",
    "train.device=cpu",
    "train.batch_size=2",
    "train.stages=[near]",
    "train.stage_start={near: 0, far: 0}",
    "train.steps=3",
    "train.log_every=2",
    "train.ckpt_every=100",
    "data.source=synthetic",
]
FAR_STAGE = ["train.stages=[near, far]"]
AUX = [
    "train.stages=[near, goal, pixel]",
    "train.stage_start={near: 0, far: 0, goal: 0, pixel: 0}",
    "model.goal_horizons=[1, 3, 5]",
    "model.goal_hidden=4",
    "model.pix_widths=[4, 4]",
]

NEAR = [1, 2, 3, 4, 5]
FAR = [10, 20]
ALL = NEAR + FAR
LEVEL = {1: "L1", 2: "L1", 3: "L1", 4: "L1", 5: "L1", 10: "L2", 20: "L3"}
TINY_SHAPE = {"L1": (8, 16, 16), "L2": (16, 8, 8), "L3": (32, 2, 2)}  # (channels, H, W)


def tiny_cfg(*overrides: str) -> dict:
    return load_config(BASE, [*TINY, *overrides])


def tiny_far_cfg(*overrides: str) -> dict:
    return load_config(BASE, [*TINY, *FAR_STAGE, *overrides])


def tiny_aux_cfg(*overrides: str) -> dict:
    return load_config(BASE, [*TINY, *AUX, *overrides])


def spec_cfg(*overrides: str) -> dict:
    """SPEC §4.2 shapes (64x64, 64/128/256 channels), near stage only unless overridden."""
    return load_config(BASE, ["train.stages=[near]", "data.source=synthetic", *overrides])


def k_max(cfg: dict) -> int:
    stages = cfg["train"]["stages"]
    ks = NEAR + (FAR if "far" in stages else [])
    if "goal" in stages:
        ks += list(cfg["model"]["goal_horizons"])
    return max(ks)


def random_frames(cfg: dict, batch: int = 2, seed: int = 0) -> torch.Tensor:
    """(B, C + k_max, 3, S, S) uniform noise. Enough for shape, gradient and leakage tests."""
    g = torch.Generator().manual_seed(seed)
    c = cfg["model"]["context_frames"]
    s = cfg["sim"]["frame_size"]
    return torch.rand(batch, c + k_max(cfg), 3, s, s, generator=g)


def random_batch(cfg: dict, batch: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    """random_frames plus a random ball_pos (B, C + k_max, 2) on the same time axis (with_gt)."""
    frames = random_frames(cfg, batch, seed)
    g = torch.Generator().manual_seed(seed + 1)
    return {"frames": frames, "ball_pos": torch.randn(batch, frames.shape[1], 2, generator=g)}
