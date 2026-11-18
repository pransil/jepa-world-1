"""Fake episodes in M1's real format (`sim.episode.Episode`, saved by `sim.io.save_episode`).

Frames encode where they came from, so loader tests can decode every pixel back to its source:
  R = frame index t,  G = terrain_seed % 251,  B = ball_seed % 251  (uint8, constant per frame).
Terrain and ball parameters come from the real samplers, so terrain hashes differ by seed as they
do in real data. Ground-truth motion is analytic, so baseline errors have exact expected values.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from jepa_world.sim.ball import sample_ball
from jepa_world.sim.episode import Episode, plain_json, simulate_episode
from jepa_world.sim.terrain import sample_terrain

G = 9.81


def encoded_frames(T: int, S: int, terrain_seed: int, ball_seed: int) -> np.ndarray:
    assert T <= 256, "frame index must fit in one uint8 channel"
    frames = np.empty((T, S, S, 3), np.uint8)
    frames[..., 0] = np.arange(T, dtype=np.uint8)[:, None, None]
    frames[..., 1] = terrain_seed % 251
    frames[..., 2] = ball_seed % 251
    return frames


def decode(batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(B, F, 3, S, S) float in [0, 1] -> (frame index, terrain code, ball code), each (B, F),
    after checking every pixel of a frame agrees (so a frame is never a blend of two)."""
    x = np.rint(np.asarray(batch, dtype=np.float64) * 255).astype(np.int64)
    assert (x == x[..., :1, :1]).all(), "a frame's pixels disagree: frames were mixed"
    return x[:, :, 0, 0, 0], x[:, :, 1, 0, 0], x[:, :, 2, 0, 0]


def fake_episode(
    sim_cfg: dict, terrain_seed: int, ball_seed: int, motion: str = "uniform"
) -> Episode:
    T, S, fps = int(sim_cfg["episode_frames"]), int(sim_cfg["frame_size"]), sim_cfg["fps"]
    t = np.arange(T) / fps
    rng = np.random.default_rng([terrain_seed, ball_seed])
    p0, v0 = rng.uniform(-5, 5, 2), rng.uniform(-3, 3, 2)
    pos = p0 + v0 * t[:, None]
    vel = np.repeat(v0[None], T, axis=0)
    if motion == "ballistic":
        pos[:, 1] -= 0.5 * G * t**2
        vel[:, 1] -= G * t
    elif motion != "uniform":
        raise ValueError(motion)
    meta = {
        "generator_version": sim_cfg["generator_version"],
        "terrain_seed": int(terrain_seed),
        "ball_seed": int(ball_seed),
        "sim_cfg": plain_json(sim_cfg),
    }
    z = np.zeros(T, bool)
    return Episode(
        t=t,
        ball_pos=pos,
        ball_vel=vel,
        ball_spin=np.zeros(T),
        cam_pos=pos.copy(),
        ball_px=np.full((T, 2), S / 2),
        contact=z.copy(),
        bounce=z.copy(),
        out_of_view=z.copy(),
        occluded=z.copy(),
        terrain=sample_terrain(sim_cfg, terrain_seed),
        ball=sample_ball(sim_cfg, terrain_seed, ball_seed),
        meta=meta,
        frames=encoded_frames(T, S, terrain_seed, ball_seed),
    )


class Recorder:
    """A `simulate` stand-in for build_split that records its calls."""

    def __init__(self, fn=fake_episode):
        self.fn, self.calls = fn, []

    def __call__(self, sim_cfg, terrain_seed, ball_seed, **kw):
        self.calls.append((terrain_seed, ball_seed))
        return self.fn(sim_cfg, terrain_seed, ball_seed)


def physics_episode(sim_cfg: dict, terrain_seed: int, ball_seed: int, **_) -> Episode:
    """Real M1 physics (no GL), with encoded frames attached in place of a render."""
    ep = simulate_episode(sim_cfg, terrain_seed, ball_seed, render=False)
    T, S = int(sim_cfg["episode_frames"]), int(sim_cfg["frame_size"])
    return dataclasses.replace(ep, frames=encoded_frames(T, S, terrain_seed, ball_seed))


SMALL_SPLITS = [
    "data.splits={dev: {terrain_seeds: [0, 2], ball_seeds: [0, 2]},"
    " train: {terrain_seeds: [1000, 3], ball_seeds: [0, 2]},"
    " test: {terrain_seeds: [100000, 2], ball_seeds: [1000, 2]}}",
]
"""A few episodes per split (dev 4, train 6, test 4), same seed layout as base.yaml."""

SHORT = ["sim.episode_frames=30", "sim.frame_size=8", *SMALL_SPLITS]
"""Short, tiny episodes: T=30 still has windows for C=4, k_max=20 (t in 3..9)."""
