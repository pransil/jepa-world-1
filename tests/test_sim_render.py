"""Rendering (render tier, needs GL; runs locally on the Mac).

SPEC §8 M1 tests covered here:
  - deterministic regeneration from seed (frames, byte for byte)
  - per-frame ground truth matches rendered geometry (ball and terrain, via segmentation)
Fast siblings without GL: test_sim_camera.py (projection), test_sim_episode.py (GT determinism).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jepa_world.sim import camera
from jepa_world.sim.episode import GT_FIELDS, simulate_episode
from jepa_world.sim.scene import geom_id
from jepa_world.utils.config import load_config

pytestmark = pytest.mark.render


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "base.yaml")["sim"]


@pytest.fixture(scope="module")
def seeds(sim_cfg):
    """The first seed pair whose episode leaves the frame and bounces, found on physics alone."""
    for ts in range(50):
        ep = simulate_episode(sim_cfg, ts, 0, render=False)
        if ep.out_of_view.any() and ep.bounce.any() and (~ep.out_of_view).sum() > 20:
            return ts, 0
    pytest.fail("no seed in 0..49 both leaves the frame and bounces")


@pytest.fixture(scope="module")
def episode(sim_cfg, seeds):
    return simulate_episode(sim_cfg, *seeds, render=True, segmentation=True)


def _ids(sim_cfg, ep):
    from jepa_world.sim.scene import build_model

    m = build_model(ep.terrain, ep.ball, sim_cfg)
    return geom_id(m, "ball"), geom_id(m, "terrain")


def test_frames_shape_and_dtype(sim_cfg, episode):
    T, S = sim_cfg["episode_frames"], sim_cfg["frame_size"]
    assert episode.frames.shape == (T, S, S, 3) and episode.frames.dtype == np.uint8
    assert episode.seg.shape == (T, S, S)


def test_same_seeds_render_identical_frames(sim_cfg, seeds, episode):
    again = simulate_episode(sim_cfg, *seeds, render=True)
    np.testing.assert_array_equal(again.frames, episode.frames)


def test_rendering_does_not_change_physics(sim_cfg, seeds, episode):
    physics_only = simulate_episode(sim_cfg, *seeds, render=False)
    for name in GT_FIELDS:
        np.testing.assert_array_equal(
            getattr(physics_only, name), getattr(episode, name), err_msg=name
        )


def test_ball_pixels_match_ground_truth(sim_cfg, episode):
    ball_id, _ = _ids(sim_cfg, episode)
    S = sim_cfg["frame_size"]
    rho = episode.ball.radius * camera.px_per_m(sim_cfg)
    inside = 0
    for t in range(len(episode.t)):
        mask = episode.seg[t] == ball_id
        u, v = episode.ball_px[t]
        if episode.out_of_view[t]:
            assert not mask.any(), f"frame {t}: ball pixels while out_of_view"
            continue
        if not (1 + rho <= u <= S - 1 - rho and 1 + rho <= v <= S - 1 - rho):
            continue  # partly out of frame
        rows, cols = np.nonzero(mask)
        assert len(rows), f"frame {t}: ball fully in frame but no ball pixels"
        np.testing.assert_allclose([cols.mean() + 0.5, rows.mean() + 0.5], [u, v], atol=0.5)
        # Pixelisation moves the edge by at most ~half a pixel, so the area is within that band.
        assert np.pi * (rho - 0.75) ** 2 <= len(rows) <= np.pi * (rho + 0.75) ** 2, (t, len(rows))
        inside += 1
    assert inside >= 10


def test_terrain_pixels_match_the_profile(sim_cfg, episode):
    ball_id, terrain_id = _ids(sim_cfg, episode)
    S, ppm = sim_cfg["frame_size"], camera.px_per_m(sim_cfg)
    checked = 0
    for t in range(0, len(episode.t), 5):
        seg, cx = episode.seg[t], episode.cam_pos[t, 0]
        for col in range(S):
            if (seg[:, col] == ball_id).any():
                continue
            x = cx + (col + 0.5 - S / 2) / ppm
            boundary_v = camera.project(
                np.array([x, episode.terrain.height_at(x)]), episode.cam_pos[t], sim_cfg
            )[1]
            if not 1 <= boundary_v <= S - 1:
                continue
            rows = np.flatnonzero(seg[:, col] == terrain_id)
            assert len(rows), (t, col)
            expected_top = np.ceil(boundary_v - 0.5)
            tol = 1 + abs(episode.terrain.slope_at(x)) * 0.5  # profile change within half a pixel
            assert abs(rows.min() - expected_top) <= tol, (t, col, rows.min(), boundary_v)
            checked += 1
    assert checked > 100
