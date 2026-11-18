"""Episodes: determinism, the planar constraint, and event flags consistent with ground truth.

SPEC §8 M1 tests covered here, on physics only (render=False, so fast, no GL, runs in CI):
  - deterministic regeneration from seed
  - event flags are consistent with ground truth
The rendered-geometry test is in test_sim_render.py (render tier).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from jepa_world.sim import camera
from jepa_world.sim.ball import sample_ball, start_z
from jepa_world.sim.episode import GT_FIELDS, simulate_episode
from jepa_world.sim.scene import body_id, build_model
from jepa_world.sim.terrain import sample_terrain
from jepa_world.utils.config import load_config

SEEDS = [(ts, bs) for ts in range(10) for bs in range(2)]  # 20 physics-only episodes, ~0.3 s


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "base.yaml")["sim"]


@pytest.fixture(scope="module")
def episodes(sim_cfg):
    return [simulate_episode(sim_cfg, ts, bs, render=False) for ts, bs in SEEDS]


def _dt(sim_cfg):
    return 1.0 / sim_cfg["fps"]


# ---------------------------------------------------------------- config and shapes


def test_base_config_sim_section_is_complete(sim_cfg):
    def walk(node, path="sim"):
        for k, v in node.items():
            if isinstance(v, dict):
                walk(v, f"{path}.{k}")
            else:
                assert v is not None, f"{path}.{k} is null"

    walk(sim_cfg)
    assert sim_cfg["camera_mode"] in camera.MODES


def test_shapes_and_dtypes(sim_cfg, episodes):
    ep, T = episodes[0], sim_cfg["episode_frames"]
    assert ep.frames is None and ep.seg is None
    for name in ("ball_pos", "ball_vel", "cam_pos", "ball_px"):
        arr = getattr(ep, name)
        assert arr.shape == (T, 2) and arr.dtype == np.float64, name
    for name in ("t", "ball_spin"):
        assert getattr(ep, name).shape == (T,)
    for name in ("contact", "bounce", "out_of_view", "occluded"):
        arr = getattr(ep, name)
        assert arr.shape == (T,) and arr.dtype == bool, name
    np.testing.assert_allclose(ep.t, np.arange(T) / sim_cfg["fps"])
    for name in GT_FIELDS:
        assert np.isfinite(getattr(ep, name)).all(), name


def test_hidden_params_are_in_range_and_recorded(sim_cfg, episodes):
    for ep in episodes:
        for key in ("radius", "friction", "restitution"):
            lo, hi = sim_cfg["ball"][key]
            assert lo <= getattr(ep.ball, key) <= hi
        assert ep.meta["generator_version"] == sim_cfg["generator_version"]
        assert {"terrain_seed", "ball_seed", "sim_cfg", "sim_cfg_hash", "damping"} <= set(ep.meta)


# ---------------------------------------------------------------- determinism (SPEC §8 M1)


def test_same_seeds_regenerate_identically(sim_cfg, episodes):
    again = simulate_episode(sim_cfg, *SEEDS[3], render=False)
    first = episodes[3]
    for name in GT_FIELDS:
        np.testing.assert_array_equal(getattr(again, name), getattr(first, name), err_msg=name)
    assert again.ball == first.ball and again.meta == first.meta
    np.testing.assert_array_equal(again.terrain.heights, first.terrain.heights)


def test_ball_seed_changes_the_ball_but_not_the_terrain(sim_cfg):
    a = simulate_episode(sim_cfg, 4, 0, render=False)
    b = simulate_episode(sim_cfg, 4, 1, render=False)
    np.testing.assert_array_equal(a.terrain.heights, b.terrain.heights)
    assert a.ball != b.ball
    assert not np.array_equal(a.ball_pos, b.ball_pos)


# ---------------------------------------------------------------- the scene


def test_ball_is_held_in_the_xz_plane(sim_cfg):
    terrain = sample_terrain(sim_cfg, 2)
    ball = sample_ball(sim_cfg, 2, 0)
    m = build_model(terrain, ball, sim_cfg)
    assert (m.nq, m.nv) == (3, 3)
    assert list(m.jnt_type) == [
        mujoco.mjtJoint.mjJNT_SLIDE,
        mujoco.mjtJoint.mjJNT_SLIDE,
        mujoco.mjtJoint.mjJNT_HINGE,
    ]
    np.testing.assert_array_equal(m.jnt_axis, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    d = mujoco.MjData(m)
    d.qpos[:] = [ball.x0, start_z(terrain, ball), 0.0]
    d.qvel[:] = [ball.vx0, 0.0, ball.vx0 / ball.radius]
    bid = body_id(m, "ball")
    for _ in range(2000):
        mujoco.mj_step(m, d)
        assert d.xpos[bid][1] == 0.0


def test_hfield_is_constant_along_y_and_encodes_the_terrain(sim_cfg):
    terrain = sample_terrain(sim_cfg, 6)
    m = build_model(terrain, sample_ball(sim_cfg, 6, 0), sim_cfg)
    assert m.nhfield == 1 and m.hfield_nrow[0] == 2
    rows = m.hfield_data.reshape(m.hfield_nrow[0], m.hfield_ncol[0])
    np.testing.assert_array_equal(rows[0], rows[1])
    data, _, _ = terrain.hfield()
    np.testing.assert_array_equal(rows[0], data)


# ---------------------------------------------------------------- events vs ground truth (SPEC §8 M1)


def test_no_contact_means_an_exactly_ballistic_step(sim_cfg, episodes):
    """contact[t] is False only if no contact acted anywhere in (t-1, t]. Then the 3-joint ball is in
    free flight, and semi-implicit Euler over n substeps of h gives exactly these updates."""
    g, n = sim_cfg["gravity"], sim_cfg["substeps"]
    h = 1.0 / (sim_cfg["fps"] * n)
    dt = n * h
    checked = 0
    for ep in episodes:
        for t in np.flatnonzero(~ep.contact[1:]) + 1:
            (x0, z0), (vx0, vz0) = ep.ball_pos[t - 1], ep.ball_vel[t - 1]
            np.testing.assert_allclose(ep.ball_vel[t], [vx0, vz0 - g * dt], atol=1e-9)
            np.testing.assert_allclose(
                ep.ball_pos[t],
                [x0 + vx0 * dt, z0 + vz0 * dt - g * h * h * n * (n + 1) / 2],
                atol=1e-9,
            )
            assert ep.ball_spin[t] == pytest.approx(ep.ball_spin[t - 1], abs=1e-9)
            checked += 1
    assert checked > 0, "no airborne frames at all in the sample"


def test_no_contact_means_the_ball_is_above_the_surface(episodes):
    for ep in episodes:
        gap = ep.terrain.gap(ep.ball_pos, ep.ball.radius)
        assert (gap[~ep.contact] > -1e-6).all()


def test_ball_never_sinks_into_the_terrain(sim_cfg, episodes):
    """Soft contacts penetrate, but by at most about v/sqrt(k) (an undamped spring at impact speed v)."""
    k = sim_cfg["contact"]["stiffness"]
    for ep in episodes:
        gap = ep.terrain.gap(ep.ball_pos, ep.ball.radius)
        vmax = np.linalg.norm(ep.ball_vel, axis=1).max()
        assert gap.min() > -(vmax / np.sqrt(k) + 1e-3)


def test_bounce_is_a_contact_whose_impulse_points_up(sim_cfg, episodes):
    """A landing on terrain gentler than 45 degrees, with mu <= 1, gets a net upward impulse. Over the
    two frames around the bounce, vz gains more than gravity takes away."""
    g, dt = sim_cfg["gravity"], _dt(sim_cfg)
    seen = 0
    for ep in episodes:
        assert not (ep.bounce & ~ep.contact).any(), "bounce without contact"
        assert not ep.bounce[0]
        for t in np.flatnonzero(ep.bounce):
            x, r = ep.ball_pos[t, 0], ep.ball.radius
            footprint = np.linspace(x - 2 * r, x + 2 * r, 9)
            if t + 1 >= len(ep.t) or np.abs(ep.terrain.slope_at(footprint)).max() >= 1.0:
                continue  # a ledge face or wall nearby: the impulse may be mostly horizontal
            assert ep.ball_vel[t + 1, 1] - ep.ball_vel[t - 1, 1] + 2 * g * dt > 0, t
            seen += 1
    assert seen > 0


def test_ball_stays_on_the_terrain(sim_cfg, episodes):
    half = sim_cfg["terrain"]["length"] / 2
    for ep in episodes:
        assert (np.abs(ep.ball_pos[:, 0]) < half - ep.ball.radius).all()


def test_occluded_is_all_false_without_obstacles(episodes):
    for ep in episodes:
        assert not ep.occluded.any()


def test_camera_and_pixel_ground_truth_follow_the_camera_rules(sim_cfg, episodes):
    for ep in episodes:
        r = ep.ball.radius
        np.testing.assert_array_equal(
            ep.cam_pos[0], camera.initial(ep.ball_pos[0], ep.ball.drop, sim_cfg)
        )
        for t in range(1, len(ep.t)):
            np.testing.assert_allclose(
                ep.cam_pos[t],
                camera.track(ep.cam_pos[t - 1], ep.ball_pos[t], r, sim_cfg),
                atol=1e-12,
            )
        np.testing.assert_allclose(ep.ball_px, camera.project(ep.ball_pos, ep.cam_pos, sim_cfg))
        np.testing.assert_array_equal(ep.out_of_view, camera.out_of_view(ep.ball_px, r, sim_cfg))


def test_events_are_not_degenerate(sim_cfg, episodes):
    """A generator whose flags are always False would pass every test above, so check they fire.
    Out-of-view is rarer (about 16% of episodes, M1.md), so search seeds for one instead of
    relying on the fixed sample."""
    contact = np.concatenate([ep.contact for ep in episodes])
    assert 0.3 < contact.mean() < 1.0
    assert sum(int(ep.bounce.sum()) for ep in episodes) >= 3
    assert any(
        simulate_episode(sim_cfg, ts, 0, render=False).out_of_view.any() for ts in range(100, 160)
    ), "no out-of-view frame in 60 episodes"
