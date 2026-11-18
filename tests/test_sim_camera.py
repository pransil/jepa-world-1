"""Camera tracking and orthographic projection (SPEC §3.2). Pure functions, no MuJoCo."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from jepa_world.sim import camera
from jepa_world.utils.config import load_config


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "base.yaml")["sim"]


def _mode(cfg, mode):
    c = copy.deepcopy(cfg)
    c["camera_mode"] = mode
    return c


def test_px_per_m(sim_cfg):
    assert camera.px_per_m(sim_cfg) == sim_cfg["frame_size"] / sim_cfg["camera"]["view_height"]


def test_projection_centre_edges_and_orientation(sim_cfg):
    s, half = sim_cfg["frame_size"], sim_cfg["camera"]["view_height"] / 2
    cam = np.array([3.0, -1.0])
    np.testing.assert_allclose(camera.project(cam, cam, sim_cfg), [s / 2, s / 2])
    np.testing.assert_allclose(camera.project(cam + [half, 0], cam, sim_cfg), [s, s / 2])
    np.testing.assert_allclose(camera.project(cam + [-half, 0], cam, sim_cfg), [0, s / 2])
    # Image v grows downward: a point above the camera centre has a smaller v.
    np.testing.assert_allclose(camera.project(cam + [0, half], cam, sim_cfg), [s / 2, 0])


def test_projection_is_vectorised(sim_cfg):
    rng = np.random.default_rng(0)
    xz, cam = rng.normal(size=(7, 2)), rng.normal(size=(7, 2))
    uv = camera.project(xz, cam, sim_cfg)
    assert uv.shape == (7, 2)
    for i in range(7):
        np.testing.assert_allclose(uv[i], camera.project(xz[i], cam[i], sim_cfg))


def test_initial_camera_frames_the_landing(sim_cfg):
    dz = sim_cfg["camera"]["deadzone_z"]
    ball = np.array([1.5, 2.0])
    np.testing.assert_allclose(camera.initial(ball, 0.0, sim_cfg), [1.5, 2.0])  # resting: centred
    np.testing.assert_allclose(camera.initial(ball, 0.5 * dz, sim_cfg), [1.5, 2.0 - 0.5 * dz])
    # A high start sits on the deadzone's top edge: in view, and the camera need not move yet.
    np.testing.assert_allclose(camera.initial(ball, 3.0 * dz, sim_cfg), [1.5, 2.0 - dz])


def test_lagged_x_is_a_lerp(sim_cfg):
    cfg = _mode(sim_cfg, "track_lagged")
    a = cfg["camera"]["lerp_x"]
    cam = camera.track(np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.2, cfg)
    np.testing.assert_allclose(cam, [a, 0.0])


def test_lagged_z_holds_inside_the_deadzone_and_follows_outside(sim_cfg):
    cfg = _mode(sim_cfg, "track_lagged")
    dz, a = cfg["camera"]["deadzone_z"], cfg["camera"]["lerp_z"]
    inside = camera.track(np.array([0.0, 0.0]), np.array([0.0, 0.9 * dz]), 0.2, cfg)
    assert inside[1] == 0.0
    below = camera.track(np.array([0.0, 0.0]), np.array([0.0, -(dz + 1.0)]), 0.2, cfg)
    np.testing.assert_allclose(below[1], -a * 1.0)  # toward the deadzone edge, by lerp_z


def test_lagged_lets_a_fast_ball_leave_the_frame(sim_cfg):
    cfg = _mode(sim_cfg, "track_lagged")
    cam = camera.initial(np.array([0.0, 0.0]), 0.0, cfg)
    ball = np.array([0.0, 0.0])
    left = False
    for _ in range(10):
        ball = ball + [0.0, 1.0]  # rising 15 m/s, much faster than the camera follows
        cam = camera.track(cam, ball, 0.2, cfg)
        uv = camera.project(ball, cam, cfg)
        left |= bool(camera.out_of_view(uv, 0.2, cfg))
    assert left


def test_track_full_keeps_the_ball_disc_in_frame(sim_cfg):
    cfg = _mode(sim_cfg, "track_full")
    s = cfg["frame_size"]
    rng = np.random.default_rng(1)
    for r in (0.15, 0.35):
        rho = r * camera.px_per_m(cfg)
        ball = np.zeros(2)
        cam = camera.initial(ball, 0.0, cfg)
        for _ in range(300):
            ball = ball + rng.normal(scale=1.5, size=2)  # wild jumps
            cam = camera.track(cam, ball, r, cfg)
            u, v = camera.project(ball, cam, cfg)
            assert rho - 1e-9 <= u <= s - rho + 1e-9
            assert rho - 1e-9 <= v <= s - rho + 1e-9


def test_unknown_camera_mode_raises(sim_cfg):
    with pytest.raises(ValueError):
        camera.track(np.zeros(2), np.zeros(2), 0.2, _mode(sim_cfg, "track_lagged_obstacles"))


def test_out_of_view_means_the_disc_misses_the_image_entirely(sim_cfg):
    s = sim_cfg["frame_size"]
    r = 0.3
    rho = r * camera.px_per_m(sim_cfg)

    def oov(u, v):
        return bool(camera.out_of_view(np.array([u, v]), r, sim_cfg))

    assert not oov(s / 2, s / 2)
    assert not oov(-0.5 * rho, s / 2)  # centre outside, but the disc still overlaps
    assert oov(-1.01 * rho, s / 2)
    assert oov(s + 1.01 * rho, s / 2)
    assert oov(s / 2, -1.01 * rho)
    assert oov(s / 2, s + 1.01 * rho)
    # Diagonal corner: each axis alone overlaps, but the corner is 1.13 rho away.
    assert oov(-0.8 * rho, -0.8 * rho)
    assert not oov(-0.6 * rho, -0.6 * rho)  # 0.85 rho from the corner
    many = camera.out_of_view(np.array([[s / 2, s / 2], [-2 * rho, 0.0]]), r, sim_cfg)
    np.testing.assert_array_equal(many, [False, True])
