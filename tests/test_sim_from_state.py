"""simulate_from_state: re-simulate known terrain from the ball's state at frame t (SPEC §7.1, for
the M6 simulator-aware baseline). Design and the measured drift: jepa_world/sim/DESIGN.md,
"Re-simulating from a state".

The round trip is the key test: take the state at frame t of a generated episode, re-simulate with
the true ball parameters, and get the rest of that episode back. That is bitwise when the ball
isn't touching at t, and within a stated tolerance when it is, because MuJoCo's solver warm start
isn't in the saved ground truth.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from jepa_world.sim import camera
from jepa_world.sim.ball import mean_ball, sample_ball
from jepa_world.sim.contact import timestep
from jepa_world.sim.episode import ContactTracker, simulate_episode, simulate_from_state
from jepa_world.utils.config import load_config

SEEDS = [(ts, 0) for ts in range(8)]  # 8 physics-only episodes, ~0.15 s
GAP_EPS = 1e-6  # m: clear of the surface / penetrating, with no borderline cases
TOUCHING_MAX_DRIFT = 0.01  # m, on any frame (0.18 px); measured max 3.8 mm over 181 starts
TOUCHING_MEDIAN_DRIFT = 1e-6  # m; measured median 5e-10 m


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "base.yaml")["sim"]


@pytest.fixture(scope="module")
def episodes(sim_cfg):
    return [simulate_episode(sim_cfg, ts, bs, render=False) for ts, bs in SEEDS]


def starts(episodes, touching: bool, every: int = 1):
    """(episode, t) pairs whose state at t is clearly touching / clearly airborne."""
    out = []
    for ep in episodes:
        gap = ep.terrain.gap(ep.ball_pos, ep.ball.radius)
        hits = np.flatnonzero(gap < -GAP_EPS if touching else gap > GAP_EPS)
        out += [(ep, int(t)) for t in hits[hits < len(ep.t) - 1][::every]]
    return out


def resim(sim_cfg, ep, t, **kw):
    return simulate_from_state(
        sim_cfg, ep.terrain, ep.ball, ep.ball_pos[t], ep.ball_vel[t], ep.ball_spin[t],
        len(ep.t) - t, cam_pos=ep.cam_pos[t], **kw,
    )  # fmt: skip


def first_frame_with_bounce_history(sim_cfg) -> int:
    """A fresh tracker has no airborne history, so a landing within min_air_s of the start can be a
    bounce in the original but not in the re-run. Landings at substep j < min_air_steps fall in
    frames <= (min_air_steps - 1) // substeps + 1, so later frames must agree."""
    n = int(sim_cfg["substeps"])
    return (ContactTracker(0, timestep(sim_cfg), sim_cfg).min_air_steps - 1) // n + 2


def test_start_frame_is_the_given_state_and_nothing_is_rendered(sim_cfg, episodes):
    ep = episodes[0]
    pos, vel, spin = np.array([1.0, ep.ball_pos[5, 1]]), np.array([0.5, -1.0]), 2.0
    r = simulate_from_state(sim_cfg, ep.terrain, ep.ball, pos, vel, spin, 7)
    assert len(r.t) == 7 and r.t[0] == 0.0 and r.t[1] == pytest.approx(1 / sim_cfg["fps"])
    assert np.array_equal(r.ball_pos[0], pos) and np.array_equal(r.ball_vel[0], vel)
    assert r.ball_spin[0] == spin
    assert np.array_equal(r.cam_pos[0], camera.initial(pos, 0.0, sim_cfg))  # centred: no drop
    assert r.frames is None and r.seg is None
    start = r.meta["from_state"]
    assert start["pos"] == list(pos) and start["vel"] == list(vel) and start["spin"] == spin
    assert start["start_gap"] == pytest.approx(float(ep.terrain.gap(pos, ep.ball.radius)))
    # A rollout, not a dataset episode: no top-level seeds to confuse it with one.
    assert r.meta["terrain_seed"] is None and r.meta["ball_seed"] is None
    assert start["terrain_seed"] == ep.meta["terrain_seed"]


def test_airborne_start_reproduces_the_episode_exactly(sim_cfg, episodes):
    cases = starts(episodes, touching=False, every=9)
    assert len(cases) >= 10 and any(t == 0 for _, t in cases)  # t = 0 ties it to simulate
    safe = first_frame_with_bounce_history(sim_cfg)
    for ep, t in cases:
        r = resim(sim_cfg, ep, t)
        # The guarantee's exact condition is MuJoCo's: no contact at the start instant.
        assert not r.contact[0], (ep.meta["terrain_seed"], t)
        for f in ("ball_pos", "ball_vel", "ball_spin", "cam_pos", "ball_px", "out_of_view"):
            assert np.array_equal(getattr(r, f), getattr(ep, f)[t:]), (
                ep.meta["terrain_seed"],
                t,
                f,
            )
        assert np.array_equal(r.contact[1:], ep.contact[t + 1 :]), (ep.meta["terrain_seed"], t)
        assert np.array_equal(r.bounce[safe:], ep.bounce[t + safe :]), (ep.meta["terrain_seed"], t)
        assert not r.occluded.any()


def test_touching_start_tracks_the_episode_within_tolerance(sim_cfg, episodes):
    cases = starts(episodes, touching=True, every=5)
    assert len(cases) >= 12
    drift = []
    for ep, t in cases:
        r = resim(sim_cfg, ep, t)
        assert r.contact[0], (ep.meta["terrain_seed"], t)
        assert np.array_equal(r.ball_pos[0], ep.ball_pos[t])  # the start itself is exact
        drift.append(np.abs(r.ball_pos - ep.ball_pos[t:]).max())
    assert max(drift) <= TOUCHING_MAX_DRIFT, max(drift)
    assert np.median(drift) <= TOUCHING_MEDIAN_DRIFT, np.median(drift)


def test_a_penetrating_start_is_reported_not_changed(sim_cfg, episodes):
    # With a radius other than the true one (the mean ball's, say), the true centre can start
    # inside the terrain. The state is used as given, and meta reports the overlap.
    ep, t = starts(episodes, touching=True)[0]
    big = replace(ep.ball, radius=ep.ball.radius + 0.05)
    r = simulate_from_state(sim_cfg, ep.terrain, big, ep.ball_pos[t], ep.ball_vel[t], 0.0, 3)
    assert r.meta["from_state"]["start_gap"] < -0.04
    assert np.array_equal(r.ball_pos[0], ep.ball_pos[t])


def test_mean_ball_is_the_midpoint_of_each_sampled_range(sim_cfg):
    mean, bc = mean_ball(sim_cfg), sim_cfg["ball"]
    draws = [sample_ball(sim_cfg, ts, 0) for ts in range(2000)]
    for field, key in (("radius", "radius"), ("friction", "friction"),
                       ("restitution", "restitution"), ("x0", "start_x"), ("drop", "drop"),
                       ("vx0", "vx0")):  # fmt: skip
        lo, hi = bc[key]
        assert getattr(mean, field) == pytest.approx((lo + hi) / 2), field
        empirical = np.mean([getattr(b, field) for b in draws])
        assert empirical == pytest.approx((lo + hi) / 2, abs=0.03 * (hi - lo)), field


def test_rejects_a_bad_start(sim_cfg, episodes):
    ep = episodes[0]
    p, v = ep.ball_pos[3], ep.ball_vel[3]
    with pytest.raises(ValueError, match="frames"):
        simulate_from_state(sim_cfg, ep.terrain, ep.ball, p, v, 0.0, 0)
    with pytest.raises(ValueError, match="pos"):
        simulate_from_state(sim_cfg, ep.terrain, ep.ball, np.zeros(3), v, 0.0, 5)
    with pytest.raises(ValueError, match="vel"):
        simulate_from_state(sim_cfg, ep.terrain, ep.ball, p, np.array([np.nan, 0.0]), 0.0, 5)
    with pytest.raises(ValueError, match="spin"):
        simulate_from_state(sim_cfg, ep.terrain, ep.ball, p, v, float("inf"), 5)
    with pytest.raises(ValueError, match="spin"):
        simulate_from_state(sim_cfg, ep.terrain, ep.ball, p, v, ep.ball_spin[3:5], 5)
    with pytest.raises(ValueError, match="segmentation"):
        simulate_from_state(sim_cfg, ep.terrain, ep.ball, p, v, 0.0, 5, segmentation=True)


@pytest.mark.render
def test_airborne_start_renders_the_episode_frames_except_the_ball_s_rotation(sim_cfg):
    """The ball's rotation angle (qpos[2]) isn't in the ground truth, only its spin rate, so a
    restart begins at angle 0. MuJoCo draws the sphere as a tessellated mesh, so its shading and
    silhouette turn with it. From frame 0 the original's angle is also 0, and the frames are
    bitwise identical. From a later frame, every pixel more than 1.5 px outside the ball's disc is."""
    ep = simulate_episode(sim_cfg, 3, 0, render=True)
    gap = ep.terrain.gap(ep.ball_pos, ep.ball.radius)
    assert gap[0] > GAP_EPS
    S = ep.frames.shape[1]
    vv, uu = np.mgrid[0:S, 0:S] + 0.5
    r_px = ep.ball.radius * camera.px_per_m(sim_cfg)
    for t in (0, int(np.flatnonzero(gap[10:] > GAP_EPS)[0]) + 10):
        r = simulate_from_state(
            sim_cfg, ep.terrain, ep.ball, ep.ball_pos[t], ep.ball_vel[t], ep.ball_spin[t],
            len(ep.t) - t, cam_pos=ep.cam_pos[t], render=True,
        )  # fmt: skip
        assert r.frames.shape == (len(ep.t) - t, *ep.frames.shape[1:])
        if t == 0:
            assert np.array_equal(r.frames, ep.frames)
            continue
        for k in range(len(r.t)):
            u, v = r.ball_px[k]
            outside = np.hypot(uu - u, vv - v) > r_px + 1.5
            assert np.array_equal(r.frames[k][outside], ep.frames[t + k][outside]), (t, k)
