"""Contact physics: restitution calibration, friction, and the hfield geometry (SPEC §3.1).

Physics only (MuJoCo on CPU, no GL), so these run in the fast tier and in Linux CI.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import mujoco
import numpy as np
import pytest
from jepa_world.sim.ball import BallParams, sample_ball
from jepa_world.sim.scene import build_model, geom_id
from jepa_world.sim.terrain import Terrain, sample_terrain

from jepa_world.sim import contact
from jepa_world.utils.config import load_config


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "base.yaml")["sim"]


def _flat(sim_cfg) -> Terrain:
    n, half = sim_cfg["terrain"]["columns"], sim_cfg["terrain"]["length"] / 2
    return Terrain(x=np.linspace(-half, half, n), heights=np.zeros(n), params={"flat": True})


def _ball(radius=0.25, friction=0.5, restitution=0.5) -> BallParams:
    return BallParams(
        radius=radius, friction=friction, restitution=restitution, x0=0.0, drop=0.0, vx0=0.0
    )


def _bounce_in_scene(sim_cfg, ball: BallParams, speed: float) -> float:
    """Drop `ball` straight down onto flat terrain in the real scene; return v_out / v_in."""
    m = build_model(_flat(sim_cfg), ball, sim_cfg)
    d = mujoco.MjData(m)
    d.qpos[:] = [0.0, ball.radius + 1e-4, 0.0]
    d.qvel[:] = [0.0, -speed, 0.0]
    mujoco.mj_forward(m, d)
    touched = False
    for _ in range(20000):
        mujoco.mj_step(m, d)
        if d.ncon:
            touched = True
        elif touched:
            return d.qvel[1] / speed
    raise AssertionError("ball never separated")


def test_calibration_table_is_monotone(sim_cfg):
    damping, e = contact.calibration_table(sim_cfg)
    assert np.all(np.diff(damping) > 0)
    assert np.all(np.diff(e) < 0), "restitution must fall as damping rises"
    lo, hi = sim_cfg["ball"]["restitution"]
    assert e.min() <= lo and e.max() >= hi, "configured range must be reachable"


def test_damping_for_inverts_the_measurement(sim_cfg):
    ref = sim_cfg["contact"]["restitution_ref_speed"]
    for target in (0.2, 0.5, 0.8):
        b = contact.damping_for(target, sim_cfg)
        assert abs(contact.measure_restitution(b, sim_cfg, ref) - target) < 0.01


def test_restitution_holds_in_the_real_scene_for_both_radius_extremes(sim_cfg):
    """The calibration uses a bare plane. This checks it carries over to the hfield, the 3-joint
    ball, and any radius (direct-mode solref is mass-normalised)."""
    ref = sim_cfg["contact"]["restitution_ref_speed"]
    r_lo, r_hi = sim_cfg["ball"]["radius"]
    for e in (0.2, 0.5, 0.8):
        for r in (r_lo, r_hi):
            got = _bounce_in_scene(sim_cfg, _ball(radius=r, restitution=e), ref)
            assert abs(got - e) < 0.03, (e, r, got)


def test_restitution_outside_the_table_raises(sim_cfg):
    with pytest.raises(ValueError):
        contact.damping_for(1.5, sim_cfg)
    with pytest.raises(ValueError):
        contact.damping_for(-0.1, sim_cfg)


def test_contact_params_encode_friction_decision(sim_cfg):
    p = contact.contact_params(_ball(friction=0.4, restitution=0.5), sim_cfg)
    c = sim_cfg["contact"]
    assert p["condim"] == 6
    np.testing.assert_allclose(p["friction"], [0.4, 0.0, c["rolling_per_friction"] * 0.4])
    assert p["solref"][0] == -c["stiffness"] and p["solref"][1] < 0
    assert list(p["solimp"]) == list(c["solimp"])


def test_contact_params_are_set_identically_on_both_geoms(sim_cfg):
    ball = _ball(friction=0.3, restitution=0.6)
    m = build_model(_flat(sim_cfg), ball, sim_cfg)
    b, t = geom_id(m, "ball"), geom_id(m, "terrain")
    p = contact.contact_params(ball, sim_cfg)
    for g in (b, t):
        np.testing.assert_allclose(m.geom_solref[g], p["solref"])
        np.testing.assert_allclose(m.geom_friction[g], p["friction"])
        assert m.geom_condim[g] == 6


def test_higher_friction_slows_a_rolling_ball_more(sim_cfg):
    speeds = []
    for f in (0.1, 1.0):
        ball = _ball(radius=0.25, friction=f)
        m = build_model(_flat(sim_cfg), ball, sim_cfg)
        d = mujoco.MjData(m)
        d.qpos[:] = [0.0, ball.radius - 1e-4, 0.0]
        d.qvel[:] = [2.0, 0.0, 2.0 / ball.radius]  # rolling without slip
        for _ in range(int(2.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        speeds.append(d.qvel[0])
    assert 0 < speeds[1] < speeds[0] - 0.1


def test_mujoco_contact_agrees_with_the_terrain_polyline(sim_cfg):
    """The hfield column spacing, z offset and float32 decoding all match Terrain: MuJoCo reports
    a contact exactly where Terrain.gap says the ball penetrates."""
    terrain = sample_terrain(sim_cfg, 11)
    ball = dataclasses.replace(sample_ball(sim_cfg, 11, 0), radius=0.25)
    m = build_model(terrain, ball, sim_cfg)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(0)
    half = sim_cfg["terrain"]["length"] / 2 - sim_cfg["terrain"]["wall_width"]
    checked = 0
    for _ in range(400):
        x = rng.uniform(-half, half)
        z = terrain.height_at(x) + rng.uniform(-0.05, 0.5)
        gap = float(terrain.gap(np.array([x, z]), ball.radius))
        if abs(gap) < 1e-3:
            continue  # within the convex solver's tolerance band
        d.qpos[:] = [x, z, 0.0]
        mujoco.mj_forward(m, d)
        assert (d.ncon > 0) == (gap < 0), (x, z, gap, d.ncon)
        checked += 1
    assert checked > 300
