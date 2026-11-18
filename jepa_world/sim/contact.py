"""Contact parameters: calibrated restitution and the friction decision (SPEC §3.1). CPU physics only.

Restitution. MuJoCo contacts are soft, so there is no restitution coefficient to set. Standard
solref (timeconst, dampratio) gave e = 0.34 for a target of 0.8, and it varied strongly with impact
speed. Direct mode, solref = (-stiffness, -damping), is a mass-normalised spring-damper. At
stiffness 3e4 its e barely depends on speed or radius. `damping_for(e)` inverts a measured,
monotone table of e(damping) at `restitution_ref_speed`. The table is built once per process.

Friction (docs/decisions/2026-11-17-friction-sliding-plus-rolling.md): one draw f gives the triple
(sliding f, torsional 0, rolling rolling_per_friction * f) with condim 6. Torsional is 0 because
the ball, held to the x-z plane, cannot spin about a contact normal.
"""

from __future__ import annotations

from functools import lru_cache

import mujoco
import numpy as np

CONDIM = 6
_GRID = 61  # damping samples from 0 to twice critical damping (2 * 2 sqrt(k))
_PROBE_RADIUS = 0.25  # direct-mode solref is mass-normalised, so the probe radius is immaterial


def timestep(sim_cfg: dict) -> float:
    return 1.0 / (sim_cfg["fps"] * sim_cfg["substeps"])


def _key(sim_cfg: dict) -> tuple:
    c = sim_cfg["contact"]
    return (
        float(c["stiffness"]),
        tuple(float(v) for v in c["solimp"]),
        timestep(sim_cfg),
        float(sim_cfg["gravity"]),
        float(c["restitution_ref_speed"]),
    )


@lru_cache(maxsize=8)
def _probe_model(h: float, gravity: float) -> mujoco.MjModel:
    """A ball on one vertical slide joint above a plane, compiled once per (h, gravity). Callers set
    solref/solimp before each use; MjData is always fresh, so reuse is deterministic."""
    xml = f"""<mujoco><option timestep="{h!r}" gravity="0 0 {-gravity!r}" cone="elliptic"/>
      <worldbody><geom name="floor" type="plane" size="5 5 0.1" condim="{CONDIM}"/>
      <body><joint type="slide" axis="0 0 1"/>
      <geom type="sphere" size="{_PROBE_RADIUS}" condim="{CONDIM}"/></body></worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def _measure(stiffness, damping, solimp, h, gravity, speed) -> float:
    """Launch a ball at `speed` straight down onto a plane, from first touch, with gravity on. Return
    the rebound ratio v_out / v_in, measured when the contact ends."""
    m = _probe_model(h, gravity)
    for g in range(m.ngeom):
        m.geom_solref[g] = (-stiffness, -damping)
        m.geom_solimp[g] = solimp
    d = mujoco.MjData(m)
    d.qpos[0] = _PROBE_RADIUS + 1e-4
    d.qvel[0] = -speed
    touched = False
    for _ in range(int(2.0 / h)):
        mujoco.mj_step(m, d)
        if d.ncon:
            touched = True
        elif touched:
            return float(d.qvel[0] / speed)
    return 0.0  # never separated: fully inelastic


@lru_cache(maxsize=8)
def _table(key: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Measured e(damping), made strictly decreasing.

    The raw curve is quantised by the substep at which the ball separates. With a stiff, short
    contact (stiffness 1e5 lasts about 6 substeps at 1/600 s), neighbouring grid points can tie or
    invert slightly. Take the running minimum (a monotone envelope) and keep the points where it
    strictly falls, so the inverse is well defined. At the base config the raw curve is already
    strictly decreasing, and test_damping_for_inverts_the_measurement holds it to 0.01.
    """
    stiffness, solimp, h, gravity, speed = key
    damping = np.linspace(0.0, 4.0 * np.sqrt(stiffness), _GRID)
    raw = np.array([_measure(stiffness, b, solimp, h, gravity, speed) for b in damping])
    envelope = np.minimum.accumulate(raw)
    keep = np.r_[True, np.diff(envelope) < 0]
    if keep.sum() < 2:
        raise RuntimeError(f"restitution does not vary with damping: {raw.round(4).tolist()}")
    return damping[keep], envelope[keep]


def calibration_table(sim_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(damping (G,), restitution (G,)): damping ascending, restitution strictly descending."""
    damping, e = _table(_key(sim_cfg))
    return damping.copy(), e.copy()


def measure_restitution(damping: float, sim_cfg: dict, speed: float) -> float:
    stiffness, solimp, h, gravity, _ = _key(sim_cfg)
    return _measure(stiffness, float(damping), solimp, h, gravity, float(speed))


def damping_for(restitution: float, sim_cfg: dict) -> float:
    damping, e = _table(_key(sim_cfg))
    if not e.min() <= restitution <= e.max():
        raise ValueError(
            f"restitution {restitution} outside the reachable [{e.min():.3f}, {e.max():.3f}]"
        )
    return float(np.interp(restitution, e[::-1], damping[::-1]))


def contact_params(ball, sim_cfg: dict) -> dict:
    """Contact attributes for BOTH the ball and the terrain geom. Equal values on the two geoms
    make MuJoCo's mixing rules (solmix average, friction max) a no-op."""
    c = sim_cfg["contact"]
    return {
        "condim": CONDIM,
        "solref": (-float(c["stiffness"]), -damping_for(ball.restitution, sim_cfg)),
        "solimp": tuple(float(v) for v in c["solimp"]),
        "friction": (ball.friction, 0.0, c["rolling_per_friction"] * ball.friction),
    }
