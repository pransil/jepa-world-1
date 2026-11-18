"""One episode: seeds in, frames and per-frame ground truth out (SPEC §3.3, §3.4).

Event flags, for frame t (frames are at t / fps):
- contact[t]: the ball touches the terrain in some physics state in [t-1, t]. That covers every
  substep state from the instant of frame t-1 to the instant of frame t, and so every state whose
  contact force acts during (t-1, t]. So when contact[t] is False, the step from t-1 to t is pure
  free flight. contact[0] is the touch state at the start.
- bounce[t]: a contact onset (a landing) at a state in (t-1, t], after at least `min_air_s` airborne,
  with normal approach speed >= `bounce_min_speed`.
- out_of_view[t]: the projected ball disc lies entirely outside the image.
- occluded[t]: always False. No obstacles exist yet (deferred to M7).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from jepa_world.sim import camera
from jepa_world.sim.ball import BallParams, sample_ball, start_z
from jepa_world.sim.contact import contact_params, timestep
from jepa_world.sim.scene import build_model, geom_id
from jepa_world.sim.terrain import Terrain, sample_terrain
from jepa_world.utils.paths import config_hash

GT_FIELDS = (
    "t",
    "ball_pos",
    "ball_vel",
    "ball_spin",
    "cam_pos",
    "ball_px",
    "contact",
    "bounce",
    "out_of_view",
    "occluded",
)


@dataclass(frozen=True, eq=False)
class Episode:
    t: np.ndarray  # (T,) seconds
    ball_pos: np.ndarray  # (T, 2) world (x, z)
    ball_vel: np.ndarray  # (T, 2) world (vx, vz)
    ball_spin: np.ndarray  # (T,) angular velocity about +y; rolling toward +x is vx / radius
    cam_pos: np.ndarray  # (T, 2) camera centre (x, z); the orientation is fixed
    ball_px: np.ndarray  # (T, 2) projected ball centre (u, v) in pixels; may lie outside the image
    contact: np.ndarray  # (T,) bool
    bounce: np.ndarray  # (T,) bool
    out_of_view: np.ndarray  # (T,) bool
    occluded: np.ndarray  # (T,) bool
    terrain: Terrain
    ball: BallParams
    meta: dict
    frames: np.ndarray | None = None  # (T, S, S, 3) uint8
    seg: np.ndarray | None = None  # (T, S, S) int16 geom id per pixel, -1 = background


class _ContactTracker:
    """Visits each physics state once, in order, and records touches and landings."""

    def __init__(self, m: mujoco.MjModel, ball_geom: int, h: float, sim_cfg: dict):
        c = sim_cfg["contact"]
        self.m, self.ball_geom, self.h = m, ball_geom, h
        self.min_air, self.min_speed = float(c["min_air_s"]), float(c["bounce_min_speed"])
        self.touching = False
        self.air = 0.0  # seconds since the last touching state

    def visit(self, d: mujoco.MjData, vel: np.ndarray) -> tuple[bool, bool]:
        """d.contact must describe the state whose velocity is `vel`. Returns (touching, landing)."""
        touching = d.ncon > 0
        landing = False
        if touching:
            if not self.touching and self.air >= self.min_air:
                landing = self._approach_speed(d, vel) >= self.min_speed
            self.air = 0.0
        else:
            self.air += self.h
        self.touching = touching
        return touching, landing

    def _approach_speed(self, d: mujoco.MjData, vel: np.ndarray) -> float:
        v = np.array([vel[0], 0.0, vel[1]])
        best = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            n = np.array(con.frame[:3])  # points from geom1 to geom2
            if con.geom1 == self.ball_geom:
                n = -n  # make it point from the terrain to the ball
            best = max(best, -float(v @ n))
        return best


def simulate_episode(
    sim_cfg: dict,
    terrain_seed: int,
    ball_seed: int,
    render: bool = True,
    segmentation: bool = False,
) -> Episode:
    """Deterministic in (sim_cfg, terrain_seed, ball_seed). render=False skips GL entirely. The
    physics, and so every ground-truth array, is identical either way."""
    if segmentation and not render:
        raise ValueError("segmentation requires render=True")
    T, n, S = int(sim_cfg["episode_frames"]), int(sim_cfg["substeps"]), int(sim_cfg["frame_size"])
    h = timestep(sim_cfg)
    rig_y = -float(sim_cfg["camera"]["distance"])

    terrain = sample_terrain(sim_cfg, terrain_seed)
    ball = sample_ball(sim_cfg, terrain_seed, ball_seed)
    m = build_model(terrain, ball, sim_cfg)
    d = mujoco.MjData(m)
    d.qpos[:] = [ball.x0, start_z(terrain, ball), 0.0]
    d.qvel[:] = [ball.vx0, 0.0, ball.vx0 / ball.radius]
    tracker = _ContactTracker(m, geom_id(m, "ball"), h, sim_cfg)

    pos, vel = np.zeros((T, 2)), np.zeros((T, 2))
    spin, cam = np.zeros(T), np.zeros((T, 2))
    contact, bounce = np.zeros(T, bool), np.zeros(T, bool)
    frames = np.zeros((T, S, S, 3), np.uint8) if render else None
    seg = np.zeros((T, S, S), np.int16) if segmentation else None

    renderer = None
    if render:
        from jepa_world.sim.render import FrameRenderer

        renderer = FrameRenderer(m, S)
    try:
        for t in range(T):
            if t > 0:
                for k in range(n):
                    v_before = d.qvel[:2].copy()
                    mujoco.mj_step(m, d)  # its contacts describe the state BEFORE this step
                    if k == 0:
                        contact[t] |= d.ncon > 0  # state at frame t-1, already visited
                    else:
                        touching, landing = tracker.visit(d, v_before)
                        contact[t] |= touching
                        bounce[t] |= landing
            pos[t], vel[t], spin[t] = d.qpos[:2], d.qvel[:2], d.qvel[2]
            cam[t] = (
                camera.initial(pos[t], ball.drop, sim_cfg)
                if t == 0
                else camera.track(cam[t - 1], pos[t], ball.radius, sim_cfg)
            )
            d.mocap_pos[0] = [cam[t, 0], rig_y, cam[t, 1]]
            mujoco.mj_forward(m, d)  # contacts at this instant + camera pose for rendering
            touching, landing = tracker.visit(d, d.qvel[:2].copy())
            contact[t] |= touching
            bounce[t] |= landing
            if renderer is not None:
                frames[t] = renderer.rgb(d)
                if seg is not None:
                    seg[t] = renderer.segmentation(d)
    finally:
        if renderer is not None:
            renderer.close()

    ball_px = camera.project(pos, cam, sim_cfg)
    meta = {
        "generator_version": sim_cfg["generator_version"],
        "terrain_seed": int(terrain_seed),
        "ball_seed": int(ball_seed),
        "camera_mode": sim_cfg["camera_mode"],
        "timestep": h,
        "px_per_m": camera.px_per_m(sim_cfg),
        "damping": -contact_params(ball, sim_cfg)["solref"][1],
        "sim_cfg_hash": config_hash(sim_cfg),
        "sim_cfg": copy.deepcopy(sim_cfg),
    }
    return Episode(
        t=np.arange(T) / sim_cfg["fps"],
        ball_pos=pos,
        ball_vel=vel,
        ball_spin=spin,
        cam_pos=cam,
        ball_px=ball_px,
        contact=contact,
        bounce=bounce,
        out_of_view=np.asarray(camera.out_of_view(ball_px, ball.radius, sim_cfg)),
        occluded=np.zeros(T, bool),
        terrain=terrain,
        ball=ball,
        meta=meta,
        frames=frames,
        seg=seg,
    )
