"""One episode: seeds in, frames and per-frame ground truth out (SPEC §3.3, §3.4).

Event flags, for frame t (frames are at t / fps):
- contact[t]: the ball touches the terrain in some physics state in [t-1, t]. That covers every
  substep state from the instant of frame t-1 to the instant of frame t, and so every state whose
  contact force acts during (t-1, t]. So when contact[t] is False, the step from t-1 to t is pure
  free flight. contact[0] is the touch state at the start.
- bounce[t]: a contact onset (a landing) at a state in (t-1, t], after at least `min_air_s` airborne,
  with normal approach speed >= `bounce_min_speed`.
- out_of_view[t]: the projected ball disc lies entirely outside the image.
- occluded[t]: the ball is in view, but posts hide all of it (obstacles.occluded). Always False
  without `sim.obstacles`, and never True together with out_of_view.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from jepa_world.sim import camera, obstacles
from jepa_world.sim.ball import BallParams, sample_ball, start_z
from jepa_world.sim.contact import timestep
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


class ContactTracker:
    """Visits each physics state once, in order, and flags touches and landings.

    A landing (a bounce) is a touch whose previous `min_air_s` of states, counted in whole substeps,
    were all airborne, and whose normal approach speed is at least `bounce_min_speed`.
    """

    def __init__(self, ball_geom: int, h: float, sim_cfg: dict):
        c = sim_cfg["contact"]
        self.ball_geom = ball_geom
        # Whole substeps, so the threshold doesn't depend on float sums of h.
        self.min_air_steps = int(np.ceil(float(c["min_air_s"]) / h - 1e-9))
        self.min_speed = float(c["bounce_min_speed"])
        self.touching = False
        self.air_steps = 0  # airborne states since the last touch

    def visit(self, d, vel: np.ndarray) -> tuple[bool, bool]:
        """`d` (MjData, or anything with .ncon and .contact[i].frame/.geom1) must describe the state
        whose (vx, vz) is `vel`. Returns (touching, landing)."""
        touching = d.ncon > 0
        landing = False
        if touching:
            if not self.touching and self.air_steps >= self.min_air_steps:
                landing = self.approach_speed(d, vel) >= self.min_speed
            self.air_steps = 0
        else:
            self.air_steps += 1
        self.touching = touching
        return touching, landing

    def approach_speed(self, d, vel: np.ndarray) -> float:
        """Largest speed of the ball toward the terrain along any contact normal (0 if separating)."""
        v = np.array([vel[0], 0.0, vel[1]])
        best = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            n = np.asarray(con.frame[:3], dtype=np.float64)  # points from geom1 to geom2
            if con.geom1 == self.ball_geom:
                n = -n  # make it point from the terrain to the ball
            best = max(best, -float(v @ n))
        return best


def plain_json(obj, path: str = "sim"):
    """A deep copy of `obj` in plain JSON types, which is what goes into episode files. numpy
    scalars and arrays become Python numbers and lists, and tuples become lists. Anything else that
    JSON can't hold (a YAML date, a set, an object) raises TypeError and names its path."""
    if isinstance(obj, dict):
        return {str(k): plain_json(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [plain_json(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.ndarray):
        return plain_json(obj.tolist(), path)
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None or isinstance(obj, bool | int | float | str):
        return obj
    raise TypeError(f"{path}: {type(obj).__name__} {obj!r} is not JSON-serialisable")


def simulate_episode(
    sim_cfg: dict,
    terrain_seed: int,
    ball_seed: int,
    render: bool = True,
    segmentation: bool = False,
) -> Episode:
    """Deterministic in (sim_cfg, terrain_seed, ball_seed). render=False skips GL entirely. The
    physics, and so every ground-truth array, is identical either way."""
    terrain = sample_terrain(sim_cfg, terrain_seed)
    ball = sample_ball(sim_cfg, terrain_seed, ball_seed)
    return simulate(
        sim_cfg,
        terrain,
        ball,
        render=render,
        segmentation=segmentation,
        seeds=(terrain_seed, ball_seed),
    )


def simulate(
    sim_cfg: dict,
    terrain: Terrain,
    ball: BallParams,
    render: bool = True,
    segmentation: bool = False,
    seeds: tuple[int | None, int | None] = (None, None),
) -> Episode:
    """Run one episode on a given terrain and ball, from the ball's own start (x0, drop, vx0).
    simulate_episode is the seeded entry point; this one serves hand-built scenes (tests)."""
    qpos = [ball.x0, start_z(terrain, ball), 0.0]
    qvel = [ball.vx0, 0.0, ball.vx0 / ball.radius]
    cam0 = camera.initial(np.array(qpos[:2]), ball.drop, sim_cfg)
    frames = int(sim_cfg["episode_frames"])
    return _run(sim_cfg, terrain, ball, qpos, qvel, cam0, frames, render, segmentation, seeds, {})


def simulate_from_state(
    sim_cfg: dict,
    terrain: Terrain,
    ball: BallParams,
    pos,
    vel,
    spin: float,
    frames: int,
    cam_pos=None,
    render: bool = False,
    segmentation: bool = False,
) -> Episode:
    """Re-simulate known terrain from a mid-episode state: the ball centre `pos` (x, z), velocity
    `vel` (vx, vz) and `spin` (rad/s about +y), as in an Episode's ground truth at frame t.

    Returns `frames` frames. Frame 0 is the given state, and `t` counts from it. `ball` supplies the
    hidden parameters: the true ones, or `jepa_world.sim.ball.mean_ball(sim_cfg)`. Its x0, drop and
    vx0 are unused.
    `cam_pos` continues an episode's camera (its `cam_pos[t]`); by default the camera is centred on
    the ball. The state is used as given, and `meta["from_state"]["start_gap"]` reports whether it
    starts inside the terrain (negative). Renders nothing unless asked.

    Started from frame t of an episode with its true ball and `cam_pos[t]`, it reproduces the rest of
    that episode's ground truth bitwise if MuJoCo has no contact at the start (the result's
    `contact[0]` is False), and to solver tolerance if it does (sim/DESIGN.md, "Re-simulating from a
    state"). It is a rollout, not a dataset episode: `meta["terrain_seed"]` and `["ball_seed"]` are
    None, and the terrain's seed is in `meta["from_state"]`."""
    if int(frames) != frames or frames < 1:
        raise ValueError(f"frames must be a positive integer, got {frames!r}")
    named = {"pos": pos, "vel": vel} | ({} if cam_pos is None else {"cam_pos": cam_pos})
    vecs = {k: np.asarray(v, dtype=np.float64) for k, v in named.items()}
    for k, v in vecs.items():
        if v.shape != (2,) or not np.isfinite(v).all():
            raise ValueError(f"{k} must be a finite (2,) array, got {named[k]!r}")
    spin_arr = np.asarray(spin, dtype=np.float64)
    if spin_arr.shape != () or not np.isfinite(spin_arr):
        raise ValueError(f"spin must be a finite scalar, got {spin!r}")
    spin = float(spin_arr)
    x, z = (float(c) for c in vecs["pos"])
    vx, vz = (float(c) for c in vecs["vel"])
    cam0 = vecs["cam_pos"] if cam_pos is not None else camera.initial(vecs["pos"], 0.0, sim_cfg)
    start = {
        "pos": [x, z],
        "vel": [vx, vz],
        "spin": float(spin),
        "cam_pos": [float(c) for c in cam0],
        "start_gap": float(terrain.gap(vecs["pos"], ball.radius)),
        "terrain_seed": (terrain.params or {}).get("seed"),
    }
    return _run(
        sim_cfg, terrain, ball, [x, z, 0.0], [vx, vz, spin], cam0, int(frames),
        render, segmentation, (None, None), {"from_state": start},
    )  # fmt: skip


def _run(
    sim_cfg: dict,
    terrain: Terrain,
    ball: BallParams,
    qpos: list[float],
    qvel: list[float],
    cam0: np.ndarray,
    T: int,
    render: bool,
    segmentation: bool,
    seeds: tuple[int | None, int | None],
    extra_meta: dict,
) -> Episode:
    """The one simulation loop behind both entry points, so obstacles (from
    `sim_cfg.get("obstacles")` and the terrain's seed) apply to both. Without them, every output is
    what it was before obstacles existed."""
    if segmentation and not render:
        raise ValueError("segmentation requires render=True")
    cfg_plain = plain_json(sim_cfg)  # fail before simulating, not halfway through a dataset build
    n, S = int(sim_cfg["substeps"]), int(sim_cfg["frame_size"])
    h = timestep(sim_cfg)
    rig_y = -float(sim_cfg["camera"]["distance"])

    obs = obstacles.sample_obstacles(sim_cfg, terrain)  # None when the key is absent
    obstacles.check_view(
        [cam0[1]], obs, sim_cfg
    )  # a bad (from-state) start fails before simulating
    m = build_model(terrain, ball, sim_cfg, obs)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    tracker = ContactTracker(geom_id(m, "ball"), h, sim_cfg)

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
            cam[t] = cam0 if t == 0 else camera.track(cam[t - 1], pos[t], ball.radius, sim_cfg)
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
        "terrain_seed": None if seeds[0] is None else int(seeds[0]),
        "ball_seed": None if seeds[1] is None else int(seeds[1]),
        "camera_mode": sim_cfg["camera_mode"],
        "timestep": h,
        "px_per_m": camera.px_per_m(sim_cfg),
        "damping": -float(m.geom_solref[geom_id(m, "ball")][1]),  # what MuJoCo actually used
        "sim_cfg_hash": config_hash(cfg_plain),
        "sim_cfg": cfg_plain,
        **({} if obs is None else {"obstacles": obs.to_json()}),
        **extra_meta,
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
        occluded=obstacles.occluded(ball_px, ball.radius, cam, obs, sim_cfg),
        terrain=terrain,
        ball=ball,
        meta=meta,
        frames=frames,
        seg=seg,
    )
