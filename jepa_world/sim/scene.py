"""MuJoCo scene: y-constant heightfield, a ball held to the x-z plane, a side-on orthographic camera.

- Ball: slide x, slide z, hinge y, all at the body origin, which is the world origin, so
  qpos = (x, z, angle) in world terms (docs/decisions/2026-11-17-ball-planar-joints.md).
- Terrain: an hfield with 2 identical rows, so it is constant along y. The geom sits at z = zmin
  and data * elevation adds the profile (Terrain.hfield()).
- Camera: "track", orthographic with fovy = view height in metres, on the mocap body "rig". Move it
  with data.mocap_pos, then mj_forward. mj_kinematics alone leaves cam_xpos stale (checked on 4.0.1).
- Shading is flat: a strong ambient headlight, no specular, no shadows, black background.
- Obstacles (M7, optional): full-height boxes in the plane y = POST_Y, in front of the terrain slab,
  in their own static body after the ball, colliding with nothing. With none, the MJCF is exactly the
  M1 scene (sim/DESIGN.md, "Obstacles and occlusion").
"""

from __future__ import annotations

import mujoco
import numpy as np

from jepa_world.sim import contact
from jepa_world.sim.ball import BallParams
from jepa_world.sim.obstacles import Obstacles
from jepa_world.sim.terrain import Terrain

TERRAIN_RGBA = "0.45 0.62 0.35 1"
BALL_RGBA = "0.92 0.28 0.2 1"
HFIELD_HALF_DEPTH = 1.0  # y half-extent of the terrain slab; the ball runs at y = 0
POST_RGBA = "0.3 0.4 0.85 1"
POST_Y = -(HFIELD_HALF_DEPTH + 0.5)  # between the camera (y = -distance) and the slab's front face
POST_HALF_DEPTH = 0.05
POST_CLEARANCE_M = 1.0  # camera to post front face, at least: clear of MuJoCo's near clip plane


def _vec(v) -> str:
    return " ".join(repr(float(x)) for x in v)


def _posts_xml(obs: Obstacles | None) -> str:
    """A static body of non-colliding boxes, or "" (and the M1 scene, byte for byte) when there are none."""
    if obs is None or len(obs.x_lo) == 0:
        return ""
    zc, hz = (obs.z_lo + obs.z_hi) / 2, (obs.z_hi - obs.z_lo) / 2
    geoms = "".join(
        f'\n      <geom name="post{i}" type="box" pos="{_vec([(lo + hi) / 2, POST_Y, zc])}" '
        f'size="{_vec([(hi - lo) / 2, POST_HALF_DEPTH, hz])}" rgba="{POST_RGBA}" '
        'contype="0" conaffinity="0"/>'
        for i, (lo, hi) in enumerate(zip(obs.x_lo, obs.x_hi, strict=True))
    )
    return f'\n    <body name="obstacles">{geoms}\n    </body>'


def build_model(
    terrain: Terrain, ball: BallParams, sim_cfg: dict, obstacles: Obstacles | None = None
) -> mujoco.MjModel:
    """The MuJoCo scene. `obstacles` (from obstacles.sample_obstacles) adds posts; None adds nothing."""
    data, zmin, elevation = terrain.hfield()
    half = float(terrain.x[-1] - terrain.x[0]) / 2
    if not np.isclose(terrain.x[0], -half) or not np.allclose(
        np.diff(terrain.x), 2 * half / (len(terrain.x) - 1)
    ):
        raise ValueError("terrain columns must be uniform and centred on x = 0 (hfield layout)")
    distance = float(sim_cfg["camera"]["distance"])
    if obstacles is not None and distance < POST_CLEARANCE_M - POST_Y + POST_HALF_DEPTH:
        raise ValueError(
            f"camera.distance {distance} m puts the camera within {POST_CLEARANCE_M} m of the posts "
            f"(y = {POST_Y}): they would be clipped or behind it, while `occluded` still counts them"
        )
    p = contact.contact_params(ball, sim_cfg)
    geom_contact = (
        f'condim="{p["condim"]}" friction="{_vec(p["friction"])}" '
        f'solref="{_vec(p["solref"])}" solimp="{_vec(p["solimp"])}"'
    )
    cam = sim_cfg["camera"]
    xml = f"""
<mujoco model="jepa_world_ball">
  <option timestep="{contact.timestep(sim_cfg)!r}" gravity="0 0 {-float(sim_cfg["gravity"])!r}" cone="elliptic"/>
  <visual>
    <headlight ambient="0.75 0.75 0.75" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    <hfield name="terrain" nrow="2" ncol="{len(terrain.x)}"
            size="{half!r} {HFIELD_HALF_DEPTH!r} {elevation!r} {float(sim_cfg["terrain"]["base"])!r}"/>
  </asset>
  <worldbody>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0 0 {zmin!r}" rgba="{TERRAIN_RGBA}"
          {geom_contact}/>
    <body name="rig" mocap="true" pos="0 {-float(cam["distance"])!r} 0">
      <camera name="track" projection="orthographic" fovy="{float(cam["view_height"])!r}"
              xyaxes="1 0 0 0 0 1"/>
    </body>
    <body name="ball" pos="0 0 0">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="ry" type="hinge" axis="0 1 0"/>
      <geom name="ball" type="sphere" size="{ball.radius!r}" rgba="{BALL_RGBA}" {geom_contact}/>
    </body>{_posts_xml(obstacles)}
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    m.hfield_data[:] = np.tile(data, 2)
    return m


def geom_id(m: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    if i < 0:
        raise KeyError(f"no geom {name!r}")
    return i


def body_id(m: mujoco.MjModel, name: str) -> int:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if i < 0:
        raise KeyError(f"no body {name!r}")
    return i
