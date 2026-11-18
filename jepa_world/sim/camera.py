"""Side-on orthographic camera: tracking rule and world -> pixel projection (SPEC §3.2). Pure.

Conventions: the camera looks along +y, so image u follows +x and image v grows downward (-z).
Pixel i covers [i, i+1), so the image centre is (S/2, S/2) and a pixel's centre is i + 0.5.
docs/decisions/2026-11-17-orthographic-camera.md, docs/decisions/2026-11-17-camera-mode-track-lagged.md
"""

from __future__ import annotations

import numpy as np

MODES = ("track_lagged", "track_full")


def px_per_m(sim_cfg: dict) -> float:
    return sim_cfg["frame_size"] / sim_cfg["camera"]["view_height"]


def initial(ball_xz, drop: float, sim_cfg: dict) -> np.ndarray:
    """Camera centre for frame 0, as if it had already been tracking a ball about to land. It is
    level with the ball in x and min(drop, deadzone_z) below it in z, so the start sits on the
    deadzone's top edge (or inside it), and a drop of up to 2 * deadzone_z lands in the deadzone.
    Centring on the ball instead made every high start drop out of view at about frame 11, an
    artefact of initialisation rather than an event."""
    x, z = float(ball_xz[0]), float(ball_xz[1])
    return np.array([x, z - min(float(drop), sim_cfg["camera"]["deadzone_z"])])


def track(prev_cam, ball_xz, radius: float, sim_cfg: dict) -> np.ndarray:
    """Camera centre (x, z) for this frame, from last frame's centre and the ball's position now."""
    mode, cc = sim_cfg["camera_mode"], sim_cfg["camera"]
    if mode not in MODES:
        raise ValueError(f"camera_mode {mode!r} is not implemented (have {MODES})")
    cx, cz = float(prev_cam[0]), float(prev_cam[1])
    bx, bz = float(ball_xz[0]), float(ball_xz[1])

    cx = cx + cc["lerp_x"] * (bx - cx)
    off = bz - cz
    if abs(off) > cc["deadzone_z"]:
        # The nearest camera z that puts the ball back on the deadzone's edge.
        target = bz - np.sign(off) * cc["deadzone_z"]
        cz = cz + cc["lerp_z"] * (target - cz)

    if mode == "track_full":
        # Clamp so the whole ball disc stays in frame (SPEC §3.2: "always in frame").
        half = cc["view_height"] / 2 - radius
        cx = float(np.clip(cx, bx - half, bx + half))
        cz = float(np.clip(cz, bz - half, bz + half))
    return np.array([cx, cz])


def project(xz, cam_xz, sim_cfg: dict) -> np.ndarray:
    """World (x, z) -> pixel (u, v). Broadcasts over leading dims of both inputs."""
    xz, cam = np.asarray(xz, dtype=np.float64), np.asarray(cam_xz, dtype=np.float64)
    s, k = sim_cfg["frame_size"], px_per_m(sim_cfg)
    u = (xz[..., 0] - cam[..., 0]) * k + s / 2
    v = (cam[..., 1] - xz[..., 1]) * k + s / 2
    return np.stack([u, v], axis=-1)


def out_of_view(uv, radius: float, sim_cfg: dict):
    """True where the projected ball disc lies entirely outside the image (no pixel of it visible)."""
    uv = np.asarray(uv, dtype=np.float64)
    s, rho = sim_cfg["frame_size"], radius * px_per_m(sim_cfg)
    du = np.maximum.reduce([-uv[..., 0], np.zeros(uv.shape[:-1]), uv[..., 0] - s])
    dv = np.maximum.reduce([-uv[..., 1], np.zeros(uv.shape[:-1]), uv[..., 1] - s])
    return du * du + dv * dv > rho * rho
