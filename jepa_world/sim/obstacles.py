"""Foreground posts that hide the ball, and the `occluded` flag (M7; SPEC §3.2, §3.4). Pure numpy.

Off unless `sim_cfg["obstacles"]` is present. An absent key means off, so base.yaml carries the block
only as a comment and the sim hash (M2's dataset keys) is unchanged
(docs/decisions/2026-11-17-obstacles-dataset-keying.md). The posts are full-height boxes in front of
the terrain that collide with nothing (built in scene.py). So in every frame a post covers the
image's full height, and occlusion depends on x alone. See sim/DESIGN.md, "Obstacles and occlusion".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jepa_world.sim import camera
from jepa_world.sim.terrain import Terrain

OBSTACLE_STREAM = 0x0B57AC  # SeedSequence stream id, distinct from the terrain's and the ball's
KEYS = ("count", "width", "gap")
POST_MARGIN_M = 10.0  # posts reach this far below the terrain's lowest point and above its highest


@dataclass(frozen=True, eq=False)
class Obstacles:
    x_lo: np.ndarray  # (K,) left edge of each post, metres, sorted
    x_hi: np.ndarray  # (K,) right edge
    z_lo: float  # every post spans z in [z_lo, z_hi]
    z_hi: float

    def to_json(self) -> dict:
        return {
            "x_lo": [float(v) for v in self.x_lo],
            "x_hi": [float(v) for v in self.x_hi],
            "z_lo": float(self.z_lo),
            "z_hi": float(self.z_hi),
        }


def _check(block: dict) -> tuple[int, int, float, float, float]:
    if not isinstance(block, dict):
        raise ValueError(f"sim.obstacles must be a mapping with keys {KEYS}, got {block!r}")
    unknown, missing = set(block) - set(KEYS), set(KEYS) - set(block)
    if unknown or missing:
        raise ValueError(
            f"sim.obstacles: unknown keys {sorted(unknown)}, missing keys {sorted(missing)} "
            f"(expected exactly {KEYS})"
        )
    (c_lo, c_hi), (w_lo, w_hi), gap = block["count"], block["width"], float(block["gap"])
    if int(c_lo) != c_lo or int(c_hi) != c_hi or not 0 <= c_lo <= c_hi:
        raise ValueError(
            f"sim.obstacles.count must be integers 0 <= lo <= hi, got {block['count']}"
        )
    if not 0 < w_lo <= w_hi:
        raise ValueError(f"sim.obstacles.width must satisfy 0 < lo <= hi, got {block['width']}")
    if gap < 0:
        raise ValueError(f"sim.obstacles.gap must be >= 0, got {gap}")
    return int(c_lo), int(c_hi), float(w_lo), float(w_hi), gap


def sample_obstacles(sim_cfg: dict, terrain: Terrain) -> Obstacles | None:
    """The posts for this terrain, or None when `sim.obstacles` is absent.

    One post per equal-width bin over the interior (the wall zones excluded), each at least gap/2
    inside its bin, so neighbours are at least `gap` apart. The draws come from the terrain's seed
    alone, so every ball on a terrain sees the same posts, and the split by terrain seed covers them.
    """
    block = sim_cfg.get("obstacles")
    if block is None:
        return None
    c_lo, c_hi, w_lo, w_hi, gap = _check(block)
    seed = (terrain.params or {}).get("seed")
    if seed is None:
        raise ValueError(
            "obstacles need a seeded terrain (Terrain.params['seed']); this one has none"
        )
    tc = sim_cfg["terrain"]
    inner = float(tc["length"]) / 2 - float(tc["wall_width"])
    if c_hi > 0 and 2 * inner / c_hi < w_hi + gap:
        raise ValueError(
            f"terrain too short for {c_hi} posts: each needs a {w_hi + gap} m bin (width + gap), "
            f"but the interior between the walls is {2 * inner:.2f} m"
        )
    rng = np.random.default_rng(np.random.SeedSequence([OBSTACLE_STREAM, int(seed)]))
    n = int(rng.integers(c_lo, c_hi + 1))
    width = rng.uniform(w_lo, w_hi, size=n)
    edges = np.linspace(-inner, inner, n + 1)
    lo = rng.uniform(edges[:-1] + gap / 2, edges[1:] - gap / 2 - width)
    return Obstacles(
        x_lo=lo,
        x_hi=lo + width,
        z_lo=float(terrain.heights.min()) - POST_MARGIN_M,
        z_hi=float(terrain.heights.max()) + POST_MARGIN_M,
    )


def visible_width_px(uv, radius: float, cam_xz, obs: Obstacles | None, sim_cfg: dict) -> np.ndarray:
    """(T,) width in pixels of the ball's visible part, measured along u.

    That is the u-extent of the ball disc's in-image part, minus what the posts cover. A post covers
    the full image height (checked here), so a pixel column inside a post hides everything in it.
    A disc whose centre lies above or below the image shows only its chord at the image edge. The
    result is 0 when the ball is out of view, or exactly 0 when it is hidden. `uv` (T, 2) is the
    projected centre in pixels, `cam_xz` (T, 2) the camera centre, `radius` in metres.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    cam = np.broadcast_to(np.asarray(cam_xz, dtype=np.float64), uv.shape)
    S, k = float(sim_cfg["frame_size"]), camera.px_per_m(sim_cfg)
    rho = radius * k
    dv = np.maximum.reduce([-uv[:, 1], np.zeros(len(uv)), uv[:, 1] - S])
    half = np.sqrt(np.clip(rho * rho - dv * dv, 0.0, None))  # half-width of the in-image chord
    lo, hi = np.maximum(uv[:, 0] - half, 0.0), np.minimum(uv[:, 0] + half, S)
    width = np.where(dv <= rho, np.clip(hi - lo, 0.0, None), 0.0)
    if obs is None or len(obs.x_lo) == 0:
        return width
    view = float(sim_cfg["camera"]["view_height"]) / 2
    if (cam[:, 1] - view < obs.z_lo).any() or (cam[:, 1] + view > obs.z_hi).any():
        raise ValueError(
            f"a frame's view leaves the posts' z-span [{obs.z_lo}, {obs.z_hi}], so a post would "
            "not cover the full image height and the 1-D occlusion rule would not hold"
        )
    out = width.copy()
    for t in np.flatnonzero(width > 0):
        b_lo = (obs.x_lo - cam[t, 0]) * k + S / 2
        b_hi = (obs.x_hi - cam[t, 0]) * k + S / 2
        out[t] = _uncovered(lo[t], hi[t], b_lo, b_hi)
    return out


def _uncovered(lo: float, hi: float, b_lo: np.ndarray, b_hi: np.ndarray) -> float:
    """Length of [lo, hi] outside the union of the bands [b_lo[i], b_hi[i]]. Exactly 0 when covered."""
    gaps, start = 0.0, lo
    for a, b in sorted(zip(b_lo.tolist(), b_hi.tolist(), strict=True)):
        if b <= start:
            continue
        if a >= hi:
            break
        gaps += max(0.0, a - start)
        start = max(start, b)
        if start >= hi:
            return gaps
    return gaps + max(0.0, hi - start)


def occluded(uv, radius: float, cam_xz, obs: Obstacles | None, sim_cfg: dict) -> np.ndarray:
    """(T,) bool: the ball is in view, yet posts hide all of it (SPEC §3.4 `occluded`).

    Fully hidden, mirroring `out_of_view` ("no pixel visible"), so the two flags never overlap. A
    partly hidden ball is not occluded."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    in_view = ~np.asarray(camera.out_of_view(uv, radius, sim_cfg), dtype=bool)
    if obs is None:
        return np.zeros(len(uv), bool)
    return in_view & (visible_width_px(uv, radius, cam_xz, obs, sim_cfg) == 0.0)
