"""Terrain: a seeded height profile h(x), extruded along y into a MuJoCo heightfield (SPEC §3.1).

h(x) = sum of `waves` sinusoids (red-noise spectrum, slope-capped at `max_slope`), plus a few
smoothed ledges (steps of `ledge_height` over `ledge_width`), plus quadratic walls over the last
`wall_width` metres at each end, so the ball stays on the terrain for a whole episode.

The geometry oracle is the piecewise-linear interpolation of the samples, exactly the surface
MuJoCo triangulates from the hfield. The heights are the values the float32 hfield decodes to, so
`gap` agrees with MuJoCo's contacts, not just approximately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TERRAIN_STREAM = 0x7E2A1  # SeedSequence stream id: terrain draws never share bits with ball draws
LEDGE_GAP = 1.0  # metres: minimum distance between two ledges


@dataclass(frozen=True, eq=False)
class Terrain:
    x: np.ndarray  # (N,) column positions, uniform over [-length/2, length/2]
    heights: np.ndarray  # (N,) surface z at each column
    params: dict = field(default_factory=dict)  # seed + sampled wave parameters (JSON-serialisable)
    zmin: float | None = None  # hfield encoding; derived from heights when not given
    elevation: float | None = None

    def height_at(self, x) -> np.ndarray:
        return np.interp(x, self.x, self.heights)

    def slope_at(self, x) -> np.ndarray:
        """dh/dx of the segment containing x (clamped to the end segments)."""
        i = np.clip(np.searchsorted(self.x, x, side="right") - 1, 0, len(self.x) - 2)
        return (self.heights[i + 1] - self.heights[i]) / (self.x[i + 1] - self.x[i])

    def gap(self, xz, r: float) -> np.ndarray:
        """Signed distance from point(s) `xz` (..., 2) to the surface polyline, minus `r`.

        Positive: a ball of radius r centred at xz is clear of the surface. Negative: it penetrates.
        """
        p = np.asarray(xz, dtype=np.float64)
        flat = p.reshape(-1, 2)
        a = np.stack([self.x[:-1], self.heights[:-1]], axis=1)  # (S, 2) segment starts
        ab = np.stack([np.diff(self.x), np.diff(self.heights)], axis=1)  # (S, 2)
        ap = flat[:, None, :] - a[None]  # (P, S, 2)
        s = np.clip((ap * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)  # (P, S)
        d = np.linalg.norm(ap - s[..., None] * ab, axis=-1).min(axis=1)  # (P,)
        sign = np.where(flat[:, 1] >= self.height_at(flat[:, 0]), 1.0, -1.0)
        return (sign * d - r).reshape(p.shape[:-1])

    def hfield(self) -> tuple[np.ndarray, float, float]:
        """(data float32 (N,) in [0, 1], zmin, elevation): heights == zmin + data * elevation."""
        zmin = float(self.heights.min()) if self.zmin is None else self.zmin
        if self.elevation is None:
            elevation = max(float(self.heights.max()) - zmin, 1e-6)
        else:
            elevation = self.elevation
        data = np.clip((self.heights - zmin) / elevation, 0.0, 1.0).astype(np.float32)
        return data, zmin, elevation


def sample_terrain(sim_cfg: dict, terrain_seed: int) -> Terrain:
    tc = sim_cfg["terrain"]
    rng = np.random.default_rng(np.random.SeedSequence([TERRAIN_STREAM, int(terrain_seed)]))
    n, half = int(tc["columns"]), float(tc["length"]) / 2
    x = np.linspace(-half, half, n)

    k = int(tc["waves"])
    lam_lo, lam_hi = tc["wavelength"]
    wavelength = np.exp(rng.uniform(np.log(lam_lo), np.log(lam_hi), size=k))
    # Red-noise spectrum: amplitude ~ wavelength**spectrum, so long waves make the hills and valleys
    # and short waves add texture. With spectrum > 1, short waves add less slope than long ones, so
    # the slope cap doesn't flatten the relief. The overall height is set by `relief` below.
    amplitude = rng.uniform(0.2, 1.0, size=k) * (wavelength / lam_hi) ** float(tc["spectrum"])
    phase = rng.uniform(0.0, 2 * np.pi, size=k)
    w = 2 * np.pi / wavelength
    unit = (amplitude[:, None] * np.sin(w[:, None] * x[None] + phase[:, None])).sum(0)

    inner = np.abs(x) <= half - tc["wall_width"]
    seg_inner = inner[:-1] & inner[1:]
    target_relief = float(rng.uniform(*tc["relief"]))
    scale = target_relief / np.ptp(unit[inner])
    # Cap the interior slope (sample to sample, i.e. the slope MuJoCo sees). This can lower the relief.
    max_slope = np.abs(np.diff(unit) / np.diff(x))[seg_inner].max() * scale
    slope_scale = min(1.0, float(tc["max_slope"]) / max_slope)
    scale *= slope_scale
    amplitude, waves = amplitude * scale, unit * scale

    # Ledges: smoothed steps the ball can roll off (a deep drop) or run into (a wall). Smooth hills
    # alone never outran the lagging camera: at most 4% of episodes had an out-of-view frame.
    lo, hi = tc["ledges"]
    n_ledges = int(rng.integers(lo, hi + 1))
    # One ledge per equal-width bin, at least LEDGE_GAP / 2 from the bin's edges, so ledges never
    # merge into one taller cliff.
    margin = tc["wall_width"] + 1.0
    edges = np.linspace(-half + margin, half - margin, n_ledges + 1)
    ledge_x = rng.uniform(edges[:-1] + LEDGE_GAP / 2, edges[1:] - LEDGE_GAP / 2)
    ledge_h = rng.uniform(*tc["ledge_height"], size=n_ledges) * rng.choice(
        [-1.0, 1.0], size=n_ledges
    )
    u = np.clip((x[None] - ledge_x[:, None]) / tc["ledge_width"] + 0.5, 0.0, 1.0)
    steps = (ledge_h[:, None] * u * u * (3 - 2 * u)).sum(0)  # smoothstep over ledge_width

    ramp = np.clip((np.abs(x) - (half - tc["wall_width"])) / tc["wall_width"], 0.0, None)
    profile = waves + steps + tc["wall_height"] * ramp**2

    params = {
        "seed": int(terrain_seed),
        "amplitude": amplitude.tolist(),
        "wavelength": wavelength.tolist(),
        "phase": phase.tolist(),
        "target_relief": target_relief,
        "slope_scale": slope_scale,
        "ledge_x": ledge_x.tolist(),
        "ledge_height": ledge_h.tolist(),  # signed: > 0 steps up toward +x
        "relief": float(np.ptp(waves + steps)),  # peak-to-peak over all x, walls excluded
    }
    # Quantise through the float32 hfield encoding, so `heights` is MuJoCo's surface exactly.
    raw = Terrain(x=x, heights=profile, params=params)
    data, zmin, elevation = raw.hfield()
    heights = zmin + data.astype(np.float64) * elevation
    return Terrain(x=x, heights=heights, params=params, zmin=zmin, elevation=elevation)
