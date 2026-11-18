"""Position baselines (SPEC §7.1), reported beside every model result. Pure numpy.

- persistence:        x(t+k) ~ x(t)
- const_velocity:     x(t+k) ~ x(t) + v(t) k dt, with the true velocity at t (simulator state)
- const_velocity_fd:  x(t+k) ~ x(t) + (x(t) - x(t-1)) k, from context positions only

None of them reads anything after t. The simulator-aware baseline (known terrain) is deferred to M6
(docs/decisions/2026-11-17-m2-baseline-scope.md). Positions are world (x, z) in metres, and errors
are Euclidean.
"""

from __future__ import annotations

import numpy as np

BASELINES = ("persistence", "const_velocity", "const_velocity_fd")


def predict(
    name: str,
    pos: np.ndarray,
    vel: np.ndarray,
    ep: np.ndarray,
    t: np.ndarray,
    k: int,
    dt: float,
) -> np.ndarray:
    """(M, 2) predicted positions at t+k for windows (ep[i], t[i]). pos, vel: (N, T, 2)."""
    x = pos[ep, t]
    if name == "persistence":
        return x.copy()
    if name == "const_velocity":
        return x + vel[ep, t] * (k * dt)
    if name == "const_velocity_fd":
        if np.any(np.asarray(t) < 1):
            raise ValueError("const_velocity_fd needs t >= 1 (it reads x(t-1))")
        return x + (x - pos[ep, t - 1]) * k
    raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINES}")


def _stats(err: np.ndarray) -> dict:
    return {
        "mean": float(err.mean()),
        "median": float(np.median(err)),
        "p90": float(np.quantile(err, 0.9)),
    }


def evaluate(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    starts: np.ndarray,
    horizons: list[int],
) -> dict:
    """Position error per baseline and horizon over every (episode, t in starts) window. Every
    horizon is scored on the same windows, so the curves are comparable. Ratios are ratios of mean
    errors to persistence's (SPEC §7.2), and None where persistence's mean error is 0."""
    starts = np.asarray(starts)
    if pos.shape[1] - 1 - starts.max() < max(horizons) or starts.min() < 1:
        raise ValueError(f"starts [{starts.min()}, {starts.max()}] don't fit horizons {horizons}")
    ep, t = (a.ravel() for a in np.meshgrid(np.arange(len(pos)), starts, indexing="ij"))
    error, ratio = {}, {}
    for name in BASELINES:
        error[name] = {}
        for k in horizons:
            e = np.linalg.norm(predict(name, pos, vel, ep, t, k, dt) - pos[ep, t + k], axis=-1)
            error[name][str(k)] = _stats(e)
    for name in BASELINES:
        ratio[name] = {}
        for k in horizons:
            p = error["persistence"][str(k)]["mean"]
            ratio[name][str(k)] = error[name][str(k)]["mean"] / p if p > 0 else None
    return {
        "episodes": int(len(pos)),
        "windows": int(len(ep)),
        "starts": [int(starts.min()), int(starts.max())],
        "horizons": list(horizons),
        "error_m": error,
        "ratio_to_persistence": ratio,
    }
