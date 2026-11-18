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


def applicable(context_frames: int) -> list[str]:
    """The baselines that see only the context: the finite difference needs x(t-1) in it (C >= 2)."""
    return [b for b in BASELINES if b != "const_velocity_fd" or context_frames >= 2]


def evaluate(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    starts: np.ndarray,
    horizons: list[int],
    context_frames: int,
    out_of_view: np.ndarray | None = None,
) -> dict:
    """Position error per baseline and horizon over every (episode, t in starts) window.

    Every horizon is scored on the same windows, so the curves and ratios are comparable across
    k. The model must be scored on the same windows (data/DESIGN.md). Ratios are ratios of mean
    errors to persistence's (SPEC §7.2), and None where persistence's mean error is 0. With
    `out_of_view` (N, T), the mean error is also split by whether the ball is out of view at
    t+k."""
    starts = np.asarray(starts)
    if starts.min() < context_frames - 1 or pos.shape[1] - 1 - starts.max() < max(horizons):
        raise ValueError(
            f"starts [{starts.min()}, {starts.max()}] don't fit C={context_frames}, "
            f"horizons {horizons}, T={pos.shape[1]}"
        )
    names = applicable(context_frames)
    ep, t = (a.ravel() for a in np.meshgrid(np.arange(len(pos)), starts, indexing="ij"))
    error, ratio, by_view = {}, {}, {}
    for name in names:
        error[name], by_view[name] = {}, {}
        for k in horizons:
            e = np.linalg.norm(predict(name, pos, vel, ep, t, k, dt) - pos[ep, t + k], axis=-1)
            error[name][str(k)] = _stats(e)
            if out_of_view is not None:
                gone = out_of_view[ep, t + k]
                by_view[name][str(k)] = {
                    "in_view": float(e[~gone].mean()) if (~gone).any() else None,
                    "out_of_view": float(e[gone].mean()) if gone.any() else None,
                    "n_out_of_view": int(gone.sum()),
                }
    for name in names:
        ratio[name] = {}
        for k in horizons:
            p = error["persistence"][str(k)]["mean"]
            ratio[name][str(k)] = error[name][str(k)]["mean"] / p if p > 0 else None
    out = {
        "episodes": int(len(pos)),
        "windows": int(len(ep)),
        "starts": [int(starts.min()), int(starts.max())],
        "context_frames": int(context_frames),
        "horizons": list(horizons),
        "error_m": error,
        "ratio_to_persistence": ratio,
    }
    if out_of_view is not None:
        out["mean_error_m_by_view_at_t_plus_k"] = by_view
    return out
