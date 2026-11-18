"""Position baselines (SPEC §7.1), reported beside every model result.

Closed form, pure numpy:
- persistence:        x(t+k) ~ x(t)
- const_velocity:     x(t+k) ~ x(t) + v(t) k dt, with the true velocity at t (simulator state)
- const_velocity_fd:  x(t+k) ~ x(t) + (x(t) - x(t-1)) k, from context positions only

Simulator-aware (MuJoCo on the CPU, no GL):
- simulator_<params>: re-simulate the known terrain from the true state at t (position, velocity,
  spin) with `sim.episode.simulate_from_state`, using the ball parameters `params` (SIM_PARAMS;
  docs/decisions/2026-11-17-simulator-baseline-params.md).

None of them reads anything after t. Positions are world (x, z) in metres, and errors are
Euclidean. Design: jepa_world/DESIGN.md.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np

from jepa_world.metrics import error_stats
from jepa_world.sim.ball import BallParams, mean_ball
from jepa_world.sim.episode import simulate_from_state
from jepa_world.sim.terrain import Terrain

BASELINES = ("persistence", "const_velocity", "const_velocity_fd")
SIM_PARAMS = ("true", "mean", "true_radius")
PROPOSED_SIM_PARAMS = (
    "true",
    "mean",
)  # proposed: docs/decisions/2026-11-17-simulator-baseline-params.md
MATCH_TOL = 1e-9  # m: match_clearance's clearance tolerance
MATCH_ITERS = 20  # match_clearance's steps along the surface normal (smooth terrain needs 1)


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


def applicable(context_frames: int) -> list[str]:
    """The baselines that see only the context: the finite difference needs x(t-1) in it (C >= 2)."""
    return [b for b in BASELINES if b != "const_velocity_fd" or context_frames >= 2]


def window_index(n: int, starts) -> tuple[np.ndarray, np.ndarray]:
    """(ep, t) for every (episode, t in starts) window, episode-major: the order `evaluate` scores
    in, and so the order of its `extra` predictions."""
    ep, t = np.meshgrid(np.arange(n), np.asarray(starts), indexing="ij")
    return ep.ravel(), t.ravel()


# --- simulator-aware baseline -------------------------------------------------------------------


def simulator_name(params: str) -> str:
    return f"simulator_{params}"


def sim_ball(params: str, true_ball: BallParams, sim_cfg: dict) -> BallParams:
    """The ball the simulator baseline uses: the episode's own (`true`, an oracle), each sampled
    range's midpoint (`mean`), or the visible radius with mean friction and restitution
    (`true_radius`)."""
    if params == "true":
        return true_ball
    if params == "mean":
        return mean_ball(sim_cfg)
    if params == "true_radius":
        return dataclasses.replace(mean_ball(sim_cfg), radius=true_ball.radius)
    raise ValueError(f"unknown simulator params {params!r}; expected one of {SIM_PARAMS}")


def _outward_normal(terrain: Terrain, p: np.ndarray) -> np.ndarray:
    """(n, 2) unit gradient of the signed distance to the terrain polyline at points p (n, 2): away
    from the closest surface point, outward (up out of the terrain from inside it). The same
    projection as `Terrain.gap`. A point exactly on the surface takes its segment's upward normal."""
    a = np.stack([terrain.x[:-1], terrain.heights[:-1]], axis=1)  # (S, 2) segment starts
    ab = np.stack([np.diff(terrain.x), np.diff(terrain.heights)], axis=1)  # (S, 2)
    ap = p[:, None, :] - a[None]  # (n, S, 2)
    s = np.clip((ap * ab).sum(-1) / (ab * ab).sum(-1), 0.0, 1.0)
    v = ap - s[..., None] * ab  # closest point → p, per segment
    d = np.linalg.norm(v, axis=-1)
    j = d.argmin(axis=1)
    rows = np.arange(len(p))
    v, d = v[rows, j], d[rows, j]
    sign = np.where(p[:, 1] >= terrain.height_at(p[:, 0]), 1.0, -1.0)
    seg = ab[j]
    up = np.stack([-seg[:, 1], seg[:, 0]], axis=1) / np.linalg.norm(seg, axis=1, keepdims=True)
    on_surface = d < 1e-12
    safe = np.where(on_surface, 1.0, d)[:, None]
    return np.where(on_surface[:, None], up, sign[:, None] * v / safe)


def match_clearance(
    terrain: Terrain,
    pos,
    true_radius: float,
    radius: float,
    tol: float = MATCH_TOL,
    max_iter: int = MATCH_ITERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Shifts (n, 2) that give a ball of `radius` centred at pos + shift the same clearance
    (`Terrain.gap`) as the true ball at pos, and a mask of the windows where that wasn't reached
    within `tol` after `max_iter` steps. Those keep their best iterate.

    Each step moves the centre along the outward surface normal by the clearance still missing,
    p ← p + (target − gap(p))·n(p). On smooth terrain the first step is exact: Δr·n, with
    Δr = radius − true_radius. Beside a vertical face it is horizontal. In a floor-and-riser
    corner the second step clears the other face. The shift is exactly 0 for the true radius."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 2)
    n = len(pos)
    if float(radius) == float(true_radius):
        return np.zeros((n, 2)), np.zeros(n, dtype=bool)
    target = terrain.gap(pos, true_radius)
    p = pos.copy()
    miss = target - terrain.gap(p, radius)
    best, best_miss = p.copy(), np.abs(miss)
    for _ in range(int(max_iter)):
        active = np.abs(miss) > tol
        if not active.any():
            break
        step = miss[:, None] * _outward_normal(terrain, p)
        p = np.where(active[:, None], p + step, p)
        miss = target - terrain.gap(p, radius)
        better = np.abs(miss) < best_miss
        best = np.where(better[:, None], p, best)
        best_miss = np.where(better, np.abs(miss), best_miss)
    return best - pos, best_miss > tol


def simulator_predict(
    sim_cfg: dict,
    terrain: Terrain,
    ball: BallParams,
    true_radius: float,
    pos,
    vel,
    spin,
    starts,
    horizons,
    cam_pos=None,
    simulate: Callable = simulate_from_state,
) -> tuple[dict[int, np.ndarray], dict]:
    """Predictions {k: (len(starts), 2)} for one episode, from one rollout of k_max + 1 frames per
    start t. pos, vel: (T, 2), spin: (T,), cam_pos: (T, 2) or None. Only index t is read.

    With `ball.radius` other than `true_radius`, the start is mapped to that ball: the same
    clearance (`match_clearance`, a shift along the surface normal), the same surface speed
    (spin·r_true/r), and the predictions are the rollout minus the shift. With the true radius,
    all three are exact no-ops. Returns the predictions, and per start:
    - "shift" (2,);
    - "unmatched" (bool), and "clearance_miss", |clearance − the true one| after the shift (0 to
      tolerance unless unmatched);
    - "gap_true", the true ball's clearance at t;
    - "gap_as_given", this ball's clearance at the unshifted centre (negative means it would have
      started inside the terrain).

    Raises if a rollout returns a non-finite position."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    spin = np.asarray(spin, dtype=np.float64).ravel()
    starts = np.asarray(starts, dtype=int).ravel()
    horizons = sorted(int(k) for k in horizons)
    if len(starts) == 0 or not horizons:
        raise ValueError("simulator_predict needs at least one start and one horizon")
    k_max = horizons[-1]
    if starts.min() < 0 or starts.max() + k_max > len(pos) - 1:
        raise ValueError(
            f"starts [{starts.min()}, {starts.max()}] with k_max={k_max} don't fit T={len(pos)}"
        )
    shifts, unmatched = match_clearance(terrain, pos[starts], true_radius, ball.radius)
    spin_scale = float(true_radius) / float(ball.radius)
    preds = {k: np.empty((len(starts), 2)) for k in horizons}
    for i, t in enumerate(starts):
        shift = shifts[i]
        roll = simulate(
            sim_cfg, terrain, ball, pos[t] + shift, vel[t], spin[t] * spin_scale, k_max + 1,
            cam_pos=None if cam_pos is None else np.asarray(cam_pos)[t],
        )  # fmt: skip
        traj = np.asarray(roll.ball_pos, dtype=np.float64)
        if not np.isfinite(traj[: k_max + 1]).all():
            raise FloatingPointError(
                f"simulator rollout from t={t} (radius {ball.radius}, shift {shift.round(4)}, "
                f"unmatched {bool(unmatched[i])}) produced a non-finite position"
            )
        for k in horizons:
            preds[k][i] = traj[k] - shift
    gap_true = terrain.gap(pos[starts], true_radius)
    return preds, {
        "shift": shifts,
        "unmatched": unmatched,
        "clearance_miss": np.abs(terrain.gap(pos[starts] + shifts, ball.radius) - gap_true),
        "gap_true": gap_true,
        "gap_as_given": terrain.gap(pos[starts], ball.radius),
    }


# --- scoring ------------------------------------------------------------------------------------


def evaluate(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    starts: np.ndarray,
    horizons: list[int],
    context_frames: int,
    out_of_view: np.ndarray | None = None,
    extra: Mapping[str, Mapping[int, np.ndarray]] | None = None,
) -> dict:
    """Position error per baseline and horizon over every (episode, t in starts) window.

    Every horizon is scored on the same windows, so the curves and ratios are comparable across
    k. The model must be scored on the same windows (data/DESIGN.md). Ratios are ratios of mean
    errors to persistence's (SPEC §7.2), and None where persistence's mean error is 0. With
    `out_of_view` (N, T), the mean error is also split by whether the ball is out of view at
    t+k. `extra` adds precomputed baselines, {name: {k: (M, 2)}} in `window_index` order (the
    simulator's), scored the same way."""
    starts = np.asarray(starts)
    if starts.min() < context_frames - 1 or pos.shape[1] - 1 - starts.max() < max(horizons):
        raise ValueError(
            f"starts [{starts.min()}, {starts.max()}] don't fit C={context_frames}, "
            f"horizons {horizons}, T={pos.shape[1]}"
        )
    names = applicable(context_frames)
    extra = dict(extra or {})
    for name in extra:
        if name in BASELINES:
            raise ValueError(f"extra baseline {name!r} shadows a closed-form one")
    ep, t = window_index(len(pos), starts)

    def prediction(name: str, k: int) -> np.ndarray:
        if name not in extra:
            return predict(name, pos, vel, ep, t, k, dt)
        p = extra[name].get(k, extra[name].get(str(k)))
        if p is None or np.shape(p) != (len(ep), 2):
            raise ValueError(
                f"extra baseline {name!r} at k={k}: expected ({len(ep)}, 2) predictions, "
                f"got {None if p is None else np.shape(p)}"
            )
        p = np.asarray(p, dtype=np.float64)
        if not np.isfinite(p).all():
            raise ValueError(f"extra baseline {name!r} at k={k} has non-finite predictions")
        return p

    error, ratio, by_view = {}, {}, {}
    for name in [*names, *extra]:
        error[name], by_view[name] = {}, {}
        for k in horizons:
            e = np.linalg.norm(prediction(name, k) - pos[ep, t + k], axis=-1)
            error[name][str(k)] = error_stats(e)
            if out_of_view is not None:
                gone = out_of_view[ep, t + k]
                by_view[name][str(k)] = {
                    "in_view": float(e[~gone].mean()) if (~gone).any() else None,
                    "out_of_view": float(e[gone].mean()) if gone.any() else None,
                    "n_out_of_view": int(gone.sum()),
                }
    for name in error:
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
