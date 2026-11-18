"""Load a built split and feed the JEPA (SPEC §3.5, §8 M2). Design: data/DESIGN.md.

Batch layout (the M3 handshake): frames (B, C + k_max, 3, S, S) float32 in [0, 1], where index
C-1 is time t. The context is frames[:, :C] (t-C+1 .. t), and target stack k is frames[:, k:k+C],
which ends at t+k. A window never crosses an episode boundary, and never reaches outside [0, T).

With with_gt=True a batch is a dict: "frames" as above, plus the GT_KEYS fields on the SAME time
axis (index C-1 is t), from the same window: ball_pos, ball_vel, cam_pos (B, C+k_max, 2) float32,
world metres; out_of_view, occluded (B, C+k_max) bool.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch

from jepa_world.data.build import dataset_name, split_dir, terrain_sha
from jepa_world.data.splits import SPLITS, LeakageError, check_disjoint, check_split_config
from jepa_world.data.splits import ball_seeds as configured_ball_seeds
from jepa_world.data.splits import terrain_seeds as configured_terrain_seeds
from jepa_world.sim.io import load_episode

_GT = {  # M1's Episode field -> (per-frame shape, dtype held in EpisodeSet)
    "ball_pos": ((2,), np.float64),
    "ball_vel": ((2,), np.float64),
    "cam_pos": ((2,), np.float64),
    "out_of_view": ((), bool),
    "occluded": ((), bool),
}
GT_KEYS = tuple(_GT)
"""Per-frame ground truth carried into with_gt batches (M1's Episode field names)."""


@dataclass(frozen=True, eq=False)
class EpisodeSet:
    split: str
    frames: torch.Tensor  # (N, T, 3, S, S) uint8, CPU
    ball_pos: np.ndarray  # (N, T, 2) float64, world (x, z)
    ball_vel: np.ndarray  # (N, T, 2) float64
    cam_pos: np.ndarray  # (N, T, 2) float64, camera centre (x, z)
    out_of_view: np.ndarray  # (N, T) bool
    occluded: np.ndarray  # (N, T) bool (always False until M7's obstacles)
    terrain_seed: np.ndarray  # (N,) int64
    ball_seed: np.ndarray  # (N,) int64
    terrain_sha: tuple[str, ...]
    fps: float

    @property
    def n(self) -> int:
        return self.frames.shape[0]

    @property
    def T(self) -> int:  # noqa: N802 (the episode length is T throughout SPEC and sim/)
        return self.frames.shape[1]


def valid_starts(T: int, context_frames: int, k_max: int) -> np.ndarray:  # noqa: N803
    """Every t with a full context (t-C+1 >= 0) and a target at t+k_max (<= T-1)."""
    if context_frames < 1 or k_max < 0:
        raise ValueError(f"context_frames={context_frames} must be >= 1, k_max={k_max} >= 0")
    lo, hi = context_frames - 1, T - 1 - k_max
    if hi < lo:
        raise ValueError(
            f"no valid start: T={T} frames can't hold C={context_frames} + k_max={k_max}"
        )
    return np.arange(lo, hi + 1)


def read_manifest(path: Path) -> dict:
    return json.loads((Path(path) / "manifest.json").read_text())


def load_dir(path: str | Path) -> EpisodeSet:
    """Load every episode a dataset dir's manifest lists. Each file's own seeds and terrain must
    agree with its manifest row."""
    path = Path(path)
    manifest = read_manifest(path)
    rows = manifest["episodes"]
    if not rows:
        raise ValueError(f"{path}: manifest lists no episodes")
    frames, gt, fps = None, {}, None
    for i, row in enumerate(rows):
        ep = load_episode(path / row["file"])
        seeds = (ep.meta.get("terrain_seed"), ep.meta.get("ball_seed"))
        if seeds != (row["terrain_seed"], row["ball_seed"]):
            raise LeakageError(f"{path / row['file']}: file holds seeds {seeds}, manifest {row}")
        if terrain_sha(ep.terrain) != row["terrain_sha"]:
            raise LeakageError(f"{path / row['file']}: terrain differs from its manifest row")
        if ep.frames is None:
            raise ValueError(f"{path / row['file']}: no frames")
        if frames is None:
            T, S = ep.frames.shape[:2]
            frames = torch.empty((len(rows), T, 3, S, S), dtype=torch.uint8)
            gt = {k: np.empty((len(rows), T, *shape), dtype) for k, (shape, dtype) in _GT.items()}
            fps = float(ep.meta["sim_cfg"]["fps"])
        if ep.frames.shape[:2] != (frames.shape[1], frames.shape[3]):
            raise ValueError(f"{path / row['file']}: frames {ep.frames.shape} differ from the rest")
        frames[i] = torch.from_numpy(ep.frames).permute(0, 3, 1, 2)
        for k, (_, dtype) in _GT.items():
            a = getattr(ep, k)
            kind_ok = a.dtype == bool if dtype is bool else np.issubdtype(a.dtype, np.floating)
            if a.shape != gt[k].shape[1:] or not kind_ok:
                raise ValueError(
                    f"{path / row['file']}: {k} is {a.dtype} {a.shape}, expected "
                    f"{np.dtype(dtype)} {gt[k].shape[1:]}"
                )
            gt[k][i] = a
    return EpisodeSet(
        split=manifest["split"],
        frames=frames,
        **gt,
        terrain_seed=np.array([r["terrain_seed"] for r in rows], dtype=np.int64),
        ball_seed=np.array([r["ball_seed"] for r in rows], dtype=np.int64),
        terrain_sha=tuple(r["terrain_sha"] for r in rows),
        fps=fps,
    )


def require_split(cfg: dict, split: str) -> Path:
    """`split`'s dataset dir for this config, or FileNotFoundError naming the command that builds
    it."""
    check_split_config(cfg["data"])
    path = split_dir(cfg, split)
    if not path.exists():
        raise FileNotFoundError(
            f"split {split!r} is not built for this config (expected {path}). Build it with:\n"
            f"  uv run python -m jepa_world.data.build --config <your config> --split {split}"
        )
    return path


def check_built_splits(cfg: dict) -> list[str]:
    """The on-disk leakage check, over every split built for THIS config: each split holds only
    its own terrain and ball seeds; no terrain seed or terrain profile (by hash) is in two splits;
    train and test share no ball seed. Returns the splits it checked.

    Not covered: a model trained under a different config's splits. M6's eval has to compare the
    run's config.yaml splits against the eval config (data/DESIGN.md)."""
    built = {s: split_dir(cfg, s) for s in SPLITS}
    rows = {s: read_manifest(p)["episodes"] for s, p in built.items() if p.exists()}
    for s, r_ in rows.items():
        for key, own in (
            ("terrain_seed", configured_terrain_seeds(cfg["data"], s)),
            ("ball_seed", configured_ball_seeds(cfg["data"], s)),
        ):
            stray = {r[key] for r in r_} - set(own)
            if stray:
                raise LeakageError(
                    f"{dataset_name(s)} holds {key}s {sorted(stray)[:10]} that the {s!r} split "
                    "doesn't own"
                )

    def column(key: str, splits=SPLITS) -> dict[str, list]:
        return {s: [r[key] for r in rows[s]] for s in splits if s in rows}

    check_disjoint(column("terrain_seed"), "terrain seed")
    check_disjoint(column("terrain_sha"), "terrain")
    check_disjoint(column("ball_seed", ("train", "test")), "ball seed")
    return sorted(rows)


def load_split(cfg: dict, split: str) -> EpisodeSet:
    path = require_split(cfg, split)
    check_built_splits(cfg)
    es = load_dir(path)
    if es.split != split:
        raise LeakageError(f"{path}: manifest says split {es.split!r}, expected {split!r}")
    return es


def window_frames(
    es: EpisodeSet,
    ep_idx: np.ndarray,
    t: np.ndarray,
    context_frames: int,
    k_max: int,
) -> torch.Tensor:
    """frames[ep, t-C+1 .. t+k_max] for each (ep, t), as (B, C+k_max, 3, S, S) float32 in [0, 1].
    Refuses any window that reaches outside the episode: a negative index would otherwise wrap
    silently to the episode's end and put future frames into the context."""
    return _gather_frames(es, *_window_index(es, ep_idx, t, context_frames, k_max))


def window_gt(
    es: EpisodeSet,
    ep_idx: np.ndarray,
    t: np.ndarray,
    context_frames: int,
    k_max: int,
) -> dict[str, torch.Tensor]:
    """The GT_KEYS fields on window_frames' time axis, from the same windows: positions and
    velocities (B, C+k_max, 2) float32 (world metres), flags (B, C+k_max) bool. float32 because
    MPS has no float64; at |x| <= 20 m that rounds a position by under 2e-6 m."""
    return _gather_gt(es, *_window_index(es, ep_idx, t, context_frames, k_max))


def _window_index(
    es: EpisodeSet, ep_idx: np.ndarray, t: np.ndarray, context_frames: int, k_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """(ep (B,), frame index (B, C+k_max)) for windows t-C+1 .. t+k_max, after the one bounds
    check every gather relies on: numpy and torch both wrap a negative index silently, onto
    another episode or onto the end of this one."""
    t = np.asarray(t, dtype=np.int64)
    ep = np.asarray(ep_idx, dtype=np.int64)
    if ep.shape != t.shape or ep.ndim != 1:
        raise ValueError(f"ep_idx {ep.shape} and t {t.shape} must be matching 1-D arrays")
    if ep.size and (ep.min() < 0 or ep.max() > es.n - 1):
        raise IndexError(f"episode index in [{ep.min()}, {ep.max()}] outside [0, {es.n - 1}]")
    lo, hi = t - (context_frames - 1), t + k_max
    if t.size and (lo.min() < 0 or hi.max() > es.T - 1):
        raise IndexError(
            f"window [t-{context_frames - 1}, t+{k_max}] leaves the episode (T={es.T}) for "
            f"t in [{t.min()}, {t.max()}]"
        )
    return ep, lo[:, None] + np.arange(context_frames + k_max)[None, :]


def _gather_frames(es: EpisodeSet, ep: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    return es.frames[torch.from_numpy(ep)[:, None], torch.from_numpy(idx)].float().div_(255.0)


def _gather_gt(es: EpisodeSet, ep: np.ndarray, idx: np.ndarray) -> dict[str, torch.Tensor]:
    out = {}
    for key, (_, dtype) in _GT.items():
        a = getattr(es, key)[ep[:, None], idx]
        out[key] = torch.from_numpy(a if dtype is bool else a.astype(np.float32))
    return out


@overload
def batches(
    es: EpisodeSet,
    batch_size: int,
    context_frames: int,
    k_max: int,
    seed: int,
    with_gt: Literal[False] = False,
) -> Iterator[torch.Tensor]: ...
@overload
def batches(
    es: EpisodeSet,
    batch_size: int,
    context_frames: int,
    k_max: int,
    seed: int,
    with_gt: Literal[True],
) -> Iterator[dict[str, torch.Tensor]]: ...
def batches(es, batch_size, context_frames, k_max, seed, with_gt=False):
    """Endless batches, uniform over every (episode, valid t) window, with replacement.
    The same seed gives the same windows whether or not with_gt is set. with_gt=False yields the
    frames tensor; with_gt=True yields {"frames": ..., **window_gt(...)}."""
    starts = valid_starts(es.T, context_frames, k_max)
    n_windows = es.n * len(starts)
    rng = np.random.default_rng(seed)
    while True:
        w = rng.integers(0, n_windows, size=batch_size)
        ep, idx = _window_index(
            es, w // len(starts), starts[w % len(starts)], context_frames, k_max
        )
        frames = _gather_frames(es, ep, idx)
        yield {"frames": frames, **_gather_gt(es, ep, idx)} if with_gt else frames
