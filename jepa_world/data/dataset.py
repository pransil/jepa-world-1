"""Load a built split and feed the JEPA (SPEC §3.5, §8 M2). Design: data/DESIGN.md.

Batch layout (the M3 handshake): frames (B, C + k_max, 3, S, S) float32 in [0, 1], where index
C-1 is time t. The context is frames[:, :C] (t-C+1 .. t), and target stack k is frames[:, k:k+C],
which ends at t+k. A window never crosses an episode boundary, and never reaches outside [0, T).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from jepa_world.data.build import dataset_name, split_dir, terrain_sha
from jepa_world.data.splits import SPLITS, LeakageError, check_disjoint, check_split_config
from jepa_world.data.splits import terrain_seeds as configured_terrain_seeds
from jepa_world.sim.io import load_episode


@dataclass(frozen=True, eq=False)
class EpisodeSet:
    split: str
    frames: torch.Tensor  # (N, T, 3, S, S) uint8, CPU
    ball_pos: np.ndarray  # (N, T, 2) float64, world (x, z)
    ball_vel: np.ndarray  # (N, T, 2) float64
    out_of_view: np.ndarray  # (N, T) bool
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
    frames = pos = vel = oov = None
    fps = None
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
            pos, vel = np.empty((len(rows), T, 2)), np.empty((len(rows), T, 2))
            oov = np.empty((len(rows), T), bool)
            fps = float(ep.meta["sim_cfg"]["fps"])
        if ep.frames.shape[:2] != (frames.shape[1], frames.shape[3]):
            raise ValueError(f"{path / row['file']}: frames {ep.frames.shape} differ from the rest")
        frames[i] = torch.from_numpy(ep.frames).permute(0, 3, 1, 2)
        pos[i], vel[i], oov[i] = ep.ball_pos, ep.ball_vel, ep.out_of_view
    return EpisodeSet(
        split=manifest["split"],
        frames=frames,
        ball_pos=pos,
        ball_vel=vel,
        out_of_view=oov,
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
    """The on-disk leakage check: across every split built for this config, no terrain seed and
    no terrain profile (by hash) appears twice, and each split holds only its own seeds. Returns
    the splits it checked."""
    built = {s: split_dir(cfg, s) for s in SPLITS}
    manifests = {s: read_manifest(p) for s, p in built.items() if p.exists()}
    for s, m in manifests.items():
        stray = {r["terrain_seed"] for r in m["episodes"]} - set(
            configured_terrain_seeds(cfg["data"], s)
        )
        if stray:
            raise LeakageError(
                f"{dataset_name(s)} holds terrain seeds {sorted(stray)[:10]} that the {s!r} split "
                "doesn't own"
            )
    rows = {s: m["episodes"] for s, m in manifests.items()}
    check_disjoint({s: [r["terrain_seed"] for r in r_] for s, r_ in rows.items()}, "terrain seed")
    check_disjoint({s: [r["terrain_sha"] for r in r_] for s, r_ in rows.items()}, "terrain")
    return sorted(manifests)


def load_split(cfg: dict, split: str) -> EpisodeSet:
    path = require_split(cfg, split)
    check_built_splits(cfg)
    es = load_dir(path)
    if es.split != split:
        raise LeakageError(f"{path}: manifest says split {es.split!r}, expected {split!r}")
    return es


def window_frames(
    frames: torch.Tensor,
    ep_idx: np.ndarray,
    t: np.ndarray,
    context_frames: int,
    k_max: int,
) -> torch.Tensor:
    """frames[ep, t-C+1 .. t+k_max] for each (ep, t), as (B, C+k_max, 3, S, S) float32 in [0, 1].
    Refuses any window that reaches outside the episode: a negative index would otherwise wrap
    silently to the episode's end and put future frames into the context."""
    t = torch.as_tensor(np.asarray(t), dtype=torch.long)
    ep = torch.as_tensor(np.asarray(ep_idx), dtype=torch.long)
    T = frames.shape[1]
    lo, hi = t - (context_frames - 1), t + k_max
    if t.numel() and (int(lo.min()) < 0 or int(hi.max()) > T - 1):
        raise IndexError(
            f"window [t-{context_frames - 1}, t+{k_max}] leaves the episode (T={T}) for "
            f"t in [{int(t.min())}, {int(t.max())}]"
        )
    idx = lo[:, None] + torch.arange(context_frames + k_max)[None, :]  # (B, C+k_max)
    return frames[ep[:, None], idx].float().div_(255.0)


def batches(
    es: EpisodeSet, batch_size: int, context_frames: int, k_max: int, seed: int
) -> Iterator[torch.Tensor]:
    """Endless batches, uniform over every (episode, valid t) window, with replacement.
    The same seed gives the same sequence."""
    starts = valid_starts(es.T, context_frames, k_max)
    n_windows = es.n * len(starts)
    rng = np.random.default_rng(seed)
    while True:
        w = rng.integers(0, n_windows, size=batch_size)
        yield window_frames(
            es.frames, w // len(starts), starts[w % len(starts)], context_frames, k_max
        )
