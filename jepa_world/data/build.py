"""Build one split's episodes into the shared artifact root, once (SPEC §3.5).

  uv run python -m jepa_world.data.build --config configs/base.yaml --split dev|train|test|all

Each split is its own write-once dataset, `datasets/episodes-<split>-<hash>/`, keyed on
`split_key` (sim config, the split's seed list, data.format_version). It holds one M1 episode
file per (terrain_seed, ball_seed), plus:
  manifest.json   split, format_version, and one row per episode {file, terrain_seed, ball_seed, terrain_sha}
  meta.json       repro.run_metadata() (machine, libraries, git) and build timing
  config.json     the split key (written by paths.build_dataset_once)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from jepa_world.data.splits import SPLITS, check_split_config, split_episodes
from jepa_world.sim.episode import Episode, plain_json, simulate_episode
from jepa_world.sim.io import save_episode
from jepa_world.sim.terrain import Terrain
from jepa_world.utils.config import load_config
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import build_dataset_once, dataset_dir
from jepa_world.utils.repro import run_metadata

Simulate = Callable[..., Episode]


def dataset_name(split: str) -> str:
    return f"episodes-{split}"


def episode_file(terrain_seed: int, ball_seed: int) -> str:
    return f"t{terrain_seed:06d}-b{ball_seed:04d}.npz"


def terrain_sha(terrain: Terrain) -> str:
    """Content hash of the terrain profile, so two seeds that gave the same terrain would still
    be caught by the cross-split check."""
    h = hashlib.sha256()
    for a in (terrain.x, terrain.heights):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def split_key(cfg: dict, split: str) -> dict:
    """Everything that determines what a split's dataset holds, and nothing else."""
    return {
        "kind": "episodes",
        "format_version": cfg["data"]["format_version"],
        "split": split,
        "episodes": [list(e) for e in split_episodes(cfg["data"], split)],
        "sim": plain_json(cfg["sim"]),
    }


def split_dir(cfg: dict, split: str) -> Path:
    """Where `split`'s dataset lives for this config. It may not exist yet."""
    return dataset_dir(split_key(cfg, split), dataset_name(split))


def build_split(cfg: dict, split: str, simulate: Simulate = simulate_episode) -> Path:
    """Return `split`'s dataset dir, simulating and rendering it first if it doesn't exist.
    `simulate(sim_cfg, terrain_seed, ball_seed)` must return an Episode with frames."""
    check_split_config(cfg["data"])
    key = split_key(cfg, split)
    sim_cfg = cfg["sim"]

    def build(staging: Path) -> None:
        rows, t0 = [], time.perf_counter()
        for i, (ts, bs) in enumerate(key["episodes"]):
            ep = simulate(sim_cfg, ts, bs)
            got = (ep.meta.get("terrain_seed"), ep.meta.get("ball_seed"))
            if got != (ts, bs):
                raise ValueError(f"asked for seeds {(ts, bs)}, the episode records {got}")
            if ep.frames is None:
                raise ValueError(f"episode {(ts, bs)} has no frames: the dataset needs rendering")
            name = episode_file(ts, bs)
            save_episode(staging / name, ep)
            rows.append(
                {
                    "file": name,
                    "terrain_seed": ts,
                    "ball_seed": bs,
                    "terrain_sha": terrain_sha(ep.terrain),
                }
            )
            if (i + 1) % 100 == 0:
                print(f"[build {split}] {i + 1}/{len(key['episodes'])}", file=sys.stderr)
        seconds = time.perf_counter() - t0
        manifest = {"split": split, "format_version": key["format_version"], "episodes": rows}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
        meta = {
            **run_metadata(),
            "episodes": len(rows),
            "build_seconds": seconds,
            "seconds_per_episode": seconds / max(len(rows), 1),
        }
        (staging / "meta.json").write_text(json.dumps(meta, indent=2))

    return build_dataset_once(key, build, name=dataset_name(split))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--split", required=True, choices=[*SPLITS, "all"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    check_split_config(cfg["data"])
    for split in SPLITS if args.split == "all" else [args.split]:
        path = split_dir(cfg, split)
        if not path.exists():  # rendering uses the GPU: don't skew a benchmark running beside it
            with gpu_lock(f"build {path.name}"):
                path = build_split(cfg, split)
        print(f"{split}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
