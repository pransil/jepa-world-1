"""Linear probes of ball position and velocity on frozen context-encoder embeddings (SPEC §7.2).

  uv run python experiments/probe_embeddings.py --run <run dir> [--random-init]
      [--device auto|mps|cpu] [--max-episodes N] [--batch 256]

Embeds every window t in [C-1, T-1-k_max] (the baselines' window set, jepa_world/data/DESIGN.md)
of the train, dev and test splits with the run's context encoder:
- by default, the run's last checkpoint;
- with --random-init, the same architecture freshly initialised with the run's seed (the floor).

Each level L1/L2/L3 is pooled globally over space (the semantics of jepa_world.probes.pool, but the
mean is taken on the device in float32, since MPS has no float64). jepa_world.probes then fits the
probes on train and scores them on dev and test. They never fit on test and never score on a
terrain they were fit on. Writes <run>/probes[_random_init].json. MPS work takes the GPU lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from jepa_world import train
from jepa_world.data.dataset import load_split, valid_starts, window_frames
from jepa_world.metrics import report
from jepa_world.models.jepa import context_stack
from jepa_world.probes import LEVELS, ProbeData, fit_probes, score_probes
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.repro import run_metadata, seed_everything

SPLITS = ("train", "dev", "test")


def load_encoder(run: Path, random_init: bool, device: torch.device):
    """(cfg, context encoder on device in eval mode, checkpoint step or 0 for random init)."""
    cfg = load_config(run / "config.yaml")
    seed_everything(cfg["seed"])
    model = train.build_model(cfg)
    step = 0
    if not random_init:
        ckpts = sorted((run / "checkpoints").glob("step_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"{run}/checkpoints has no step_*.pt")
        ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)  # our own file
        model.load_state_dict(ckpt["model"])
        step = int(ckpt["step"])
    return cfg, model.context_encoder.to(device).eval(), step


@torch.no_grad()
def embed(encoder, es, context_frames: int, k_max: int, device, batch: int, max_episodes=None):
    """ProbeData for one split: pooled features per level, targets pos and vel at t."""
    n = es.n if max_episodes is None else min(es.n, max_episodes)
    starts = valid_starts(es.T, context_frames, k_max)
    ep = np.repeat(np.arange(n), len(starts))
    t = np.tile(starts, n)
    feats = {level: [] for level in LEVELS}
    for i in range(0, len(t), batch):
        # k_max = 0: only frames t-C+1 .. t are read, nothing after t
        frames = window_frames(es, ep[i : i + batch], t[i : i + batch], context_frames, 0)
        levels = encoder(context_stack(frames.to(device), context_frames))
        for level in LEVELS:
            feats[level].append(levels[level].mean(dim=(2, 3)).cpu().numpy().astype(np.float64))
    return ProbeData(
        split=es.split,
        features={level: np.concatenate(v) for level, v in feats.items()},
        targets={"pos": es.ball_pos[ep, t], "vel": es.ball_vel[ep, t]},
        terrain_seed=es.terrain_seed[ep],
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-episodes", type=int, default=None, help="per split, for smoke checks")
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args(argv)

    device = pick_device(args.device)
    cfg, encoder, step = load_encoder(args.run, args.random_init, device)
    c = cfg["model"]["context_frames"]
    k_max = max(train.model_horizons(cfg))
    data = {}
    for split in SPLITS:
        es = load_split(cfg, split)
        label = f"probe-embed {args.run.name} {split}"
        with gpu_lock(label, enabled=device.type == "mps"):
            data[split] = embed(encoder, es, c, k_max, device, args.batch, args.max_episodes)
        del es
    probes = fit_probes(data["train"])
    out = {
        "run": str(args.run),
        "random_init": args.random_init,
        "checkpoint_step": step,
        "seed": cfg["seed"],
        "windows": {"t_first": c - 1, "k_max": k_max},
        "rows": {s: d.n for s, d in data.items()},
        "report": report([score_probes(probes, data[s]) for s in ("dev", "test")]),
        "meta": run_metadata(),
    }
    name = "probes_random_init.json" if args.random_init else "probes.json"
    if args.max_episodes is not None:
        name = name.replace(".json", f"_smoke{args.max_episodes}.json")
    (args.run / name).write_text(json.dumps(out, indent=2))
    rep = out["report"]
    by_split = {**rep["development"], rep["headline"]["split"]: rep["headline"]["metrics"]}
    for split, metrics in by_split.items():
        for level, targets in metrics["probes"]["by_level"].items():
            r2 = "  ".join(f"{k} R2 {v['r2']:+.3f}" for k, v in targets.items())
            print(f"{split:>5} {level:>7}  {r2}")
    print(f"wrote {args.run / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
