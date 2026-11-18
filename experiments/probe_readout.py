"""Follow-up readout probes: spatial-grid pooling and camera-relative targets (SPEC §7.2).

  uv run python experiments/probe_readout.py --run <run dir> [--random-init]
      [--device auto|mps|cpu] [--max-episodes N] [--train-t-stride 2]

Same protocol as experiments/probe_embeddings.py: the run's context encoder (last checkpoint, or
--random-init with the run's seed), windows t in [C-1, T-1-k_max], ridge probes from
jepa_world.probes fit on train and scored on dev and test, never fit on test.

What changes:
- **Features.** Each level is average-pooled to a g x g grid, then flattened (C*g*g dims), for
  every g in GRIDS that the level's size allows. g = 1 is the global mean of the first experiment.
- **Targets**, all at t:
  - pos and vel: the ball's world position and velocity, as before;
  - cam_vel: camera velocity, the backward difference (cam_pos[t] - cam_pos[t-1]) * fps, which
    is the scroll visible between the last two context frames;
  - vel_rel = vel - cam_vel: the ball's velocity relative to the camera.
  The camera is orthographic, so pixels per frame is vel_rel * px_per_m / fps: a linear rescale
  with the same R². Only its RMSE is reported, in px/frame.
- **Rows.** The train split keeps every --train-t-stride-th window to bound the ridge fits
  (8x8 L2 is 8192 dims). Dev and test keep every window.

Writes <run>/readout[_random_init].json. The MPS embedding pass takes the GPU lock.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from probe_embeddings import load_encoder

from jepa_world import train
from jepa_world.data.dataset import load_split, valid_starts, window_frames
from jepa_world.metrics import report
from jepa_world.models.jepa import context_stack
from jepa_world.probes import LEVELS, ProbeData, fit_probes, score_probes
from jepa_world.sim.camera import px_per_m
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.repro import run_metadata

SPLITS = ("train", "dev", "test")
GRIDS = (1, 2, 4, 8)
# Probe sets: (name, grid, levels). "pooled" (the concatenation) is built when a set has > 1
# level. Sets whose features would exceed ~8k dims per level are left out (8x8 L3 = 16384).
PROBE_SETS = (
    ("g1", 1, ("L1", "L2", "L3")),
    ("g2", 2, ("L1", "L2", "L3")),
    ("g4", 4, ("L1", "L2", "L3")),
    ("g8_L1", 8, ("L1",)),
    ("g8_L2", 8, ("L2",)),
)


def windows(es, context_frames: int, k_max: int, t_stride: int, max_episodes):
    n = es.n if max_episodes is None else min(es.n, max_episodes)
    starts = valid_starts(es.T, context_frames, k_max)[::t_stride]
    return np.repeat(np.arange(n), len(starts)), np.tile(starts, n)


@torch.no_grad()
def embed(encoder, es, ep, t, context_frames: int, device, batch: int) -> dict:
    """{(level, g): (M, C*g*g) float32} for every g a level's spatial size allows."""
    out: dict[tuple[str, int], list] = {}
    for i in range(0, len(t), batch):
        frames = window_frames(es, ep[i : i + batch], t[i : i + batch], context_frames, 0)
        levels = encoder(context_stack(frames.to(device), context_frames))
        for level in LEVELS:
            z = levels[level]
            for g in GRIDS:
                if z.shape[-1] % g == 0 and g <= z.shape[-1]:
                    pooled = F.adaptive_avg_pool2d(z, g).flatten(1)
                    out.setdefault((level, g), []).append(pooled.cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in out.items()}


def targets(es, ep, t, fps: float) -> dict:
    vel = es.ball_vel[ep, t]
    cam_vel = (es.cam_pos[ep, t] - es.cam_pos[ep, t - 1]) * fps
    return {"pos": es.ball_pos[ep, t], "vel": vel, "cam_vel": cam_vel, "vel_rel": vel - cam_vel}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-episodes", type=int, default=None, help="per split, for smoke checks")
    p.add_argument("--train-t-stride", type=int, default=2)
    p.add_argument("--batch", type=int, default=256)
    args = p.parse_args(argv)

    meta = run_metadata()  # at start: the code that ran, even if the tree changes mid-run
    device = pick_device(args.device)
    cfg, encoder, step = load_encoder(args.run, args.random_init, device)
    c = cfg["model"]["context_frames"]
    k_max = max(train.model_horizons(cfg))
    fps = float(cfg["sim"]["fps"])
    px_per_frame = px_per_m(cfg["sim"]) / fps  # (m/s) -> px/frame
    feats, tgts, seeds, rows = {}, {}, {}, {}
    for split in SPLITS:
        es = load_split(cfg, split)
        stride = args.train_t_stride if split == "train" else 1
        ep, t = windows(es, c, k_max, stride, args.max_episodes)
        with gpu_lock(f"probe-readout {args.run.name} {split}", enabled=device.type == "mps"):
            feats[split] = embed(encoder, es, ep, t, c, device, args.batch)
        tgts[split] = targets(es, ep, t, fps)
        seeds[split] = es.terrain_seed[ep]
        rows[split] = len(t)
        del es

    results = {}
    for name, g, levels in PROBE_SETS:
        data = {
            split: ProbeData(
                split=split,
                features={lv: feats[split][(lv, g)] for lv in levels},
                targets=tgts[split],
                terrain_seed=seeds[split],
            )
            for split in SPLITS
        }
        probes = fit_probes(data["train"])
        rep = report([score_probes(probes, data[s]) for s in ("dev", "test")])
        results[name] = {
            "grid": g,
            "dims": {lv: int(feats["train"][(lv, g)].shape[1]) for lv in levels},
            "report": rep,
        }
        for split, block in (
            ("dev", rep["development"]["dev"]),
            ("test", rep["headline"]["metrics"]),
        ):
            for level, by_target in block["probes"]["by_level"].items():
                r2 = "  ".join(f"{k} {v['r2']:+.3f}" for k, v in by_target.items())
                print(f"{name:>6} {split:>4} {level:>6}  {r2}", flush=True)
        del data, probes

    out = {
        "run": str(args.run),
        "random_init": args.random_init,
        "checkpoint_step": step,
        "seed": cfg["seed"],
        "windows": {"t_first": c - 1, "k_max": k_max, "train_t_stride": args.train_t_stride},
        "rows": rows,
        "px_per_frame_per_mps": px_per_frame,
        "results": results,
        "meta": meta,
    }
    name = "readout_random_init.json" if args.random_init else "readout.json"
    if args.max_episodes is not None:
        name = name.replace(".json", f"_smoke{args.max_episodes}.json")
    (args.run / name).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.run / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
