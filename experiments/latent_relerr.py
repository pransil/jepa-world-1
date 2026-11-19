"""Held-out latent prediction error: the training terms, scored on a held-out split.

  uv run python experiments/latent_relerr.py --run <run dir> [--split dev] [--batches 200]
      [--checkpoint step_0008000.pt] [--device cpu|mps|auto]

Training logs relerr/k and the other loss terms on train-split batches only. This script scores a
run's checkpoint on held-out terrains (dev, by default) with the same loss code
(models/world.compute_world_losses, every built horizon and aux stage active, the checkpoint's own
loss config), so each number means exactly what the training column with the same name means:
  relerr/k    error relative to predicting the batch-mean target (1.0 = no better)
  loss/k      the latent loss of horizon k; sqerr/k, logvar/k for the Gaussian far heads
  emb_std/Lx  the context embedding's spread per level
loss/k, sqerr/k, var/Lx and loss/var follow the run's loss.latent_reduce: under 'mean' they are
the 'sum' values divided by the level's channel count, so compare them only between runs with the
same reduction (recorded as "latent_reduce" in the output). relerr, logvar and emb_std don't
depend on it.
No gradients are taken and nothing is fit, so scoring dev or test selects nothing.

Batches: jepa_world.data.dataset.batches on the split, batch size = the run's train.batch_size,
seed --seed (0), so every run sees the same windows, drawn as training draws them (uniform over
every (episode, valid t) window, with replacement). relerr is a per-batch ratio, which is why the
batch size matches training. Each term is reported as the mean over batches, with the per-batch
standard deviation.

Writes <run>/latent_<split>.json (latent_<split>_<checkpoint stem>.json with --checkpoint). The MPS
pass takes the GPU lock. CPU needs no lock, and is fast enough (near + far, 200 batches: about a
minute).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import torch

from jepa_world import train
from jepa_world.data.dataset import batches, load_split
from jepa_world.eval import load_run
from jepa_world.models.world import compute_world_losses
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.repro import run_metadata

SPLITS = ("dev", "test", "train")


def score(run, split: str, n_batches: int, seed: int, device: torch.device) -> dict:
    """Mean and per-batch std of every loss term over n_batches held-out batches."""
    cfg, model = run.cfg, run.model
    es = load_split(cfg, split)
    c, k_max = cfg["model"]["context_frames"], train.batch_k_max(cfg, model.horizons)
    with_gt = train.needs_gt(cfg)
    it = batches(es, cfg["train"]["batch_size"], c, k_max, seed, with_gt)
    rows: dict[str, list[float]] = {}
    with torch.no_grad():
        for _ in range(n_batches):
            b = next(it)
            if with_gt:
                b = {"frames": b["frames"], "ball_pos": b["ball_pos"]}  # as train._episode_batches
            losses = compute_world_losses(
                model,
                train.to_device(b, device),
                cfg["loss"],
                active=list(model.horizons),
                aux_active=list(model.aux_stages),
            )
            vals = torch.stack(list(losses.values())).cpu().tolist()
            for k, v in zip(losses, vals, strict=True):
                rows.setdefault(k, []).append(v)
    out = {}
    for k, vs in rows.items():
        finite = [v for v in vs if math.isfinite(v)]  # goalratio is NaN on a still batch
        n = len(finite)
        mean = sum(finite) / n if n else float("nan")
        var = sum((v - mean) ** 2 for v in finite) / (n - 1) if n > 1 else float("nan")
        out[k] = {"mean": mean, "std": math.sqrt(var), "n": n}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--split", default="dev", choices=SPLITS)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    meta = run_metadata()  # at start: the code that ran, even if the tree changes mid-run
    device = pick_device(args.device)
    run = load_run(args.run, args.checkpoint, device)
    lock = gpu_lock(f"latent_relerr {args.run.name}") if device.type == "mps" else None
    with lock or contextlib.nullcontext():
        terms = score(run, args.split, args.batches, args.seed, device)
    record = {
        "run": str(args.run),
        "checkpoint": run.checkpoint.name,
        "step": run.step,
        "split": args.split,
        "batches": args.batches,
        "batch_size": run.cfg["train"]["batch_size"],
        "seed": args.seed,
        "device": str(device),
        "latent_reduce": run.cfg["loss"].get("latent_reduce", "sum"),  # scale of loss/k, var/L
        "meta": meta,
        "terms": terms,
    }
    suffix = f"_{Path(args.checkpoint).stem}" if args.checkpoint else ""
    path = args.run / f"latent_{args.split}{suffix}.json"
    path.write_text(json.dumps(record, indent=1))
    keys = sorted(k for k in terms if k.startswith(("relerr/", "emb_std/", "logvar/")))
    for k in keys:
        print(f"{k:14s} {terms[k]['mean']:.4f} ± {terms[k]['std']:.4f}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
