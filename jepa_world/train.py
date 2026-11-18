"""Training entry point. Built in M3 (near heads), extended in M4 and M5. See SPEC §5-§6.

Contract every version of this script keeps (the /experiment skill relies on it):
  uv run python -m jepa_world.train --config configs/<x>.yaml [--set key=value ...] [--name NAME]
  - runs under utils.locks.gpu_lock()
  - writes to utils.paths.new_run_dir(): config.yaml, meta.json (utils.repro.run_metadata),
    metrics.jsonl (every loss term, every step), checkpoints/, tensorboard/
"""

from __future__ import annotations

# `python -m jepa_world.train` imports the jepa_world package (which sets
# PYTORCH_ENABLE_MPS_FALLBACK) before this module, so before torch.
import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import torch
import yaml

from jepa_world.data.synthetic import moving_disc_batch
from jepa_world.models.jepa import JEPA, compute_losses
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import new_run_dir
from jepa_world.utils.repro import run_metadata, seed_everything

LATER_STAGES = {"far": "M4", "goal": "M5", "pixel": "M5"}


def model_horizons(cfg: dict) -> list[int]:
    """The horizons that get heads under train.stages. M3 builds the near stage only."""
    later = [s for s in cfg["train"]["stages"] if s != "near"]
    if later:
        built_in = ", ".join(f"{s} in {LATER_STAGES.get(s, '?')}" for s in later)
        raise NotImplementedError(
            f"train.stages {later} are not built yet ({built_in}); use [near]"
        )
    far = set(cfg["model"]["far_horizons"])
    horizons = [k for k in cfg["model"]["horizons"] if k not in far]
    if not horizons:
        raise ValueError(
            f"model.horizons {cfg['model']['horizons']} has no near horizon "
            f"(far_horizons {sorted(far)}); the near stage needs at least one"
        )
    return horizons


def _require_built_source(cfg: dict) -> None:
    source = cfg["data"]["source"]
    if source != "synthetic":
        raise NotImplementedError(
            f"data.source={source!r}: the episode loader is built in M2 (jepa_world/data). "
            "Until then only 'synthetic' exists"
        )


def check_config(cfg: dict) -> list[int]:
    """Everything a training run needs, checked once, before any run dir exists.
    Returns the horizons that get heads."""
    horizons = model_horizons(cfg)
    _require_built_source(cfg)
    if cfg["train"]["optimizer"] != "adamw":
        raise ValueError(f"train.optimizer={cfg['train']['optimizer']!r}: only 'adamw' exists")
    if cfg["train"]["precision"] != "fp32":
        raise NotImplementedError(
            f"train.precision={cfg['train']['precision']!r}: only fp32 is implemented. A "
            "lower-precision path must be built and tested explicitly first (SPEC §6)"
        )
    return horizons


def build_model(cfg: dict) -> JEPA:
    return JEPA(cfg, model_horizons(cfg))


def build_optimizer(model: JEPA, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    return torch.optim.AdamW(
        model.trainable_parameters(), lr=t["lr"], weight_decay=t["weight_decay"]
    )


def train_step(
    model: JEPA, opt: torch.optim.Optimizer, frames: torch.Tensor, cfg: dict
) -> dict[str, torch.Tensor]:
    """One step: losses -> backward -> optimizer step -> EMA update (in that order).
    Returns every term detached, still on the device (no host sync here)."""
    model.train()
    losses = compute_losses(model, frames, cfg["loss"])
    opt.zero_grad(set_to_none=True)
    losses["loss/total"].backward()
    opt.step()
    model.ema_update(cfg["model"]["ema_momentum"])
    return {k: v.detach() for k, v in losses.items()}


def make_batches(cfg: dict, horizons: list[int]) -> Iterator[torch.Tensor]:
    """Endless CPU batches in the M3 layout (B, C + k_max, 3, S, S), seeded from cfg['seed']."""
    _require_built_source(cfg)
    g = torch.Generator().manual_seed(cfg["seed"])
    n = cfg["model"]["context_frames"] + max(horizons)
    while True:
        yield moving_disc_batch(
            cfg["train"]["batch_size"], n, cfg["sim"]["frame_size"], generator=g
        )


def _flush(rows: list[tuple[int, torch.Tensor]], keys: list[str], f, writer) -> None:
    """One host sync for a block of steps; each step still gets its own line."""
    values = torch.stack([v for _, v in rows]).cpu().tolist()
    for (step, _), vals in zip(rows, values, strict=True):
        rec = {"step": step, **dict(zip(keys, vals, strict=True))}
        f.write(json.dumps(rec) + "\n")
        for k, v in rec.items():
            if k != "step":
                writer.add_scalar(k, v, step)
    f.flush()
    rows.clear()


def save_checkpoint(path: Path, model: JEPA, opt: torch.optim.Optimizer, step: int, cfg: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": cfg},
        path,
    )


def run(
    cfg: dict,
    run_dir: Path,
    batches: Iterator[torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Train for cfg['train']['steps'] steps, logging every term of every step to
    run_dir/metrics.jsonl and TensorBoard. Returns the last step's terms.

    If a step raises, the steps buffered since the last flush are still written before the
    exception propagates, because those are the steps that explain the failure."""
    from torch.utils.tensorboard import SummaryWriter  # slow import, only needed here

    device = device or torch.device("cpu")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    model = build_model(cfg).to(device)
    opt = build_optimizer(model, cfg)
    batches = batches if batches is not None else make_batches(cfg, model.horizons)
    t = cfg["train"]
    steps, log_every, ckpt_every = t["steps"], t["log_every"], t["ckpt_every"]
    keys: list[str] = []
    rows: list[tuple[int, torch.Tensor]] = []
    last: torch.Tensor | None = None
    writer = SummaryWriter(str(run_dir / "tensorboard"))
    try:
        with open(run_dir / "metrics.jsonl", "a") as f:
            try:
                for step in range(1, steps + 1):
                    losses = train_step(model, opt, next(batches).to(device), cfg)
                    keys = keys or list(losses)
                    last = torch.stack([losses[k] for k in keys])
                    rows.append((step, last))
                    if step % log_every == 0 or step == steps:
                        _flush(rows, keys, f, writer)
                    if step % ckpt_every == 0 or step == steps:
                        save_checkpoint(
                            run_dir / "checkpoints" / f"step_{step:07d}.pt", model, opt, step, cfg
                        )
            finally:
                if rows:  # only after an exception: the last step always flushes
                    try:
                        _flush(rows, keys, f, writer)
                    except Exception as e:  # never mask the original error
                        print(f"[train] lost {len(rows)} buffered steps: {e}", file=sys.stderr)
    finally:
        writer.close()
    return dict(zip(keys, last.cpu().tolist(), strict=True)) if last is not None else {}


def start_run(cfg: dict, name: str, device: torch.device, **meta) -> Path:
    """A fresh run dir holding config.yaml and meta.json (code, libraries, machine, device)."""
    run_dir = new_run_dir(name)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {**run_metadata(), "device": str(device), **meta}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--name", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    check_config(cfg)  # before a run dir exists
    device = pick_device(cfg["train"]["device"])
    run_dir = start_run(cfg, args.name or Path(args.config).stem, device)
    print(f"run dir: {run_dir}  device: {device}", flush=True)
    with gpu_lock(f"train {run_dir.name}"):
        final = run(cfg, run_dir, device=device)
    print(json.dumps({"final": final}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
