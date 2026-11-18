"""Training entry point. Built in M3 (near heads), extended in M4 (far) and M5 (goal, pixel).
See SPEC §5-§6.

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
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
import yaml

from jepa_world.data.dataset import batches as episode_batches
from jepa_world.data.dataset import load_split, require_split
from jepa_world.data.splits import TRAINABLE, LeakageError
from jepa_world.data.synthetic import moving_disc_batch
from jepa_world.models.world import AUX_STAGES, WorldModel, compute_world_losses
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import new_run_dir
from jepa_world.utils.repro import run_metadata, seed_everything

HORIZON_STAGES = ("near", "far")  # stages that own predictor horizons
STAGES = (*HORIZON_STAGES, *AUX_STAGES)  # SPEC §5.5 order: near, far, goal, pixel

Batch = torch.Tensor | dict[str, torch.Tensor]  # frames (M3 layout), or a with_gt dict


def _stage_of(cfg: dict, k: int) -> str:
    return "far" if k in cfg["model"]["far_horizons"] else "near"


def model_horizons(cfg: dict) -> list[int]:
    """The horizons that get heads under train.stages: near ones for 'near', far ones for 'far'.
    'goal' and 'pixel' own no horizon: they add the aux heads (models/world.py)."""
    stages = cfg["train"]["stages"]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"train.stages {unknown}: unknown stage (have {list(STAGES)})")
    horizons = [k for k in cfg["model"]["horizons"] if _stage_of(cfg, k) in stages]
    if not horizons:
        raise ValueError(
            f"model.horizons {cfg['model']['horizons']} has no {' or '.join(stages)} horizon "
            f"for train.stages {stages} (far_horizons {cfg['model']['far_horizons']})"
        )
    return horizons


def _check_stage_start(cfg: dict, horizons: list[int]) -> None:
    """stage_start.<s> = the number of steps completed before stage s joins the loss.

    Every stage in train.stages needs an int >= 0 (a YAML bool is not a step). Among the stages
    that own a built horizon, one must start at 0 (or step 1 would have nothing to train). Every
    stage that builds something (a horizon stage with a horizon, or an aux stage) must start
    before train.steps, or it would be built and logged but never trained."""
    t = cfg["train"]
    stages, start, steps = t["stages"], t["stage_start"], t["steps"]
    for s in stages:
        v = start.get(s)
        if type(v) is not int or v < 0:
            raise ValueError(f"train.stage_start.{s} must be an int >= 0, got {v!r}")
    owned = sorted({_stage_of(cfg, k) for k in horizons}, key=stages.index)
    if min(start[s] for s in owned) != 0:
        raise ValueError(
            f"train.stage_start {start}: one of the stages that own a horizon ({owned}) must "
            "start at 0"
        )
    builds = owned + [s for s in stages if s in AUX_STAGES]
    late = [s for s in builds if start[s] >= steps]
    if late:
        raise ValueError(
            f"train.stage_start {start}: {late} would never train in train.steps={steps}"
        )


def active_horizons(cfg: dict, horizons: list[int], step: int | None) -> list[int]:
    """The horizons whose stage has started once `step` steps are complete (SPEC §5.5,
    docs/decisions/2026-11-17-stage-schedule-within-run.md). `step` counts completed steps, so
    run() passes metrics step - 1: far: 2000 -> metrics steps 1..2000 near-only, far from 2001.
    step=None: all of them."""
    if step is None:
        return list(horizons)
    start = cfg["train"]["stage_start"]
    return [k for k in horizons if step >= start[_stage_of(cfg, k)]]


def active_aux(cfg: dict, step: int | None) -> list[str]:
    """The aux stages in train.stages that have started once `step` steps are complete (same
    convention as active_horizons). step=None: all of them."""
    stages = [s for s in AUX_STAGES if s in cfg["train"]["stages"]]
    if step is None:
        return stages
    start = cfg["train"]["stage_start"]
    return [s for s in stages if step >= start[s]]


def batch_k_max(cfg: dict, horizons: list[int]) -> int:
    """How far past t a batch must reach: the largest model horizon, or goal horizon if the goal
    stage is on (its labels are ball_pos[t+k]). The pixel target, t+1, is always inside."""
    ks = list(horizons)
    if "goal" in cfg["train"]["stages"]:
        ks += list(cfg["model"]["goal_horizons"])
    return max(ks)


def _with_gt(cfg: dict) -> bool:
    return "goal" in cfg["train"]["stages"]  # only the goal loss needs a label beyond the frames


def _require_built_source(cfg: dict) -> None:
    """'episodes' needs data.split to be dev or train (never test) and built for this config
    (FileNotFoundError names the command)."""
    source = cfg["data"]["source"]
    if source == "episodes":
        split = cfg["data"]["split"]
        if split not in TRAINABLE:
            raise LeakageError(
                f"data.split={split!r}: training reads only {list(TRAINABLE)}. The test "
                "terrains are held out for evaluation (SPEC §3.5, §7.3)"
            )
        require_split(cfg, split)
    elif source != "synthetic":
        raise ValueError(f"data.source={source!r}: expected 'episodes' or 'synthetic'")


def _episode_batches(cfg: dict, horizons: list[int]) -> Iterator[Batch]:
    """The M2 loader on data.split, in the same layout as the synthetic source. Loads the split
    now, not on the first next(), so main() can do it before taking the GPU lock."""
    es = load_split(cfg, cfg["data"]["split"])
    c, k_max = cfg["model"]["context_frames"], batch_k_max(cfg, horizons)
    with_gt = _with_gt(cfg)
    batches = episode_batches(es, cfg["train"]["batch_size"], c, k_max, cfg["seed"], with_gt)
    # with_gt carries every GT_KEYS field; the goal loss reads only ball_pos.
    return (
        ({"frames": b["frames"], "ball_pos": b["ball_pos"]} for b in batches)
        if with_gt
        else batches
    )


def _synthetic_batches(cfg: dict, horizons: list[int]) -> Iterator[Batch]:
    g = torch.Generator().manual_seed(cfg["seed"])
    n = cfg["model"]["context_frames"] + batch_k_max(cfg, horizons)
    while True:
        yield moving_disc_batch(
            cfg["train"]["batch_size"],
            n,
            cfg["sim"]["frame_size"],
            generator=g,
            with_gt=_with_gt(cfg),
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
    _check_stage_start(cfg, horizons)
    return horizons


def build_model(cfg: dict) -> WorldModel:
    """The JEPA core, plus the goal head / pixel decoder when train.stages has goal / pixel."""
    stages = cfg["train"]["stages"]
    return WorldModel(cfg, model_horizons(cfg), goal="goal" in stages, pixel="pixel" in stages)


def build_optimizer(model: WorldModel, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]
    return torch.optim.AdamW(
        model.trainable_parameters(), lr=t["lr"], weight_decay=t["weight_decay"]
    )


def train_step(
    model: WorldModel,
    opt: torch.optim.Optimizer,
    batch: Batch,
    cfg: dict,
    step: int | None = None,
) -> dict[str, torch.Tensor]:
    """One step: losses -> backward -> optimizer step -> EMA update (in that order).
    `batch` is a frames tensor, or a with_gt dict ({"frames", "ball_pos", ...}; the goal stage
    needs it). `step` (steps completed before this one) selects the active stages via
    train.stage_start; None means every built stage.
    Returns every term detached, still on the device (no host sync here)."""
    model.train()
    active = active_horizons(cfg, model.horizons, step)
    aux = [s for s in active_aux(cfg, step) if s in model.aux_stages]
    losses = compute_world_losses(model, batch, cfg["loss"], active=active, aux_active=aux)
    opt.zero_grad(set_to_none=True)
    losses["loss/total"].backward()
    opt.step()
    model.ema_update(cfg["model"]["ema_momentum"])
    return {k: v.detach() for k, v in losses.items()}


def make_batches(cfg: dict, horizons: list[int]) -> Iterator[Batch]:
    """Endless CPU batches, seeded from cfg['seed']: frames (B, C + k_max, 3, S, S) in the M3
    layout, or, when train.stages has goal, {"frames", "ball_pos" (B, C + k_max, 2)}."""
    _require_built_source(cfg)
    if cfg["data"]["source"] == "episodes":
        return _episode_batches(cfg, horizons)
    return _synthetic_batches(cfg, horizons)


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


def to_device(batch: Batch, device: torch.device) -> Batch:
    if isinstance(batch, Mapping):
        return {k: v.to(device) for k, v in batch.items()}
    return batch.to(device)


def save_checkpoint(
    path: Path, model: WorldModel, opt: torch.optim.Optimizer, step: int, cfg: dict
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": cfg},
        path,
    )


def run(
    cfg: dict,
    run_dir: Path,
    batches: Iterator[Batch] | None = None,
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
                    batch = to_device(next(batches), device)
                    losses = train_step(model, opt, batch, cfg, step=step - 1)
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
    horizons = check_config(cfg)  # before a run dir exists
    batches = make_batches(cfg, horizons)  # loads an episode split here, not under the GPU lock
    device = pick_device(cfg["train"]["device"])
    run_dir = start_run(cfg, args.name or Path(args.config).stem, device)
    print(f"run dir: {run_dir}  device: {device}", flush=True)
    with gpu_lock(f"train {run_dir.name}"):
        final = run(cfg, run_dir, batches=batches, device=device)
    print(json.dumps({"final": final}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
