"""L1 near-head benchmark (SPEC §9.4, §11: "Benchmark the L1 heads on the Mac early").

  uv run python -m jepa_world.bench --config configs/m3_near_synthetic.yaml \
      [--hidden 16 32 64] [--kernel 3 1] [--iters 20] [--warmup 5] [--set key=value ...]

For each (hidden width, kernel) variant: ms per forward+backward of the five near heads alone,
ms per full train_step (encoder + target + heads + AdamW + EMA), parameter counts, and MPS driver
memory. Ops that fell back to CPU are recorded across the whole run. Runs under the GPU lock and
writes results.json to a new run dir. Frames are always synthetic: timing needs shapes, not data.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from jepa_world.data.synthetic import moving_disc_batch
from jepa_world.models.jepa import context_stack
from jepa_world.train import build_model, build_optimizer, model_horizons, start_run, train_step
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.repro import seed_everything


@contextmanager
def record_mps_fallbacks() -> Iterator[list[str]]:
    """Collect the 'will fall back to run on the CPU' warnings raised inside the block (the list
    is filled when the block exits). torch emits each one ONCE PER PROCESS, so this only sees an
    op's first use. Wrap everything from the first MPS op on, or use a fresh process."""
    got: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield got
    got.extend(str(w.message) for w in caught if "fall back" in str(w.message))


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _time_ms(fn: Callable[[], object], device: torch.device, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return 1000 * (time.perf_counter() - t0) / iters


def _bench_variant(cfg: dict, device: torch.device, iters: int, warmup: int) -> dict:
    """All of a variant's tensors are locals, so they are freed when this returns."""
    seed_everything(cfg["seed"])
    model = build_model(cfg).to(device)
    opt = build_optimizer(model, cfg)
    n = cfg["model"]["context_frames"] + max(model.horizons)
    frames = moving_disc_batch(
        cfg["train"]["batch_size"],
        n,
        cfg["sim"]["frame_size"],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    ).to(device)
    with torch.no_grad():
        feat = model.context_encoder(context_stack(frames, model.context_frames), upto="L1")["L1"]

    # The variant under test is the L1 (near) heads; far heads only enter the full step.
    l1_heads = [model.heads[str(k)] for k in model.horizons if model.head_level[k] == "L1"]
    far_heads = [model.heads[str(k)] for k in model.far_horizons]

    def heads_fwd_bwd():
        model.heads.zero_grad(set_to_none=True)
        sum(h(feat).square().mean() for h in l1_heads).backward()

    def full_step():
        train_step(model, opt, frames, cfg)  # step=None: every built stage active

    ms_heads = _time_ms(heads_fwd_bwd, device, iters, warmup)
    ms_step = _time_ms(full_step, device, iters, warmup)
    return {
        "hidden": cfg["model"]["l1_head_hidden"],
        "kernel": cfg["model"]["l1_head_kernel"],
        "batch_size": cfg["train"]["batch_size"],
        "device": str(device),
        "ms_heads_fwd_bwd": ms_heads,
        "ms_train_step": ms_step,
        "head_params": sum(p.numel() for h in l1_heads for p in h.parameters()),
        "far_head_params": sum(p.numel() for h in far_heads for p in h.parameters()),
        "stages": list(cfg["train"]["stages"]),
        "total_trainable_params": sum(p.numel() for p in model.trainable_parameters()),
        # The cache is emptied between variants, and the caching allocator keeps this variant's
        # freed blocks, so driver memory here approximates this variant's peak.
        "mps_driver_mb": (
            torch.mps.driver_allocated_memory() / 2**20 if device.type == "mps" else None
        ),
    }


def bench_heads(
    cfg: dict,
    device: torch.device,
    variants: list[tuple[int, int]],
    iters: int = 20,
    warmup: int = 5,
) -> dict:
    """{"rows": one dict per (hidden, kernel) variant, "fallbacks": ops that fell back to CPU}."""
    model_horizons(cfg)  # fail early on configs M3 can't build
    rows = []
    with record_mps_fallbacks() as fallbacks:
        for hidden, kernel in variants:
            c = copy.deepcopy(cfg)
            c["model"]["l1_head_hidden"], c["model"]["l1_head_kernel"] = hidden, kernel
            if device.type == "mps":
                torch.mps.empty_cache()
            rows.append(_bench_variant(c, device, iters, warmup))
    return {"rows": rows, "fallbacks": sorted(set(fallbacks))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--hidden", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--kernel", type=int, nargs="+", default=[3, 1])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    model_horizons(cfg)  # before a run dir exists
    device = pick_device(cfg["train"]["device"])
    variants = [(h, k) for k in args.kernel for h in args.hidden]
    run_dir = start_run(cfg, "bench-l1-heads", device, iters=args.iters, warmup=args.warmup)
    with gpu_lock("bench-l1-heads"):  # always: a timing taken while another job runs is wrong
        result = bench_heads(cfg, device, variants, iters=args.iters, warmup=args.warmup)
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"run dir: {run_dir}")
    print(
        f"{'hidden':>6} {'kernel':>6} {'heads ms':>9} {'step ms':>8} {'head params':>11} {'MB':>7}"
    )
    for r in result["rows"]:
        mb = f"{r['mps_driver_mb']:.0f}" if r["mps_driver_mb"] is not None else "-"
        print(
            f"{r['hidden']:>6} {r['kernel']:>6} {r['ms_heads_fwd_bwd']:>9.1f} "
            f"{r['ms_train_step']:>8.1f} {r['head_params']:>11} {mb:>7}"
        )
    print(f"CPU fallbacks: {result['fallbacks'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
