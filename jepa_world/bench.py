"""L1 near-head benchmark (SPEC §9.4, §11: "Benchmark the L1 heads on the Mac early").

  uv run python -m jepa_world.bench --config configs/m3_near_synthetic.yaml \
      [--hidden 16 32 64] [--kernel 3 1] [--iters 20] [--warmup 5] [--set key=value ...]

For each (hidden width, kernel) variant: ms per forward+backward of the five near heads alone,
ms per full train_step (encoder + target + heads + AdamW + EMA), parameter count, MPS driver memory,
and any op that fell back to CPU. Runs under the GPU lock. Writes results.json to a new run dir.
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
import yaml

from jepa_world.data.synthetic import moving_disc_batch
from jepa_world.models.jepa import context_stack
from jepa_world.train import build_model, build_optimizer, check_config, train_step
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import new_run_dir
from jepa_world.utils.repro import run_metadata, seed_everything


@contextmanager
def record_mps_fallbacks() -> Iterator[list[str]]:
    """Collect the 'will fall back to run on the CPU' warnings raised inside the block.
    torch warns once per op per process, so run this around the first use of each op."""
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


def _mps_mb() -> float | None:
    if not torch.backends.mps.is_available():
        return None
    return torch.mps.driver_allocated_memory() / 2**20


def bench_heads(
    cfg: dict,
    device: torch.device,
    variants: list[tuple[int, int]],
    iters: int = 20,
    warmup: int = 5,
) -> list[dict]:
    rows = []
    for hidden, kernel in variants:
        c = copy.deepcopy(cfg)
        c["model"]["l1_head_hidden"], c["model"]["l1_head_kernel"] = hidden, kernel
        horizons = check_config(c)
        seed_everything(c["seed"])
        model = build_model(c).to(device)
        opt = build_optimizer(model, c)
        n = c["model"]["context_frames"] + max(horizons)
        frames = moving_disc_batch(
            c["train"]["batch_size"],
            n,
            c["sim"]["frame_size"],
            generator=torch.Generator().manual_seed(c["seed"]),
        ).to(device)
        with torch.no_grad():
            feat = model.context_encoder(context_stack(frames, model.context_frames), upto="L1")[
                "L1"
            ]

        def heads_fwd_bwd(model=model, feat=feat):
            model.heads.zero_grad(set_to_none=True)
            sum(h(feat).square().mean() for h in model.heads.values()).backward()

        with record_mps_fallbacks() as fallbacks:
            ms_heads = _time_ms(heads_fwd_bwd, device, iters, warmup)
            ms_step = _time_ms(
                lambda m=model, o=opt, x=frames, c=c: train_step(m, o, x, c), device, iters, warmup
            )
        rows.append(
            {
                "hidden": hidden,
                "kernel": kernel,
                "batch_size": c["train"]["batch_size"],
                "device": str(device),
                "ms_heads_fwd_bwd": ms_heads,
                "ms_train_step": ms_step,
                "head_params": sum(p.numel() for p in model.heads.parameters()),
                "total_trainable_params": sum(p.numel() for p in model.trainable_parameters()),
                "mps_driver_mb": _mps_mb() if device.type == "mps" else None,
                "fallbacks": sorted(set(fallbacks)),
            }
        )
        del model, opt, frames, feat
        if device.type == "mps":
            torch.mps.empty_cache()
    return rows


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
    check_config(cfg)
    device = pick_device(cfg["train"]["device"])
    variants = [(h, k) for k in args.kernel for h in args.hidden]
    run_dir = new_run_dir("bench-l1-heads")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {**run_metadata(), "device": str(device), "iters": args.iters, "warmup": args.warmup}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    with gpu_lock("bench-l1-heads", enabled=device.type == "mps"):
        rows = bench_heads(cfg, device, variants, iters=args.iters, warmup=args.warmup)
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"run dir: {run_dir}")
    print(
        f"{'hidden':>6} {'kernel':>6} {'heads ms':>9} {'step ms':>8} {'head params':>11} {'MB':>7}"
    )
    for r in rows:
        mb = f"{r['mps_driver_mb']:.0f}" if r["mps_driver_mb"] is not None else "-"
        print(
            f"{r['hidden']:>6} {r['kernel']:>6} {r['ms_heads_fwd_bwd']:>9.1f} "
            f"{r['ms_train_step']:>8.1f} {r['head_params']:>11} {mb:>7}"
            + (f"  FALLBACKS: {r['fallbacks']}" if r["fallbacks"] else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
