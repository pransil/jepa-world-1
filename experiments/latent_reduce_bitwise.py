"""Is loss.latent_reduce: sum (or no key) bitwise the loss code of an earlier commit, in training?

  uv run python experiments/latent_reduce_bitwise.py [--ref origin/main] [--steps 6]
      [--reduce sum|mean] [--match <substring of a setup name>]

Loads jepa_world/models/jepa.py and models/world.py as they were at --ref (git show) beside the
current ones, and trains the same tiny model on CPU through each for --steps steps (AdamW + EMA, as
train.train_step does). Compares every logged term at every step and every parameter at the end,
bitwise:
  - jepa.losses_from_output: near only, far joining mid-run, the mean x1 guard (exp_guard_mean),
    top-down, and a config without the latent_reduce key;
  - world.compute_world_losses: near + far with no aux head (what the collapse check trained), and
    with the goal head + pixel decoder the fixed w_pix and the 10% share, all four stages with the
    goal and pixel joining mid-run, and a config without the key.
--reduce sets loss.latent_reduce in every setup (default sum, the claim that the default path didn't
move; the no-key setups are skipped under mean). --match keeps the setups whose name contains it,
e.g. `--ref 00a4e78 --reduce mean --match "near+far"` checks the collapse-check runs' code path
against a later commit. A log key the current code adds (w_goal_eff, 2026-11-19) is reported, not
compared; every key of --ref's must match. The old world.py runs against the current jepa.py (its
imports resolve to the package), so the world setups check world.py alone; the jepa setups check
jepa.py. Prints one line per setup and BITWISE IDENTICAL or DIFFERENT.
docs/decisions/2026-11-19-latent-loss-normalization.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import torch  # noqa: E402
from tiny import random_batch, random_frames, tiny_aux_cfg, tiny_cfg, tiny_far_cfg  # noqa: E402

import jepa_world  # noqa: E402,F401  (MPS fallback env before torch)
from jepa_world import train  # noqa: E402
from jepa_world.models import jepa as current_jepa  # noqa: E402
from jepa_world.models import world as current_world  # noqa: E402


def module_at(ref: str, rel: str):
    """`rel` as it was at `ref`. Its own imports resolve to the current package."""
    src = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    name = Path(rel).stem + "_at_ref"
    path = Path(tempfile.mkdtemp()) / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NEW_KEYS = {"w_goal_eff"}  # logged since 2026-11-19; absent at earlier refs


def _with(reduce: str, cfgs: dict[str, dict], keyless: str) -> dict[str, dict]:
    for cfg in cfgs.values():
        cfg["loss"]["latent_reduce"] = reduce
    if reduce == "sum":
        del cfgs[keyless]["loss"]["latent_reduce"]  # a config from before the key: means sum
    else:
        del cfgs[keyless]
    return cfgs


def jepa_setups(reduce: str) -> dict[str, dict]:
    return _with(
        reduce,
        {
            "near only": tiny_cfg(),
            "near+far, far joins at step 3": tiny_far_cfg("train.stage_start={near: 0, far: 3}"),
            "near+far, guard mean x1": tiny_far_cfg("loss.var_reduce=mean", "loss.w_var=1.0"),
            "near+far+top-down": tiny_far_cfg("model.top_down=true"),
            "near+far, no latent_reduce key": tiny_far_cfg(),
        },
        "near+far, no latent_reduce key",
    )


def world_setups(reduce: str) -> dict[str, dict]:
    four = ["train.stages=[near, far, goal, pixel]"]
    return _with(
        reduce,
        {
            "world, near+far (no aux head)": tiny_far_cfg("train.stage_start={near: 0, far: 3}"),
            "world, fixed w_pix": tiny_aux_cfg("loss.pix_share=null"),
            "world, 10% share": tiny_aux_cfg(),
            "world, all four stages, goal/pixel join at 2/4": tiny_aux_cfg(
                *four, "train.stage_start={near: 0, far: 0, goal: 2, pixel: 4}"
            ),
            "world, fixed w_pix, no latent_reduce key": tiny_aux_cfg("loss.pix_share=null"),
        },
        "world, fixed w_pix, no latent_reduce key",
    )


def jepa_step(losses_from_output):
    def step_fn(model, cfg, step):
        act = train.active_horizons(cfg, model.horizons, step)
        out = model(random_frames(cfg, seed=step), active=act)
        return losses_from_output(model, out, cfg["loss"], act)

    return step_fn


def world_step(compute_world_losses):
    def step_fn(model, cfg, step):
        act = train.active_horizons(cfg, model.horizons, step)
        aux = [s for s in train.active_aux(cfg, step) if s in model.aux_stages]
        batch = random_batch(cfg, seed=step)
        return compute_world_losses(model, batch, cfg["loss"], active=act, aux_active=aux)

    return step_fn


def trace(step_fn, cfg: dict, steps: int):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    opt = train.build_optimizer(model, cfg)
    logs = []
    for step in range(steps):
        losses = step_fn(model, cfg, step)
        opt.zero_grad(set_to_none=True)
        losses["loss/total"].backward()
        opt.step()
        model.ema_update(cfg["model"]["ema_momentum"])
        logs.append({k: v.detach().clone() for k, v in losses.items()})
    return logs, [p.detach().clone() for p in model.parameters()]


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a, b) or (math.isnan(a.item()) and math.isnan(b.item()))


def compare(name: str, old_fn, new_fn, cfg: dict, steps: int) -> bool:
    l_old, p_old = trace(old_fn, cfg, steps)
    l_new, p_new = trace(new_fn, cfg, steps)
    added = set(l_new[0]) - set(l_old[0])
    pairs = list(zip(l_old, l_new, strict=True))
    keys_ok = all(set(a) <= set(b) and set(b) - set(a) <= NEW_KEYS for a, b in pairs)
    logs = keys_ok and all(same(a[k], b[k]) for a, b in zip(l_old, l_new, strict=True) for k in a)
    params = all(torch.equal(a, b) for a, b in zip(p_old, p_new, strict=True))
    extra = f" (+ new key {', '.join(sorted(added))}, not compared)" if added else ""
    print(
        f"{name}: {steps} steps, all {len(l_old[0])} logged terms bitwise equal={logs}{extra}, "
        f"all {len(p_new)} params bitwise equal={params}"
    )
    return logs and params


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref", default="origin/main")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--reduce", choices=["sum", "mean"], default="sum")
    p.add_argument("--match", default="")
    args = p.parse_args(argv)
    old_jepa = module_at(args.ref, "jepa_world/models/jepa.py")
    old_world = module_at(args.ref, "jepa_world/models/world.py")
    ok = True
    print(f"--ref {args.ref}, loss.latent_reduce={args.reduce}")
    for name, cfg in jepa_setups(args.reduce).items():
        if args.match in name:
            old, new = old_jepa.losses_from_output, current_jepa.losses_from_output
            ok &= compare(name, jepa_step(old), jepa_step(new), cfg, args.steps)
    for name, cfg in world_setups(args.reduce).items():
        if args.match in name:
            old, new = old_world.compute_world_losses, current_world.compute_world_losses
            ok &= compare(name, world_step(old), world_step(new), cfg, args.steps)
    print("BITWISE IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
