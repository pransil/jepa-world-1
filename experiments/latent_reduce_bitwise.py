"""Is loss.latent_reduce: sum (or no key) bitwise the loss code of an earlier commit, in training?

  uv run python experiments/latent_reduce_bitwise.py [--ref origin/main] [--steps 6]

Loads jepa_world/models/jepa.py as it was at --ref (git show) beside the current one, and trains the
same tiny model on CPU through each module's losses_from_output for --steps steps (AdamW + EMA, as
train.train_step does). Compares every logged term at every step and every parameter at the end,
bitwise, in five setups: near only, far joining mid-run, the mean x1 guard (exp_guard_mean), top-down,
and a config without the latent_reduce key. Prints one line per setup and BITWISE IDENTICAL or
DIFFERENT. docs/decisions/2026-11-19-latent-loss-normalization.md.
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
import jepa_world  # noqa: E402,F401  (MPS fallback env before torch)
import torch  # noqa: E402
from tiny import random_frames, tiny_cfg, tiny_far_cfg  # noqa: E402

from jepa_world import train  # noqa: E402
from jepa_world.models import jepa as current  # noqa: E402


def module_at(ref: str):
    src = subprocess.run(
        ["git", "show", f"{ref}:jepa_world/models/jepa.py"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    path = Path(tempfile.mkdtemp()) / "jepa_at_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("jepa_at_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setups() -> dict[str, dict]:
    out = {
        "near only": tiny_cfg(),
        "near+far, far joins at step 3": tiny_far_cfg("train.stage_start={near: 0, far: 3}"),
        "near+far, guard mean x1": tiny_far_cfg("loss.var_reduce=mean", "loss.w_var=1.0"),
        "near+far+top-down": tiny_far_cfg("model.top_down=true"),
        "near+far, no latent_reduce key": tiny_far_cfg(),
    }
    for cfg in out.values():
        cfg["loss"]["latent_reduce"] = "sum"
    del out["near+far, no latent_reduce key"]["loss"]["latent_reduce"]
    return out


def trace(losses_from_output, cfg: dict, steps: int):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    opt = train.build_optimizer(model, cfg)
    logs = []
    for step in range(steps):
        act = train.active_horizons(cfg, model.horizons, step)
        losses = losses_from_output(
            model, model(random_frames(cfg, seed=step), active=act), cfg["loss"], act
        )
        opt.zero_grad(set_to_none=True)
        losses["loss/total"].backward()
        opt.step()
        model.ema_update(cfg["model"]["ema_momentum"])
        logs.append({k: v.detach().clone() for k, v in losses.items()})
    return logs, [p.detach().clone() for p in model.parameters()]


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a, b) or (math.isnan(a.item()) and math.isnan(b.item()))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref", default="origin/main")
    p.add_argument("--steps", type=int, default=6)
    args = p.parse_args(argv)
    old = module_at(args.ref)
    ok = True
    for name, cfg in setups().items():
        l_old, p_old = trace(old.losses_from_output, cfg, args.steps)
        l_new, p_new = trace(current.losses_from_output, cfg, args.steps)
        logs = all(a.keys() == b.keys() for a, b in zip(l_old, l_new, strict=True)) and all(
            same(a[k], b[k]) for a, b in zip(l_old, l_new, strict=True) for k in a
        )
        params = all(torch.equal(a, b) for a, b in zip(p_old, p_new, strict=True))
        ok &= logs and params
        print(
            f"{name}: {args.steps} steps, all {len(l_new[0])} logged terms bitwise equal={logs}, "
            f"all {len(p_new)} params bitwise equal={params}"
        )
    print("BITWISE IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
