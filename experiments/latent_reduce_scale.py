"""What would loss.latent_reduce: mean do to a sum run's terms? Read off its metrics.jsonl.

  uv run python experiments/latent_reduce_scale.py --run <run dir> [--run <run dir> ...]

Every latent term a sum run logs (loss/k, var/<L>) is a channel sum, so its mean value is exactly
sum / C(level). For window means over the run, this prints:
  latent sum   near + far + w_var * guard (the active ones), as trained
  latent mean  the same terms, each divided by its level's channel count
  C_eff        their ratio: how much smaller loss/total's latent part is under mean
  near/far     the near terms (k <= 5) over the far terms (k = 10, 20), under each reduction
  goal share   w_goal * loss/goal over each latent total, at the run's own w_goal (if a goal head)
Only the run's own config and metrics are read. The table in
docs/decisions/2026-11-19-latent-loss-normalization.md comes from the two 24k headline runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

WINDOWS = [
    (1501, 2000),
    (2001, 2500),
    (3501, 4000),
    (4001, 4500),
    (6001, 6500),
    (11501, 12000),
    (23501, 24000),
]


def table(run: Path) -> list[str]:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    m, loss, starts = cfg["model"], cfg["loss"], cfg["train"]["stage_start"]
    ch = {lv: spec["channels"] for lv, spec in m["levels"].items()}
    near = [k for k in m["horizons"] if k not in m["far_horizons"]]
    far = list(m["far_horizons"])
    level = {int(k): v for k, v in m["head_level"].items()}
    has_goal = "goal" in cfg["train"]["stages"]
    rows = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
    out = [
        run.name,
        f"{'steps':>13} {'lat sum':>9} {'lat mean':>9} {'C_eff':>6} {'near/far sum':>12} "
        f"{'near/far mean':>13} {'goal/lat sum':>12} {'goal/lat mean':>13}",
    ]
    for a, b in WINDOWS:
        w = [r for r in rows if a <= r["step"] <= b]
        if not w:
            continue
        far_on, goal_on = a > starts.get("far", 0), has_goal and a > starts.get("goal", 0)
        ks = near + (far if far_on else [])
        lvls = sorted({level[k] for k in ks})
        s_tot, m_tot, nf_s, nf_m, g_s, g_m = [], [], [], [], [], []
        for r in w:
            near_s = sum(r[f"loss/k{k}"] for k in near)
            near_m = sum(r[f"loss/k{k}"] / ch[level[k]] for k in near)
            far_s = sum(r[f"loss/k{k}"] for k in far) if far_on else 0.0
            far_m = sum(r[f"loss/k{k}"] / ch[level[k]] for k in far) if far_on else 0.0
            guard = loss["collapse_guard"]
            g_sum = loss["w_var"] * sum(r[f"var/{lv}"] for lv in lvls) if guard else 0.0
            g_mean = loss["w_var"] * sum(r[f"var/{lv}"] / ch[lv] for lv in lvls) if guard else 0.0
            s, mm = near_s + far_s + g_sum, near_m + far_m + g_mean
            s_tot.append(s)
            m_tot.append(mm)
            if far_on:
                nf_s.append(near_s / far_s)
                nf_m.append(near_m / far_m)
            if goal_on:
                # as trained: the logged weight if the run has it (w_goal_eff, 2026-11-19 on;
                # / C(L3) under mean), else the configured one (every earlier run: sum)
                goal = r.get("w_goal_eff", loss["w_goal"]) * r["loss/goal"]
                g_s.append(goal / s)
                g_m.append(goal / mm)

        def f(x):
            return f"{np.mean(x):.3g}" if x else "-"

        c_eff = np.mean(s_tot) / np.mean(m_tot)
        out.append(
            f"{a:>6}-{b:<6} {f(s_tot):>9} {f(m_tot):>9} {c_eff:6.1f} {f(nf_s):>12} "
            f"{f(nf_m):>13} {f(g_s):>12} {f(g_m):>13}"
        )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, action="append", required=True)
    for run in p.parse_args(argv).run:
        print("\n".join(table(run)))


if __name__ == "__main__":
    main()
