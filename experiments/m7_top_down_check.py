"""M7 §4 guard and probe tables: top-down arm against the 48k reference.

  uv run python experiments/m7_top_down_check.py --arm <s0 dir> --arm <s1 dir> \
      --ref <ref s0 dir> --ref <ref s1 dir>

The rules are in experiments/2026-11-20-m7-top-down.md (committed before the runs).
- **Guard, latent heads not worse:** each run's `latent_dev.json` from
  `experiments/latent_relerr.py --run <dir> --split dev` (200 batches, CPU, step 48,000). `relerr` at
  k1–k5, k10 and k20; under the both-seeds rule, none may be worse in the arm.
- **Top-down terms** (arm only), on dev: `relerr/td/k`, its copy bar, and `logvar/td/k`.
- **Probes**, TEST, from each run's step-48,000 test+dev eval (`by_level`): `pos_cam` at L2 and L3, and
  the hidden parameters at L3.
Verdicts: "better" / "worse" only if both arm seeds are beyond both reference seeds. Reads JSON only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m7_ablation_check import eval_report, pair, probe_rmse, test_metrics, verdict  # noqa: E402

STEP = 48000
KS = (1, 2, 3, 4, 5, 10, 20)


def latent(run: Path) -> dict:
    p = run / "latent_dev.json"
    d = json.load(open(p))
    if d["step"] != STEP or d["split"] != "dev" or d["batches"] != 200:
        raise SystemExit(f"{p}: expected step {STEP}, split dev, 200 batches")
    g = d["meta"]["git"]
    print(
        f"  {run.name}: latent_dev.json (sha {g['sha'][:7]}, dirty {g['dirty']}, "
        f"{d['device']}, {d['latent_reduce']})"
    )
    return d["terms"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", type=Path, action="append", required=True)
    ap.add_argument("--ref", type=Path, action="append", required=True)
    a = ap.parse_args()
    if len(a.arm) != 2 or len(a.ref) != 2:
        raise SystemExit("give two --arm and two --ref, seed 0 first")
    print("latent scores:")
    la, lr = [latent(r) for r in a.arm], [latent(r) for r in a.ref]
    print("evals:")
    ea, er = [eval_report(r, STEP) for r in a.arm], [eval_report(r, STEP) for r in a.ref]

    print("\nlatent heads, DEV, relerr (mean over 200 batches; s0 / s1)")
    print("| k | reference | top-down | verdict | copy bar, reference | copy bar, top-down |")
    print("|---:|---|---|---|---|---|")
    worse = []
    for k in KS:
        key = f"relerr/k{k}"
        xa, xr = [t[key]["mean"] for t in la], [t[key]["mean"] for t in lr]
        v = verdict(xa, xr)
        if v == "worse":
            worse.append(k)
        ca = [t[f"relerr/copy/k{k}"]["mean"] for t in la]
        cr = [t[f"relerr/copy/k{k}"]["mean"] for t in lr]
        print(
            f"| {k} | {pair(xr, '.4f')} | {pair(xa, '.4f')} | {v} | {pair(cr, '.4f')} | "
            f"{pair(ca, '.4f')} |"
        )
    print(
        f"GUARD latent heads not worse on dev: {'FAIL, worse at k=' + str(worse) if worse else 'PASS'}"
    )

    print("\nother dev terms (s0 / s1)")
    print("| key | reference | top-down |")
    print("|---|---|---|")
    for key in ("emb_std/L1", "emb_std/L2", "emb_std/L3", "logvar/k10", "logvar/k20"):
        print(
            f"| {key} | {pair([t[key]['mean'] for t in lr], '.3f')} | "
            f"{pair([t[key]['mean'] for t in la], '.3f')} |"
        )

    print("\ntop-down terms, DEV, top-down runs (s0 / s1)")
    print("| k | relerr/td | copy bar | logvar/td |")
    print("|---:|---|---|---|")
    for k in (10, 20):
        print(
            f"| {k} | {pair([t[f'relerr/td/k{k}']['mean'] for t in la], '.4f')} | "
            f"{pair([t[f'relerr/copy/td/k{k}']['mean'] for t in la], '.4f')} | "
            f"{pair([t[f'logvar/td/k{k}']['mean'] for t in la], '.3f')} |"
        )

    print("\nprobes by level, TEST: R^2, and Euclidean RMSE (s0 / s1)")
    print(
        "| target | level | R^2 reference | R^2 top-down | RMSE reference | RMSE top-down | RMSE verdict |"
    )
    print("|---|---|---|---|---|---|---|")
    for tgt, levels in (
        ("pos_cam", ("L2", "L3")),
        ("radius", ("L3",)),
        ("friction", ("L3",)),
        ("restitution", ("L3",)),
    ):
        for lv in levels:
            r2r = [test_metrics(r)["probes"]["by_level"][lv][tgt]["r2"] for r in er]
            r2a = [test_metrics(r)["probes"]["by_level"][lv][tgt]["r2"] for r in ea]
            rr, ra = [probe_rmse(r, lv, tgt) for r in er], [probe_rmse(r, lv, tgt) for r in ea]
            print(
                f"| {tgt} | {lv} | {pair(r2r)} | {pair(r2a)} | {pair(rr, '.4f')} | "
                f"{pair(ra, '.4f')} | {verdict(ra, rr)} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
