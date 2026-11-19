"""Score the latent_reduce: mean collapse check against its pre-registered pass rule.

  uv run python experiments/latent_mean_check.py --mean <s0 dir> --mean <s1 dir> \
      --sum <s0 dir> --sum <s1 dir>

The rule is in experiments/2026-11-19-latent-mean-collapse-check.md (committed before the runs):
  1. L3 emb_std never below 50% of its step-2000 value over steps 2001-8000 (train batches, per step);
  2. logvar/k10 and logvar/k20: mean over steps 7501-8000 >= -3.0, and no step below -8;
  3. dev relerr/k5 at step 8000 (latent_dev.json, from experiments/latent_relerr.py) at or below
     the sum arm's worse seed (0.30203), in each seed.
Seeds pair up in the order given (--mean s0 with --sum s0). Also prints the secondary numbers: the
same per-level std and log-variance for the sum runs, dev relerr at every k, train-batch k5 relerr
before and after far joins, and how closely steps 1-2000 (near only: mean = sum / 64 under Adam)
track the sum run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FAR_START, END = 2000, 8000
LOGVAR_WINDOW_MIN, LOGVAR_STEP_MIN, STD_FLOOR = -3.0, -8.0, 0.5
NEAR_KEYS = [f"relerr/k{k}" for k in (1, 2, 3, 4, 5)] + ["emb_std/L1"]


def rows(run: Path) -> list[dict]:
    return [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]


def window(rs: list[dict], key: str, a: int, b: int) -> float:
    return float(np.mean([r[key] for r in rs if a <= r["step"] <= b]))


def health(rs: list[dict]) -> dict:
    at = {r["step"]: r for r in rs}
    after = [r for r in rs if FAR_START < r["step"] <= END]
    out = {}
    for lv in ("L1", "L2", "L3"):
        ref = at[FAR_START][f"emb_std/{lv}"]
        low = min(after, key=lambda r: r[f"emb_std/{lv}"])
        out[lv] = (ref, low[f"emb_std/{lv}"] / ref, low["step"], at[END][f"emb_std/{lv}"])
    for k in (10, 20):
        key = f"logvar/k{k}"
        out[key] = (window(rs, key, END - 499, END), min(r[key] for r in after))
    out["k5 train"] = (window(rs, "relerr/k5", 1501, 2000), window(rs, "relerr/k5", 7501, 8000))
    return out


def dev(run: Path) -> dict:
    terms = json.loads((run / "latent_dev.json").read_text())["terms"]
    return {k: terms[f"relerr/k{k}"]["mean"] for k in (1, 2, 3, 4, 5, 10, 20)}


def near_phase_gap(mean_rs: list[dict], sum_rs: list[dict]) -> dict:
    """Largest relative difference over steps 1-2000, per key."""
    s = {r["step"]: r for r in sum_rs}
    out = {}
    for key in NEAR_KEYS:
        rel = [
            abs(r[key] - s[r["step"]][key]) / abs(s[r["step"]][key])
            for r in mean_rs
            if r["step"] <= FAR_START
        ]
        out[key] = (max(rel), float(np.median(rel)))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mean", type=Path, action="append", required=True)
    p.add_argument("--sum", type=Path, action="append", required=True)
    args = p.parse_args(argv)
    k5_max = max(dev(s)[5] for s in args.sum)  # the sum arm's worse seed
    print(f"rule 3 threshold: dev relerr/k5 <= {k5_max:.5f} (the sum arm's worse seed)")
    passed = True
    for seed, (m, s) in enumerate(zip(args.mean, args.sum, strict=True)):
        mr, sr = rows(m), rows(s)
        hm, hs, dm, ds = health(mr), health(sr), dev(m), dev(s)
        print(f"== seed {seed}: mean {m.name} | sum {s.name}")
        for lv in ("L1", "L2", "L3"):
            for tag, h in (("mean", hm), ("sum ", hs)):
                ref, ratio, step, end = h[lv]
                print(
                    f"  {tag} emb_std/{lv}: step 2000 {ref:.3f}, min {ratio:.0%} of it at step "
                    f"{step}, step 8000 {end:.3f}"
                )
        ok1 = hm["L3"][1] >= STD_FLOOR
        for k in (10, 20):
            for tag, h in (("mean", hm), ("sum ", hs)):
                w, lo = h[f"logvar/k{k}"]
                print(
                    f"  {tag} logvar/k{k}: steps 7501-8000 mean {w:+.2f}, min over 2001-8000 {lo:+.2f}"
                )
        ok2 = all(
            hm[f"logvar/k{k}"][0] >= LOGVAR_WINDOW_MIN and hm[f"logvar/k{k}"][1] >= LOGVAR_STEP_MIN
            for k in (10, 20)
        )
        print("  dev relerr  k: " + " ".join(f"{k:>6}" for k in dm))
        print("        mean   : " + " ".join(f"{v:6.3f}" for v in dm.values()))
        print("        sum    : " + " ".join(f"{v:6.3f}" for v in ds.values()))
        ok3 = dm[5] <= k5_max
        for tag, h in (("mean", hm), ("sum ", hs)):
            a, b = h["k5 train"]
            print(f"  {tag} train relerr/k5: steps 1501-2000 {a:.3f} -> 7501-8000 {b:.3f}")
        for key, (mx, med) in near_phase_gap(mr, sr).items():
            print(f"  steps 1-2000, mean vs sum {key}: max rel diff {mx:.2e}, median {med:.2e}")
        print(f"  RULE 1 (L3 std >= 50% of step 2000): {'PASS' if ok1 else 'FAIL'}")
        print(f"  RULE 2 (far logvar, window >= -3, steps >= -8): {'PASS' if ok2 else 'FAIL'}")
        print(f"  RULE 3 (dev k5 relerr {dm[5]:.5f} <= {k5_max:.5f}): {'PASS' if ok3 else 'FAIL'}")
        passed &= ok1 and ok2 and ok3
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
