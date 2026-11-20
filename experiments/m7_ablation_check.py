"""Score an M7 ablation arm against the 48k M7 reference (docs/milestones/M7-plan.md).

  uv run python experiments/m7_ablation_check.py --arm <s0 dir> --arm <s1 dir> \
      --ref <s0 dir> --ref <s1 dir> --deciding goal_k20|probe_l3_pos_cam|occluded_std_k1

Seeds pair up in the order given. Reads each run's metrics.jsonl and its one test+dev eval/*.json
at --step (default 48,000; `jepa_world.eval --split test --split dev`). Reads JSON only: no model,
no GPU.

Verdict rule, per metric: **better** if both arm seeds beat both reference seeds, **worse** if both
are beaten by both, otherwise **within** the seed spread. "Beat" means lower for errors and ratios,
higher for R^2 and std ratios.

Training-sanity gates (the thresholds of experiments/2026-11-19-headline-48k-new-defaults.md, A1-A5),
for each arm run:
- every logged float finite;
- emb_std at each level never below 50% of its step-2000 value after step 2000;
- logvar/k10 and logvar/k20 average >= -3 over the last 500 steps, and never below -8 after 2000;
- pix_share within 1e-4 of 0.10 after step 6000 (when the run logs it);
- w_goal_eff equal to 0.5 / 256 after step 4000 (when the run logs it).
Exit status 1 if any gate fails, else 0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

KS = ("1", "2", "3", "4", "5", "10", "20")
LEVELS = ("L1", "L2", "L3")
FAR, GOAL, PIX = 2000, 4000, 6000
SHARE, SHARE_TOL, W_GOAL_EFF = 0.10, 1e-4, 0.5 / 256


def rows(run: Path) -> list[dict]:
    return [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]


def window(rs: list[dict], key: str, a: int, b: int) -> float:
    vals = [r[key] for r in rs if a <= r["step"] <= b and key in r]
    return float(np.nanmean(vals)) if vals else float("nan")


def eval_report(run: Path, step: int) -> dict:
    """The run's one `jepa_world.eval --split test --split dev` report at `step`. Other files in
    eval/ (e.g. regimes scoring, whose meta has `split`, not `splits`, and no `report`) are skipped."""
    found = []
    for p in sorted(run.glob("eval/*.json")):
        d = json.loads(p.read_text())
        meta = d.get("meta", {})
        if (
            "report" in d
            and meta.get("step") == step
            and {"test", "dev"} <= set(meta.get("splits", []))
        ):
            found.append((p.name, d))
    if len(found) != 1:
        raise SystemExit(f"{run.name}: expected one step-{step} test+dev eval, found {len(found)}")
    name, d = found[0]
    git = d["meta"]["git"]
    print(f"  {run.name}: {name} (sha {git['sha'][:7]}, dirty {git['dirty']})")
    return d["report"]


def verdict(arm: list[float], ref: list[float], lower_is_better: bool = True) -> str:
    if not lower_is_better:
        arm, ref = [-x for x in arm], [-x for x in ref]
    if max(arm) < min(ref):
        return "better"
    if min(arm) > max(ref):
        return "worse"
    return "within"


def pair(xs: list[float], f: str = ".3f") -> str:
    return " / ".join(format(x, f) for x in xs)


def test_metrics(rep: dict) -> dict:
    return rep["headline"]["metrics"]


def ratio(rep: dict, split: str, base: str = "persistence") -> dict:
    m = test_metrics(rep) if split == "test" else rep["development"]["dev"]
    return m["goal_head"]["ratio_to_baseline"][base]


def probe_rmse(rep: dict, level: str, target: str) -> float:
    """Euclidean RMSE of a probe target over its dims (metres for pos_cam)."""
    dims = test_metrics(rep)["probes"]["by_level"][level][target]["rmse_per_dim"]
    return float(math.sqrt(sum(d * d for d in dims)))


def deciding_values(rep: dict, name: str) -> tuple[float, bool]:
    """(value, lower_is_better) of the deciding metric."""
    if name == "goal_k20":
        return ratio(rep, "test")["20"], True
    if name == "probe_l3_pos_cam":
        return probe_rmse(rep, "L3", "pos_cam"), True
    if name == "occluded_std_k1":
        return test_metrics(rep)["visibility"]["1"]["occluded"]["std_ratio"], False
    raise SystemExit(f"unknown --deciding {name}")


def goal_tables(arm: list[dict], ref: list[dict]) -> None:
    print("\ngoal head, TEST, ratio to persistence (s0 / s1)")
    print("| k | reference | arm | ref mean | arm mean | change | verdict |")
    print("|---:|---|---|---:|---:|---:|---|")
    for k in KS:
        a = [ratio(r, "test")[k] for r in arm]
        o = [ratio(r, "test")[k] for r in ref]
        print(
            f"| {k} | {pair(o, '.4f')} | {pair(a, '.4f')} | {np.mean(o):.3f} | {np.mean(a):.3f} | "
            f"{np.mean(a) / np.mean(o) - 1:+.1%} | {verdict(a, o)} |"
        )

    print("\nTEST, arm: / const-velocity mean (ref); / mean-parameter simulator s0 / s1 (ref mean)")
    print("| k | / const-vel | / sim-mean | error m arm s0 / s1 |")
    print("|---:|---|---|---|")
    for k in KS:
        cv = np.mean([ratio(r, "test", "const_velocity")[k] for r in arm])
        cvo = np.mean([ratio(r, "test", "const_velocity")[k] for r in ref])
        sm = [ratio(r, "test", "simulator_mean")[k] for r in arm]
        smo = np.mean([ratio(r, "test", "simulator_mean")[k] for r in ref])
        err = [test_metrics(r)["goal_head"]["error_m"][k]["mean"] for r in arm]
        print(f"| {k} | {cv:.2f} ({cvo:.2f}) | {pair(sm, '.1f')} ({smo:.1f}) | {pair(err)} |")

    print("\nDEV (development only), ratio to persistence (s0 / s1)")
    print("| run | " + " | ".join(KS) + " |")
    print("|---|" + "---|" * len(KS))
    for name, runs in (("reference", ref), ("arm", arm)):
        print(
            f"| {name} | " + " | ".join(pair([ratio(r, "dev")[k] for r in runs]) for k in KS) + " |"
        )

    print("\ncalibration, TEST: joint fraction of truths inside the interval (s0 / s1)")
    print("| k | 50% ref | 50% arm | 90% ref | 90% arm |")
    print("|---:|---|---|---|---|")
    for k in KS:

        def cov(runs, lvl, k=k):
            return [test_metrics(r)["goal_head"]["calibration"][k][lvl]["joint"] for r in runs]

        print(
            f"| {k} | {pair(cov(ref, '0.5'), '.2f')} | {pair(cov(arm, '0.5'), '.2f')} | "
            f"{pair(cov(ref, '0.9'), '.2f')} | {pair(cov(arm, '0.9'), '.2f')} |"
        )

    for kind in ("out_of_view", "occluded", "hidden"):
        if not any(
            test_metrics(r)["visibility"]["1"][kind].get("has_hidden") for r in (*arm, *ref)
        ):
            print(f"\nvisibility {kind}: no such windows in any run")
            continue
        print(
            f"\nvisibility, TEST, {kind}: n, std ratio (hidden / visible), 90% coverage when hidden"
        )
        print(
            "| k | n ref | n arm | std ratio ref | std ratio arm | 90% hidden ref | 90% hidden arm |"
        )
        print("|---:|---|---|---|---|---|---|")
        for k in KS:

            def v(runs, k=k, kind=kind):
                return [test_metrics(r)["visibility"][k][kind] for r in runs]

            def get(vs, key, f):
                vals = [
                    x.get(key, float("nan")) if x.get("has_hidden") else float("nan") for x in vs
                ]
                return pair(vals, f)

            def cov90(vs):
                vals = [
                    x["coverage_hidden"]["0.9"]["joint"] if x.get("has_hidden") else float("nan")
                    for x in vs
                ]
                return pair(vals, ".2f")

            print(
                f"| {k} | {get(v(ref), 'n_hidden', '.0f')} | {get(v(arm), 'n_hidden', '.0f')} | "
                f"{get(v(ref), 'std_ratio', '.1f')} | {get(v(arm), 'std_ratio', '.1f')} | "
                f"{cov90(v(ref))} | {cov90(v(arm))} |"
            )

    print("\nprobes, TEST: R^2, and Euclidean RMSE for pos_cam (s0 / s1)")
    print("| target | level | R^2 ref | R^2 arm | RMSE ref | RMSE arm | RMSE verdict |")
    print("|---|---|---|---|---|---|---|")
    for tgt, lvls in (
        ("pos_cam", (*LEVELS, "pooled")),
        ("radius", ("pooled",)),
        ("pos", ("pooled",)),
        ("friction", ("pooled",)),
        ("restitution", ("pooled",)),
    ):
        for lv in lvls:

            def r2(runs, lv=lv, tgt=tgt):
                return [test_metrics(r)["probes"]["by_level"][lv][tgt]["r2"] for r in runs]

            ra = [probe_rmse(r, lv, tgt) for r in arm]
            ro = [probe_rmse(r, lv, tgt) for r in ref]
            print(
                f"| {tgt} | {lv} | {pair(r2(ref))} | {pair(r2(arm))} | {pair(ro, '.4f')} | "
                f"{pair(ra, '.4f')} | {verdict(ra, ro)} |"
            )


def gates(rs: list[dict], name: str) -> bool:
    ok = True
    bad = [
        (r["step"], k) for r in rs for k, v in r.items() if isinstance(v, float) and math.isinf(v)
    ]
    nan_keys = sorted(
        {k for r in rs for k, v in r.items() if isinstance(v, float) and math.isnan(v)}
    )
    print(
        f"\n{name}: {len(rs)} steps; inf values: {len(bad)}; keys with any NaN: {nan_keys or 'none'}"
    )
    ok &= not bad
    at2000 = next(r for r in rs if r["step"] == FAR)
    for lv in LEVELS:
        lo = min(r[f"emb_std/{lv}"] for r in rs if r["step"] > FAR) / at2000[f"emb_std/{lv}"]
        ok &= lo >= 0.5
        print(
            f"  emb_std/{lv}: lowest after {FAR} / step {FAR} = {lo:.2f}; "
            f"last 1000 steps {window(rs, f'emb_std/{lv}', rs[-1]['step'] - 999, rs[-1]['step']):.3f}"
        )
    last = rs[-1]["step"]
    for k in ("10", "20"):
        key = f"logvar/k{k}"
        m = window(rs, key, last - 499, last)
        lo = min(r[key] for r in rs if r["step"] > FAR)
        ok &= m >= -3.0 and lo >= -8.0
        print(f"  {key}: last-500 mean {m:.2f}, lowest after {FAR} {lo:.2f}")
    if any("pix_share" in r for r in rs):
        dev = max(abs(r["pix_share"] - SHARE) for r in rs if r["step"] > PIX)
        ok &= dev <= SHARE_TOL
        print(f"  pix_share after {PIX}: largest |share - {SHARE}| = {dev:.2g}")
    if any("w_goal_eff" in r for r in rs):
        wrong = sum(r["w_goal_eff"] != W_GOAL_EFF for r in rs if r["step"] > GOAL)
        ok &= wrong == 0
        print(f"  w_goal_eff after {GOAL}: steps not equal to 0.5/256: {wrong}")
    print(f"  gates: {'PASS' if ok else 'FAIL'}")
    return ok


def training_tables(arm: list[list[dict]], ref: list[list[dict]]) -> None:
    last = min(rs[-1]["step"] for rs in (*arm, *ref))
    a, b = last - 999, last
    keys = [f"relerr/k{k}" for k in KS] + [f"emb_std/{lv}" for lv in LEVELS]
    keys += ["logvar/k10", "logvar/k20", "loss/pix", "w_pix_eff", "pix_share", "w_goal_eff"]
    keys += sorted({k for rs in arm for k in rs[-1] if k not in ref[0][-1]})  # arm-only keys
    print(f"\ntraining batches, steps {a:,}-{b:,} (s0 / s1)")
    print("| key | reference | arm |")
    print("|---|---|---|")
    for key in keys:
        cells = [pair([window(rs, key, a, b) for rs in runs], ".4g") for runs in (ref, arm)]
        print(f"| {key} | {cells[0]} | {cells[1]} |")
    print("\n`goalratio` on training batches, mean over each 4k window (s0 / s1)")
    print("| steps | k1 ref | k1 arm | k5 ref | k5 arm | k20 ref | k20 arm |")
    print("|---|---|---|---|---|---|---|")
    for lo in range(GOAL + 1, last, 4000):
        hi = lo + 3999
        cells = [
            pair([window(rs, f"goalratio/k{k}", lo, hi) for rs in runs])
            for k in ("1", "5", "20")
            for runs in (ref, arm)
        ]
        print(f"| {lo:,}-{hi:,} | " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", type=Path, action="append", required=True)
    p.add_argument("--ref", type=Path, action="append", required=True)
    p.add_argument(
        "--deciding", required=True, choices=("goal_k20", "probe_l3_pos_cam", "occluded_std_k1")
    )
    p.add_argument("--step", type=int, default=48000)
    args = p.parse_args(argv)
    if len(args.arm) != len(args.ref):
        raise SystemExit("--arm and --ref pair up by seed: give the same number of each")
    print("evals:")
    arm = [eval_report(r, args.step) for r in args.arm]
    ref = [eval_report(r, args.step) for r in args.ref]
    vals = [deciding_values(r, args.deciding) for r in arm]
    rvals = [deciding_values(r, args.deciding) for r in ref]
    lower = vals[0][1]
    a, o = [v for v, _ in vals], [v for v, _ in rvals]
    print(
        f"\nDECIDING ({args.deciding}, {'lower' if lower else 'higher'} is better): "
        f"reference {pair(o, '.4f')}, arm {pair(a, '.4f')} -> {verdict(a, o, lower)}"
    )
    goal_tables(arm, ref)
    arm_rows = [rows(r) for r in args.arm]
    ok = all([gates(rs, r.name) for rs, r in zip(arm_rows, args.arm, strict=True)])
    training_tables(arm_rows, [rows(r) for r in args.ref])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
