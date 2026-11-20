"""Score the 48k reference re-run at the new defaults against its pre-registered rules.

  uv run python experiments/headline_48k_new_check.py \
      --new <s0 dir> --new <s1 dir> --old <s0 dir> --old <s1 dir> \
      --mean-ref <s0 dir> --mean-ref <s1 dir>

The rules are in experiments/2026-11-19-headline-48k-new-defaults.md (committed before the runs).
--new: the runs at the new defaults; --old: the old 48k reference; --mean-ref: the collapse check's
`mean` runs (A6). Seeds pair up in the order given. Reads each run's metrics.jsonl and, for
--new and --old, its one eval/*.json at step 48,000 (`jepa_world.eval --split test --split dev`).
Prints the gates (A1-A6), the goal-head verdicts (B), the latent-head verdict (C), and the
reported-only numbers (D). Reads JSON only: no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

STEPS, FAR, GOAL, PIX = 48000, 2000, 4000, 6000
STD_FLOOR, LOGVAR_WINDOW_MIN, LOGVAR_STEP_MIN = 0.5, -3.0, -8.0
SHARE, SHARE_TOL = 0.10, 1e-4
W_GOAL_EFF = 0.5 / 256
KS = ("1", "2", "3", "4", "5", "10", "20")
NEAR = ("1", "2", "3", "4", "5")
LEVELS = ("L1", "L2", "L3")
WINDOWS = [
    (1001, 2000),
    (3001, 4000),
    (5001, 6000),
    (11001, 12000),
    (23001, 24000),
    (35001, 36000),
    (47001, 48000),
]


def rows(run: Path) -> list[dict]:
    return [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]


def window(rs: list[dict], key: str, a: int, b: int) -> float:
    vals = [r[key] for r in rs if a <= r["step"] <= b and key in r]
    return float(np.nanmean(vals)) if vals else float("nan")


def eval_report(run: Path) -> dict:
    found = []
    for p in sorted(run.glob("eval/*.json")):
        d = json.loads(p.read_text())
        if d["meta"]["step"] == STEPS and {"test", "dev"} <= set(d["meta"]["splits"]):
            found.append((p.name, d))
    if len(found) != 1:
        raise SystemExit(f"{run.name}: expected one step-{STEPS} test+dev eval, found {len(found)}")
    name, d = found[0]
    git = d["meta"]["git"]
    print(f"  {run.name}: {name} (sha {git['sha'][:7]}, dirty {git['dirty']})")
    return d["report"]


def gates(rs: list[dict]) -> bool:
    ok = True
    steps = [r["step"] for r in rs]
    a0 = steps == list(range(1, STEPS + 1))
    print(f"    steps 1..{STEPS} all present: {a0}")
    ok &= a0
    bad = sorted(
        {k for r in rs for k, v in r.items() if isinstance(v, float) and not math.isfinite(v)}
    )
    print(f"    A1 every term finite: {not bad}" + (f" (non-finite: {bad})" if bad else ""))
    ok &= not bad
    at = {r["step"]: r for r in rs}
    after = [r for r in rs if r["step"] > FAR]
    for lv in LEVELS:
        key = f"emb_std/{lv}"
        ref = at[FAR][key]
        low = min(after, key=lambda r: r[key])
        frac = low[key] / ref
        print(
            f"    A2 {lv}: step-2000 {ref:.3f}, lowest {low[key]:.3f} = {frac:.3f} at step "
            f"{low['step']}, final {at[STEPS][key]:.3f}: {frac >= STD_FLOOR}"
        )
        ok &= frac >= STD_FLOOR
    for k in ("10", "20"):
        key = f"logvar/k{k}"
        w = window(rs, key, STEPS - 499, STEPS)
        lo = min(r[key] for r in after)
        good = w >= LOGVAR_WINDOW_MIN and lo >= LOGVAR_STEP_MIN
        print(f"    A3 {key}: last-500 mean {w:+.3f}, lowest after {FAR} {lo:+.3f}: {good}")
        ok &= good
    for name, key, start, want, tol in (
        ("A4", "pix_share", PIX, SHARE, SHARE_TOL),
        ("A5", "w_goal_eff", GOAL, W_GOAL_EFF, 0.0),
    ):
        if key not in rs[-1]:
            print(f"    {name} {key}: not logged: False")
            ok = False
            continue
        on = [r[key] for r in rs if r["step"] > start]
        off = [r[key] for r in rs if r["step"] <= start]
        dev = max(abs(v - want) for v in on)
        good = dev <= tol and all(v == 0 for v in off)
        print(
            f"    {name} {key}: steps > {start} within {dev:.2e} of {want:g}, "
            f"zero through step {start}: {all(v == 0 for v in off)} -> {good}"
        )
        ok &= good
    return ok


def _same(v, w) -> bool:
    return v == w or (
        isinstance(v, float) and isinstance(w, float) and math.isnan(v) and math.isnan(w)
    )


def reproducibility(rs: list[dict], ref: list[dict]) -> None:
    """First step where a key both rows have differs (NaN equals NaN). Per row, because the
    diagnostic columns (relerr/copy/, gradnorm/) appear only on the steps they were measured."""
    by = {r["step"]: r for r in ref}
    common = sorted(k for k in ref[0] if k in rs[0] and k != "step")
    first = next(
        (
            r["step"]
            for r in rs
            if r["step"] in by
            and any(
                not _same(r[k], by[r["step"]][k])
                for k in r.keys() & by[r["step"]].keys()
                if k != "step"
            )
        ),
        None,
    )
    upto = min(len(ref), len(rs))
    print(
        f"    A6 vs the mean run: {len(common)} common keys; first differing step over 1..{upto}: "
        f"{first} (expected {GOAL + 1}): {first == GOAL + 1}"
    )


def verdict(new: list[float], old: list[float]) -> str:
    if max(new) < min(old):
        return "better"
    if min(new) > max(old):
        return "worse"
    return "within"


def pair(xs: list[float], f: str = ".3f") -> str:
    return " / ".join(format(x, f) for x in xs)


def goal_tables(new: list[dict], old: list[dict]) -> None:
    def ratio(rep, split, base="persistence"):
        m = rep["headline"]["metrics"] if split == "test" else rep["development"]["dev"]
        return m["goal_head"]["ratio_to_baseline"][base]

    print("\nB. goal head, TEST, ratio to persistence at step 48,000 (s0 / s1)")
    print("| k | old | new | new mean | old mean | change | verdict |")
    print("|---:|---|---|---:|---:|---:|---|")
    verdicts = {}
    for k in KS:
        n = [ratio(r, "test")[k] for r in new]
        o = [ratio(r, "test")[k] for r in old]
        verdicts[k] = verdict(n, o)
        print(
            f"| {k} | {pair(o, '.4f')} | {pair(n, '.4f')} | {np.mean(n):.3f} | {np.mean(o):.3f} | "
            f"{np.mean(n) / np.mean(o) - 1:+.1%} | {verdicts[k]} |"
        )
    better = [k for k in NEAR if verdicts[k] == "better"]
    near = "supported" if len(better) == 5 else "refuted" if not better else "mixed"
    print(
        f"  deciding (k=20): {verdicts['20']} -> prediction 'not worse' "
        f"{'REFUTED' if verdicts['20'] == 'worse' else 'holds'}"
    )
    print(f"  secondary (k=1-5): better at {better or 'none'} -> {near}")

    print(
        "\nD. test, mean of seeds: new vs const-velocity, new vs mean-parameter simulator (s0 / s1)"
    )
    print("| k | / const-vel new (old) | / sim-mean new s0 / s1 (old) | error m new s0 / s1 |")
    print("|---:|---|---|---|")
    for k in KS:
        cv = np.mean([ratio(r, "test", "const_velocity")[k] for r in new])
        cvo = np.mean([ratio(r, "test", "const_velocity")[k] for r in old])
        sm = [ratio(r, "test", "simulator_mean")[k] for r in new]
        smo = np.mean([ratio(r, "test", "simulator_mean")[k] for r in old])
        err = [r["headline"]["metrics"]["goal_head"]["error_m"][k]["mean"] for r in new]
        print(f"| {k} | {cv:.2f} ({cvo:.2f}) | {pair(sm, '.1f')} ({smo:.1f}) | {pair(err)} |")

    print("\nD. DEV (development only), ratio to persistence (s0 / s1)")
    print("| k | " + " | ".join(KS) + " |")
    print("|---|" + "---|" * len(KS))
    for name, runs in (("old 48k", old), ("new 48k", new)):
        cells = [pair([ratio(r, "dev")[k] for r in runs]) for k in KS]
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\nD. calibration, TEST: joint fraction of truths inside the interval (s0 / s1)")
    print("| k | 50% old | 50% new | 90% old | 90% new |")
    print("|---:|---|---|---|---|")
    for k in KS:

        def cov(runs, lvl, k=k):
            return [
                r["headline"]["metrics"]["goal_head"]["calibration"][k][lvl]["joint"] for r in runs
            ]

        print(
            f"| {k} | {pair(cov(old, '0.5'), '.2f')} | {pair(cov(new, '0.5'), '.2f')} | "
            f"{pair(cov(old, '0.9'), '.2f')} | {pair(cov(new, '0.9'), '.2f')} |"
        )

    print(
        "\nD. visibility, TEST (ball hidden): std ratio hidden/visible, 90% coverage hidden (s0 / s1)"
    )
    print("| k | std ratio old | std ratio new | 90% hidden old | 90% hidden new |")
    print("|---:|---|---|---|---|")
    for k in KS:

        def vis(runs, k=k):
            return [r["headline"]["metrics"]["visibility"][k]["hidden"] for r in runs]

        def sr(vs):
            return pair([v["std_ratio"] for v in vs], ".1f")

        def ch(vs):
            return pair([v["coverage_hidden"]["0.9"]["joint"] for v in vs], ".2f")

        print(f"| {k} | {sr(vis(old))} | {sr(vis(new))} | {ch(vis(old))} | {ch(vis(new))} |")

    print("\nD. probes, TEST R^2 (s0 / s1)")
    print("| target | level | old | new |")
    print("|---|---|---|---|")
    for tgt, lvls in (
        ("pos_cam", ("L1", "L2", "L3", "pooled")),
        ("radius", ("pooled",)),
        ("pos", ("pooled",)),
    ):
        for lv in lvls:

            def r2(runs, lv=lv, tgt=tgt):
                return [r["headline"]["metrics"]["probes"]["by_level"][lv][tgt]["r2"] for r in runs]

            print(f"| {tgt} | {lv} | {pair(r2(old))} | {pair(r2(new))} |")


def latent_tables(new: list[list[dict]], old: list[list[dict]]) -> None:
    print("\nC. near k5 relerr, train batches, steps 47,001-48,000 (s0 / s1)")
    n = [window(rs, "relerr/k5", 47001, STEPS) for rs in new]
    o = [window(rs, "relerr/k5", 47001, STEPS) for rs in old]
    v = verdict(n, o)
    print(
        f"  old {pair(o)}, new {pair(n)}: {v} -> prediction 'lower' "
        f"{'supported' if v == 'better' else 'not supported'}"
    )
    f = [window(rs, "relerr/k20", 47001, STEPS) for rs in new]
    fo = [window(rs, "relerr/k20", 47001, STEPS) for rs in old]
    print(
        f"  far k20: old {pair(fo)}, new {pair(f)} ({np.mean(f) / np.mean(fo) - 1:+.1%} on the mean)"
    )

    print(
        "\nD. latent relerr over training, 1000-step windows (train batches; new s0 / s1, then old)"
    )
    keys = [f"relerr/k{k}" for k in KS]
    print("| steps | " + " | ".join(k.split("/")[1] for k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for a, b in WINDOWS:
        cells = [pair([window(rs, key, a, b) for rs in new]) for key in keys]
        print(f"| {a:,}-{b:,} new | " + " | ".join(cells) + " |")
        cells = [pair([window(rs, key, a, b) for rs in old]) for key in keys]
        print(f"| {a:,}-{b:,} old | " + " | ".join(cells) + " |")

    print("\nD. emb_std per level and far logvar, same windows (new s0 / s1, then old)")
    keys = [f"emb_std/{lv}" for lv in LEVELS] + ["logvar/k10", "logvar/k20"]
    print("| steps | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for a, b in WINDOWS:
        for name, runs in (("new", new), ("old", old)):
            cells = [pair([window(rs, key, a, b) for rs in runs], ".2f") for key in keys]
            print(f"| {a:,}-{b:,} {name} | " + " | ".join(cells) + " |")

    print("\nD. pixel term (new s0 / s1; old loss/pix)")
    print("| steps | loss/pix new | loss/pix old | w_pix_eff new | pix_share new |")
    print("|---|---|---|---|---|")
    for a, b in [(6001, 6250)] + [w for w in WINDOWS if w[0] > PIX]:
        print(
            f"| {a:,}-{b:,} | {pair([window(rs, 'loss/pix', a, b) for rs in new], '.5f')} | "
            f"{pair([window(rs, 'loss/pix', a, b) for rs in old], '.5f')} | "
            f"{pair([window(rs, 'w_pix_eff', a, b) for rs in new], '.0f')} | "
            f"{pair([window(rs, 'pix_share', a, b) for rs in new], '.4f')} |"
        )

    print("\nD. goalratio on training batches, 4k windows (new s0 / s1, then old)")
    print("| steps | k1 new | k1 old | k5 new | k5 old | k20 new | k20 old |")
    print("|---|---|---|---|---|---|---|")
    for a in range(GOAL + 1, STEPS, 4000):
        b = a + 3999
        cells = []
        for k in ("1", "5", "20"):
            for runs in (new, old):
                cells.append(pair([window(rs, f"goalratio/k{k}", a, b) for rs in runs]))
        print(f"| {a:,}-{b:,} | " + " | ".join(cells) + " |")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--new", type=Path, action="append", required=True)
    p.add_argument("--old", type=Path, action="append", required=True)
    p.add_argument("--mean-ref", type=Path, action="append", required=True)
    args = p.parse_args(argv)
    new_rows = [rows(r) for r in args.new]
    old_rows = [rows(r) for r in args.old]
    print("A. gates")
    ok = True
    for seed, (run, rs, ref) in enumerate(zip(args.new, new_rows, args.mean_ref, strict=True)):
        print(f"  seed {seed}: {run.name}")
        ok &= gates(rs)
        reproducibility(rs, rows(ref))
    print(f"  A1-A5 in both seeds: {'PASS' if ok else 'FAIL'}")
    print("\nevals")
    new_rep = [eval_report(r) for r in args.new]
    old_rep = [eval_report(r) for r in args.old]
    goal_tables(new_rep, old_rep)
    latent_tables(new_rows, old_rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
