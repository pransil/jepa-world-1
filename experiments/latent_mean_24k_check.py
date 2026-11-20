"""Score per-channel latents against channel-summed ones at 24k steps, with the pixel share at 10%.

  uv run python experiments/latent_mean_24k_check.py \
      --mean <s0 dir> --mean <s1 dir> --sum <s0 dir> --sum <s1 dir>

The rules are in experiments/2026-11-19-latent-mean-at-24k.md (committed before the evals).
--mean: the 48k reference at the new defaults (`latent_reduce: mean`), scored at its step-24,000
checkpoint; --sum: the 24k `m7_pixel_share10` runs (`sum`). Seeds pair up in the order given.
Reads each run's metrics.jsonl and its step-24,000 test+dev eval/*.json (`jepa_world.eval`), and
the --mean runs' step-48,000 eval for context. Reads JSON only: no model, no GPU.

Control: if a --sum run holds more than one step-24,000 test+dev eval (the original, plus a
re-eval at the current commit), the script compares every number in their reports and says
whether they are identical. It then scores the earliest one (the original), unless
--sum-eval latest is given.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

STEP = 24000
KS = ("1", "2", "3", "4", "5", "10", "20")
NEAR = ("1", "2", "3", "4", "5")
LEVELS = ("L1", "L2", "L3")


def rows(run: Path) -> list[dict]:
    return [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]


def window(rs: list[dict], key: str, a: int, b: int) -> float:
    vals = [r[key] for r in rs if a <= r["step"] <= b and key in r]
    return float(np.nanmean(vals)) if vals else float("nan")


def evals_at(run: Path, step: int) -> list[tuple[str, dict]]:
    """Every test+dev eval of `run` at `step`, oldest first (names start with a UTC stamp). Other
    files in eval/ (e.g. regimes scoring: meta has `split`, not `splits`, and no report) are skipped."""
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
    return found


def numbers(x, path=""):
    """Every numeric leaf of a JSON value, keyed by its path."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numbers(v, f"{path}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numbers(v, f"{path}[{i}]")
    elif isinstance(x, int | float) and not isinstance(x, bool):
        yield path, float(x)


def compare_reports(a: dict, b: dict) -> tuple[int, int, float]:
    """(numbers compared, numbers that differ, largest absolute difference). NaN equals NaN."""
    na, nb = dict(numbers(a)), dict(numbers(b))
    if set(na) != set(nb):
        raise SystemExit(f"control: the reports have different keys ({len(set(na) ^ set(nb))})")
    differ, worst = 0, 0.0
    for k, v in na.items():
        w = nb[k]
        if (math.isnan(v) and math.isnan(w)) or v == w:
            continue
        differ += 1
        worst = max(worst, abs(v - w) if not (math.isnan(v) or math.isnan(w)) else math.inf)
    return len(na), differ, worst


def pick_eval(run: Path, step: int, which: str = "earliest") -> dict:
    found = evals_at(run, step)
    if not found:
        raise SystemExit(f"{run.name}: no step-{step} test+dev eval")
    for name, d in found:
        git = d["meta"]["git"]
        print(f"  {run.name}: {name} (step {step}, sha {git['sha'][:7]}, dirty {git['dirty']})")
    if len(found) > 1:
        first = found[0][1]["report"]
        for name, d in found[1:]:
            n, differ, worst = compare_reports(first, d["report"])
            same = "IDENTICAL" if differ == 0 else f"{differ} differ, largest |diff| {worst:.3g}"
            print(f"    control: {found[0][0]} vs {name}: {n} numbers, {same}")
    name, d = found[0] if which == "earliest" else found[-1]
    print(f"    scoring {name}")
    return d["report"]


def verdict(new: list[float], old: list[float]) -> str:
    if max(new) < min(old):
        return "better"
    if min(new) > max(old):
        return "worse"
    return "within"


def pair(xs: list[float], f: str = ".3f") -> str:
    return " / ".join(format(x, f) for x in xs)


def ratio(rep: dict, split: str, base: str = "persistence") -> dict:
    m = rep["headline"]["metrics"] if split == "test" else rep["development"]["dev"]
    return m["goal_head"]["ratio_to_baseline"][base]


def goal_tables(mean: list[dict], summ: list[dict], mean48: list[dict]) -> None:
    print(f"\nB. goal head, TEST, ratio to persistence at step {STEP:,} (s0 / s1)")
    print(
        "| k | sum (m7_pixel_share10) | mean (step 24k) | sum mean | mean mean | change | verdict "
        "| mean runs at 48k |"
    )
    print("|---:|---|---|---:|---:|---:|---|---|")
    verdicts = {}
    for k in KS:
        n = [ratio(r, "test")[k] for r in mean]
        o = [ratio(r, "test")[k] for r in summ]
        late = [ratio(r, "test")[k] for r in mean48]
        verdicts[k] = verdict(n, o)
        print(
            f"| {k} | {pair(o, '.4f')} | {pair(n, '.4f')} | {np.mean(o):.3f} | {np.mean(n):.3f} | "
            f"{np.mean(n) / np.mean(o) - 1:+.1%} | {verdicts[k]} | {pair(late, '.4f')} |"
        )
    v20 = verdicts["20"]
    word = {"better": "SUPPORTED", "worse": "REFUTED", "within": "INCONCLUSIVE"}[v20]
    print(f"  deciding (k=20): {v20} -> hypothesis {word}")
    print(f"  k=1-5 verdicts: {[verdicts[k] for k in NEAR]}")

    print(
        "\nD. TEST, mean of seeds: / const-velocity; / mean-parameter simulator (s0 / s1); error m"
    )
    print(
        "| k | / const-vel mean (sum) | / sim-mean mean s0 / s1 (sum mean) | error m mean s0 / s1 |"
    )
    print("|---:|---|---|---|")
    for k in KS:
        cv = np.mean([ratio(r, "test", "const_velocity")[k] for r in mean])
        cvo = np.mean([ratio(r, "test", "const_velocity")[k] for r in summ])
        sm = [ratio(r, "test", "simulator_mean")[k] for r in mean]
        smo = np.mean([ratio(r, "test", "simulator_mean")[k] for r in summ])
        err = [r["headline"]["metrics"]["goal_head"]["error_m"][k]["mean"] for r in mean]
        print(f"| {k} | {cv:.2f} ({cvo:.2f}) | {pair(sm, '.1f')} ({smo:.1f}) | {pair(err)} |")

    print("\nD. DEV (development only), ratio to persistence (s0 / s1)")
    print("| k | " + " | ".join(KS) + " |")
    print("|---|" + "---|" * len(KS))
    for name, runs in (("sum", summ), ("mean", mean)):
        print(
            f"| {name} | " + " | ".join(pair([ratio(r, "dev")[k] for r in runs]) for k in KS) + " |"
        )
    print(
        "dev verdicts: "
        + ", ".join(
            f"k{k} {verdict([ratio(r, 'dev')[k] for r in mean], [ratio(r, 'dev')[k] for r in summ])}"
            for k in KS
        )
    )

    print("\nD. calibration, TEST: joint fraction of truths inside the interval (s0 / s1)")
    print("| k | 50% sum | 50% mean | 90% sum | 90% mean |")
    print("|---:|---|---|---|---|")
    for k in KS:

        def cov(runs, lvl, k=k):
            return [
                r["headline"]["metrics"]["goal_head"]["calibration"][k][lvl]["joint"] for r in runs
            ]

        print(
            f"| {k} | {pair(cov(summ, '0.5'), '.2f')} | {pair(cov(mean, '0.5'), '.2f')} | "
            f"{pair(cov(summ, '0.9'), '.2f')} | {pair(cov(mean, '0.9'), '.2f')} |"
        )

    print("\nD. visibility, TEST (ball hidden): std ratio hidden/visible, 90% coverage hidden")
    print("| k | std ratio sum | std ratio mean | 90% hidden sum | 90% hidden mean |")
    print("|---:|---|---|---|---|")
    for k in KS:

        def vis(runs, k=k):
            return [r["headline"]["metrics"]["visibility"][k]["hidden"] for r in runs]

        sr = [pair([v["std_ratio"] for v in vis(x)], ".1f") for x in (summ, mean)]
        ch = [
            pair([v["coverage_hidden"]["0.9"]["joint"] for v in vis(x)], ".2f")
            for x in (summ, mean)
        ]
        print(f"| {k} | {sr[0]} | {sr[1]} | {ch[0]} | {ch[1]} |")

    print("\nD. probes, TEST R^2 (s0 / s1)")
    print("| target | level | sum | mean |")
    print("|---|---|---|---|")
    for tgt, lvls in (
        ("pos_cam", (*LEVELS, "pooled")),
        ("radius", ("pooled",)),
        ("pos", ("pooled",)),
    ):
        for lv in lvls:

            def r2(runs, lv=lv, tgt=tgt):
                return [r["headline"]["metrics"]["probes"]["by_level"][lv][tgt]["r2"] for r in runs]

            print(f"| {tgt} | {lv} | {pair(r2(summ))} | {pair(r2(mean))} |")


def training_tables(mean: list[list[dict]], summ: list[list[dict]]) -> None:
    print("\nD. training batches, steps 23,001-24,000 (s0 / s1)")
    print("| key | sum | mean |")
    print("|---|---|---|")
    keys = [f"relerr/k{k}" for k in KS] + [f"emb_std/{lv}" for lv in LEVELS]
    keys += ["logvar/k10", "logvar/k20", "loss/pix", "w_pix_eff", "pix_share", "w_goal_eff"]
    for key in keys:
        cells = [
            pair([window(rs, key, 23001, STEP) for rs in runs], ".4g") for runs in (summ, mean)
        ]
        print(f"| {key} | {cells[0]} | {cells[1]} |")
    print("\n`goalratio` on training batches, mean over each 4k window (s0 / s1)")
    print("| steps | k1 sum | k1 mean | k5 sum | k5 mean | k20 sum | k20 mean |")
    print("|---|---|---|---|---|---|---|")
    for a in range(4001, STEP, 4000):
        b = a + 3999
        cells = []
        for k in ("1", "5", "20"):
            for runs in (summ, mean):
                cells.append(pair([window(rs, f"goalratio/k{k}", a, b) for rs in runs]))
        print(f"| {a:,}-{b:,} | " + " | ".join(cells) + " |")
    finite = [
        all(
            math.isfinite(v)
            for r in rs[:STEP]
            for k, v in r.items()
            if isinstance(v, float) and k != "step"
        )
        for rs in (*summ, *mean)
    ]
    print(f"\nevery logged float finite through step {STEP:,} (sum s0, s1, mean s0, s1): {finite}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--mean", type=Path, action="append", required=True)
    p.add_argument("--sum", type=Path, action="append", required=True)
    p.add_argument("--sum-eval", choices=("earliest", "latest"), default="earliest")
    args = p.parse_args(argv)
    if len(args.mean) != len(args.sum):
        raise SystemExit("--mean and --sum pair up by seed: give the same number of each")
    print("evals:")
    mean = [pick_eval(r, STEP) for r in args.mean]
    summ = [pick_eval(r, STEP, args.sum_eval) for r in args.sum]
    mean48 = [pick_eval(r, 48000) for r in args.mean]
    goal_tables(mean, summ, mean48)
    training_tables([rows(r) for r in args.mean], [rows(r) for r in args.sum])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
