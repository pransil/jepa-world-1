"""Score the M7 §5 occlusion rules from the obstacle runs' evals (experiments/2026-11-20-m7-obstacles.md).

  uv run python experiments/m7_occlusion_check.py --run <s0 dir> --run <s1 dir> [--step 48000]

Per run, from its one `jepa_world.eval --split test --split dev` report at --step (test split):
1. **rise:** at k = 1, the goal head's mean predictive std on windows whose last context frame is
   `occluded`, divided by the same on the other windows (`std_ratio`), is at least 1.2;
2. **calibration:** at k = 1, the joint 90% interval holds at least 0.8 of the truths on occluded
   windows;
3. **recovery:** at k = 1, the mean std 3 frames after the ball reappears (the `occluded` offset
   curve at offset +3, metrics.event_response), divided by the std on non-occluded windows, is at
   most 1.2.

The hypothesis holds if all three hold in every run, and is refuted if any rule fails in every
run. Anything else is mixed. It also prints every horizon's occluded and out-of-view numbers, and
the k = 1 onset and offset curves. Reads JSON only: no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RISE, COVERAGE, RECOVERY, AFTER = 1.2, 0.8, 1.2, 3


def eval_report(run: Path, step: int) -> dict:
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
    return d["report"]["headline"]["metrics"]


def fmt(x, f=".3f") -> str:
    return "n/a" if x is None else format(x, f)


def rules(m: dict) -> dict:
    """The three rules at k = 1, as {name: (value, holds)}; value None if there is nothing to score."""
    v = m["visibility"]["1"]["occluded"]
    if not v["has_hidden"]:
        return {name: (None, None) for name in ("rise", "calibration", "recovery")}
    off = m["events"]["1"]["occluded"]["offset"]
    curve = dict(zip(off["offsets"], off["curve"], strict=True))
    after = curve.get(AFTER)
    rec = None if after is None else after / v["std_visible"]
    cov = v["coverage_hidden"]["0.9"]["joint"]
    return {
        "rise": (v["std_ratio"], v["std_ratio"] >= RISE),
        "calibration": (cov, cov >= COVERAGE),
        "recovery": (rec, None if rec is None else rec <= RECOVERY),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, action="append", required=True)
    p.add_argument("--step", type=int, default=48000)
    args = p.parse_args(argv)
    print("evals:")
    ms = [eval_report(r, args.step) for r in args.run]

    for kind in ("occluded", "out_of_view"):
        print(f"\nTEST, {kind}, per horizon (runs in order given, separated by /)")
        print(
            "| k | n hidden | std hidden | std visible | std ratio | 50% hidden | 90% hidden | "
            "error m hidden | error m visible |"
        )
        print("|---:|---|---|---|---|---|---|---|---|")
        for k in m_keys(ms[0]):
            vs = [m["visibility"][k][kind] for m in ms]

            def col(key, f=".3f", vs=vs):
                return " / ".join(fmt(v.get(key) if v["has_hidden"] else None, f) for v in vs)

            def cov(lvl, vs=vs):
                return " / ".join(
                    fmt(v["coverage_hidden"][lvl]["joint"] if v["has_hidden"] else None, ".2f")
                    for v in vs
                )

            n = " / ".join(str(v["n_hidden"]) for v in vs)
            print(
                f"| {k} | {n} | {col('std_hidden', '.4f')} | {col('std_visible', '.4f')} | "
                f"{col('std_ratio', '.2f')} | {cov('0.5')} | {cov('0.9')} | "
                f"{col('error_m_hidden')} | {col('error_m_visible')} |"
            )

    print(
        "\nk = 1, occluded: mean predictive std around onsets and offsets (offset 0 = event frame)"
    )
    for edge in ("onset", "offset"):
        for i, m in enumerate(ms):
            e = m["events"]["1"]["occluded"][edge]
            cells = ", ".join(
                f"{o:+d}: {fmt(c, '.4f')}" for o, c in zip(e["offsets"], e["curve"], strict=True)
            )
            print(
                f"  run {i} {edge} (n_events {e['n_events']}, delta {fmt(e['delta'], '.4f')}): {cells}"
            )

    print("\nrules at k = 1 (value, holds):")
    held = []
    for i, m in enumerate(ms):
        r = rules(m)
        held.append(r)
        print(
            f"  run {i}: rise {fmt(r['rise'][0], '.2f')} (>= {RISE}: {r['rise'][1]}), "
            f"90% coverage {fmt(r['calibration'][0], '.2f')} (>= {COVERAGE}: {r['calibration'][1]}), "
            f"recovery at +{AFTER} {fmt(r['recovery'][0], '.2f')} (<= {RECOVERY}: {r['recovery'][1]})"
        )
    names = ("rise", "calibration", "recovery")
    if any(r[n][1] is None for r in held for n in names):
        print("VERDICT: not scorable (no occluded windows, or no event at +3)")
    elif all(r[n][1] for r in held for n in names):
        print("VERDICT: SUPPORTED (all three rules hold in every run)")
    elif any(all(not r[n][1] for r in held) for n in names):
        failed = [n for n in names if all(not r[n][1] for r in held)]
        print(f"VERDICT: REFUTED (fails in every run: {failed})")
    else:
        print("VERDICT: MIXED (the runs disagree)")
    return 0


def m_keys(m: dict) -> list[str]:
    return sorted(m["visibility"], key=int)


if __name__ == "__main__":
    raise SystemExit(main())
