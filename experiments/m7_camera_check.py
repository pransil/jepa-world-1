"""M7 §3 tables: the goal head on out-of-view windows (flagged under track_lagged), lagged vs full.

  uv run python experiments/m7_camera_check.py \
      --lagged-eval <ref s0>/eval/<test-dev>.json --lagged-eval <ref s1>/eval/<...>.json \
      --full <track_full s0 dir> --full <track_full s1 dir>

The rules are in experiments/2026-11-20-m7-camera-full.md (committed before the runs).
- **Lagged (reference):** the out_of_view `visibility` and `events` from each run's test eval.
- **Full:** each run's one `matched/*-test-out_of_view-*.json` from
  `experiments/m7_matched_windows.py --flags-config configs/exp_headline_48k.yaml --flag out_of_view`.
- **Symmetry check:** each lagged run's own `matched/*-test-out_of_view-*.json` (the matched script on
  the reference with its own flags) must equal its eval's numbers exactly.
- **Verdict:** supported if the lagged k = 1 `std_ratio` is ≥ 1.2 in both seeds and both lagged seeds
  exceed both full seeds; otherwise refuted.
Reads JSON only: no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KS = ["1", "2", "3", "4", "5", "10", "20"]
SEEDS = ("s0", "s1")
BAR = 1.2


def one_matched(run: Path) -> Path:
    hits = sorted((run / "matched").glob("*-test-out_of_view-*.json"))
    if len(hits) != 1:
        raise SystemExit(f"{run}: expected one matched test out_of_view JSON, found {len(hits)}")
    return hits[0]


def same(a, b, path: str, out: dict) -> None:
    """Count leaves and collect any that differ (NaN equals NaN)."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            out["diff"].append((path, "keys"))
            return
        for k in a:
            same(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            out["diff"].append((path, "length"))
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same(x, y, f"{path}[{i}]", out)
    else:
        out["n"] += 1
        nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        if a != b and not nan:
            out["diff"].append((path, a, b))


def pair(d: dict, k: str, f) -> str:
    return " / ".join(f(d[s][k]) for s in SEEDS)


def num(key: str, nd: int):
    """Formatter for one number of a visibility dict, to `nd` decimals."""
    return lambda v: f"{v[key]:.{nd}f}"


def mean2(d: dict, k: str, key: str) -> float:
    return sum(d[s][k][key] for s in SEEDS) / len(SEEDS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lagged-eval", type=Path, action="append", required=True)
    ap.add_argument("--full", type=Path, action="append", required=True)
    a = ap.parse_args()
    if len(a.lagged_eval) != 2 or len(a.full) != 2:
        raise SystemExit("give two --lagged-eval and two --full, seed 0 first")

    lagvis, lagev, fullvis, fullev = {}, {}, {}, {}
    for s, p in zip(SEEDS, a.lagged_eval, strict=True):
        d = json.load(open(p))
        rep = d["report"]["headline"]
        assert rep["split"] == "test", p
        m = rep["metrics"]
        lagvis[s] = {k: m["visibility"][k]["out_of_view"] for k in KS}
        lagev[s] = {k: m["events"][k]["out_of_view"] for k in KS}
        g = d["meta"]["git"]
        print(
            f"lagged {s}: {p.parent.parent.name}/eval/{p.name} (sha {g['sha'][:7]}, "
            f"dirty {g['dirty']}, step {d['meta']['step']})"
        )
        sp = one_matched(p.parent.parent)
        sym = json.load(open(sp))
        assert sym["meta"]["step"] == d["meta"]["step"] and sym["meta"]["split"] == "test", sp
        out = {"n": 0, "diff": []}
        for k in KS:
            same(sym["visibility"][k], lagvis[s][k], f"visibility.{k}", out)
            same(sym["events"][k], lagev[s][k], f"events.{k}", out)
        sg = sym["meta"]["git"]
        print(
            f"  symmetry: matched/{sp.name} (sha {sg['sha'][:7]}, dirty {sg['dirty']}): "
            f"{out['n']} values compared, differing {len(out['diff'])} {out['diff'][:3]}"
        )
    for s, run in zip(SEEDS, a.full, strict=True):
        p = one_matched(run)
        d = json.load(open(p))
        meta = d["meta"]
        assert meta["split"] == "test" and meta["flag"] == "out_of_view", p
        assert "configs/exp_headline_48k.yaml" in meta["argv"], (
            "flags must come from the lagged set",
            p,
        )
        fullvis[s], fullev[s] = d["visibility"], d["events"]
        print(
            f"full {s}: {run.name}/matched/{p.name} (sha {meta['git']['sha'][:7]}, "
            f"dirty {meta['git']['dirty']}, step {meta['step']}, windows {d['windows']})"
        )

    lr = [lagvis[s]["1"]["std_ratio"] for s in SEEDS]
    fr = [fullvis[s]["1"]["std_ratio"] for s in SEEDS]
    ok = all(x >= BAR for x in lr) and min(lr) > max(fr)
    print(
        f"\nDECIDING k=1 std_ratio on out_of_view windows (flagged under track_lagged), TEST: "
        f"lagged {lr[0]:.3f} / {lr[1]:.3f}, full {fr[0]:.3f} / {fr[1]:.3f}; "
        f"n hidden {pair(lagvis, '1', lambda v: str(v['n_hidden']))} (lagged), "
        f"{pair(fullvis, '1', lambda v: str(v['n_hidden']))} (full) -> "
        f"{'SUPPORTED' if ok else 'REFUTED'}"
    )

    print("\nstd ratio (hidden ÷ visible), TEST (s0 / s1)")
    print("| k | n hidden | lagged | full | lagged ÷ full (seed means) |")
    print("|---:|---|---|---|---:|")
    for k in KS:
        print(
            f"| {k} | {pair(lagvis, k, lambda v: str(v['n_hidden']))} | "
            f"{pair(lagvis, k, num('std_ratio', 2))} | "
            f"{pair(fullvis, k, num('std_ratio', 2))} | "
            f"{mean2(lagvis, k, 'std_ratio') / mean2(fullvis, k, 'std_ratio'):.2f} |"
        )

    print("\npredicted std, m (s0 / s1)")
    print("| k | lagged, hidden | full, hidden | lagged, visible | full, visible |")
    print("|---:|---|---|---|---|")
    for k in KS:
        print(
            f"| {k} | {pair(lagvis, k, num('std_hidden', 3))} | "
            f"{pair(fullvis, k, num('std_hidden', 3))} | "
            f"{pair(lagvis, k, num('std_visible', 3))} | "
            f"{pair(fullvis, k, num('std_visible', 3))} |"
        )

    print("\nposition error, m (s0 / s1)")
    print(
        "| k | lagged, hidden | full, hidden | lagged ÷ full, hidden (seed means) "
        "| lagged, visible | full, visible |"
    )
    print("|---:|---|---|---:|---|---|")
    for k in KS:
        print(
            f"| {k} | {pair(lagvis, k, num('error_m_hidden', 3))} | "
            f"{pair(fullvis, k, num('error_m_hidden', 3))} | "
            f"{mean2(lagvis, k, 'error_m_hidden') / mean2(fullvis, k, 'error_m_hidden'):.2f} | "
            f"{pair(lagvis, k, num('error_m_visible', 3))} | "
            f"{pair(fullvis, k, num('error_m_visible', 3))} |"
        )

    print("\ncoverage on the hidden windows, joint (s0 / s1)")
    print("| k | 50%, lagged | 50%, full | 90%, lagged | 90%, full |")
    print("|---:|---|---|---|---|")
    for k in KS:

        def cov(q):
            return lambda v: f"{v['coverage_hidden'][q]['joint']:.2f}"

        print(
            f"| {k} | {pair(lagvis, k, cov('0.5'))} | {pair(fullvis, k, cov('0.5'))} | "
            f"{pair(lagvis, k, cov('0.9'))} | {pair(fullvis, k, cov('0.9'))} |"
        )

    for k in ("1", "5"):
        for side, what in (("onset", "exit"), ("offset", "re-entry")):
            e0 = lagev["s0"][k][side]
            offs = e0["offsets"]
            print(
                f"\nevents k={k} {what}: mean predicted std, m, by frames from the {what} "
                f"(0 = first frame with the flag changed)"
            )
            print("| run | " + " | ".join(f"{o:+d}" for o in offs) + " |")
            print("|---|" + "---:|" * len(offs))
            for name, ev in (("lagged", lagev), ("full", fullev)):
                for s in SEEDS:
                    cur = ev[s][k][side]["curve"]
                    print(f"| {name} {s} | " + " | ".join(f"{x:.3f}" for x in cur) + " |")
            print("| events per offset | " + " | ".join(str(x) for x in e0["n_per_offset"]) + " |")
            print(
                "delta (after − before), per event, mean: "
                + "; ".join(
                    f"{name} {s} {ev[s][k][side]['delta']:+.3f}"
                    for name, ev in (("lagged", lagev), ("full", fullev))
                    for s in SEEDS
                )
                + f" (paired events: {e0['n_paired']}, of {e0['n_events']})"
            )
            same_n = all(
                ev[s][k][side]["n_per_offset"] == e0["n_per_offset"]
                for ev in (lagev, fullev)
                for s in SEEDS
            )
            print(f"same events per offset in all four runs: {same_n}")


if __name__ == "__main__":
    main()
