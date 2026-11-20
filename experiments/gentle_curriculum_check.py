"""Score H, the gentle-world curriculum, against its pre-registered rules.

  uv run python experiments/gentle_curriculum_check.py --arm <H s0 dir> [--arm <H s1 dir>] \
      --ref <reference s0 dir> --ref <reference s1 dir> [--ref36]

The rules are in experiments/2026-11-19-gentle-curriculum.md (committed before the runs). Reads
each run's metrics.jsonl, its one step-48,000 `jepa_world.eval --split test --split dev` report,
and its latest `jepa_world.regimes --split test` file at step 48,000. With --ref36 it also reads
the reference runs' step-36,000 eval (matched full-world steps, D). The gates, the dev,
calibration, visibility and probe tables, and the last-1000-step training table come from
experiments/m7_ablation_check.py. Reads JSON only: no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m7_ablation_check as m7  # noqa: E402

from jepa_world.utils.config import deep_merge, load_config  # noqa: E402

STEPS, SWITCH = 48000, 12000
KS, NEAR = m7.KS, ("1", "2", "3", "4", "5")
REGIMES = ("rolling", "airborne", "bounce")
GENTLE = Path(__file__).resolve().parents[1] / "configs" / "world_gentle.yaml"


def regimes(run: Path, step: int = STEPS) -> dict:
    found = []
    for p in sorted(run.glob("eval/regimes/*.json")):
        d = json.loads(p.read_text())
        if d["meta"].get("step") == step and d["meta"].get("split") == "test":
            found.append((p.name, d))
    if not found:
        raise SystemExit(f"{run.name}: no step-{step} test regimes file in eval/regimes/")
    name, d = found[-1]
    print(
        f"  {run.name}: {name} (sha {d['meta']['git']['sha'][:7]}, dirty {d['meta']['git']['dirty']})"
    )
    return d["regimes"]


def switch_report(run: Path, rs: list[dict], ref: list[dict]) -> None:
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    cur = cfg["data"].get("curriculum")
    if cur is None:
        print(f"  {run.name}: no data.curriculum in its config.yaml")
        return
    gentle = load_config(GENTLE)["sim"]
    merged = deep_merge(cfg["sim"], cur["sim"])
    print(
        f"  config: curriculum.steps {cur['steps']} (want {SWITCH}); gentle sim = "
        f"configs/world_gentle.yaml's: {merged == gentle}"
    )
    near = [f"relerr/k{k}" for k in NEAR]

    def nr(rows, a, b):
        return float(np.mean([m7.window(rows, key, a, b) for key in near]))

    print(
        f"  loss/total 11,901-12,000 {m7.window(rs, 'loss/total', 11901, 12000):.4f} -> "
        f"12,001-12,100 {m7.window(rs, 'loss/total', 12001, 12100):.4f} "
        f"(reference same steps {m7.window(ref, 'loss/total', 12001, 12100):.4f})"
    )
    print(
        f"  near relerr k1-5 mean 11,901-12,000 {nr(rs, 11901, 12000):.4f} -> 12,001-12,100 "
        f"{nr(rs, 12001, 12100):.4f} (reference same steps {nr(ref, 12001, 12100):.4f})"
    )
    back = next(
        (
            a
            for a in range(SWITCH + 1, STEPS, 100)
            if abs(nr(rs, a, a + 99) / nr(ref, a, a + 99) - 1) <= 0.10
        ),
        None,
    )
    print(
        "  first 100-step window after the switch with near relerr k1-5 within 10% of the "
        f"reference's at the same steps: {back if back else 'never'}"
    )


def deciding(arm: list[dict], ref: list[dict]) -> None:
    prov = " (provisional: one seed)" if len(arm) == 1 else ""
    print(f"\nB. DECIDING: goal head, full-world TEST, ratio to persistence, step 48,000{prov}")
    print("| k | reference s0 / s1 | H | ref mean | H mean | change | verdict |")
    print("|---:|---|---|---:|---:|---:|---|")
    v = {}
    for k in KS:
        a = [m7.ratio(r, "test")[k] for r in arm]
        o = [m7.ratio(r, "test")[k] for r in ref]
        v[k] = m7.verdict(a, o)
        print(
            f"| {k} | {m7.pair(o, '.4f')} | {m7.pair(a, '.4f')} | {np.mean(o):.4f} | "
            f"{np.mean(a):.4f} | {np.mean(a) / np.mean(o) - 1:+.1%} | {v[k]} |"
        )
    s0 = m7.ratio(arm[0], "test")
    o0, o1 = (m7.ratio(r, "test") for r in ref)
    d20, sp20 = abs(s0["20"] - (o0["20"] + o1["20"]) / 2), abs(o0["20"] - o1["20"])
    m = [np.mean([r[k] for k in NEAR]) for r in (s0, o0, o1)]
    dn, spn = abs(m[0] - (m[1] + m[2]) / 2), abs(m[1] - m[2])
    print(
        f"  seed-1 trigger: k20 |s0 - ref mean| {d20:.4f} vs spread {sp20:.4f} -> {d20 > sp20}; "
        f"k1-5 mean {m[0]:.4f}: |s0 - ref mean| {dn:.4f} vs spread {spn:.4f} -> {dn > spn}; "
        f"RUN SEED 1: {d20 > sp20 or dn > spn}"
    )
    if v["20"] == "better" and "worse" not in v.values():
        h = "SUPPORTED"
    elif v["20"] == "worse" or set(v.values()) == {"within"}:
        h = "REFUTED"
    else:
        h = "MIXED"
    print(f"  hypothesis{prov}: {h} (per k: {v})")


def regime_table(arm: list[dict], ref: list[dict]) -> None:
    prov = " (provisional: one seed)" if len(arm) == 1 else ""
    print(f"\nB2. per regime, TEST, goal head ratio to persistence{prov}")
    print("| k | regime | share | reference s0 / s1 | H | verdict | H / const-vel fd (ref) |")
    print("|---:|---|---:|---|---|---|---|")
    v = {g: {} for g in REGIMES}
    for k in KS:
        for g in REGIMES:
            a = [r[k][g]["ratio"]["persistence"] for r in arm]
            o = [r[k][g]["ratio"]["persistence"] for r in ref]
            fa = np.mean([r[k][g]["ratio"]["const_velocity_fd"] for r in arm])
            fo = np.mean([r[k][g]["ratio"]["const_velocity_fd"] for r in ref])
            v[g][k] = m7.verdict(a, o)
            print(
                f"| {k} | {g} | {arm[0][k][g]['share']:.2f} | {m7.pair(o, '.4f')} | "
                f"{m7.pair(a, '.4f')} | {v[g][k]} | {fa:.2f} ({fo:.2f}) |"
            )
    better = sum(x == "better" for x in v["rolling"].values())
    worse = [k for k, x in v["bounce"].items() if x == "worse"]
    print(
        f"  rolling better at {better} of {len(KS)} k; bounce worse at {worse or 'no k'} -> "
        f"'rolling improved without hurting bounces': {better >= 4 and not worse}"
    )


def latent_c(arm: list[list[dict]], ref: list[list[dict]]) -> None:
    print("\nC. latent relerr, full-world training batches, steps 47,001-48,000")
    print("| k | reference s0 / s1 | H | verdict |")
    print("|---:|---|---|---|")
    for k in KS:
        a = [m7.window(rs, f"relerr/k{k}", 47001, STEPS) for rs in arm]
        o = [m7.window(rs, f"relerr/k{k}", 47001, STEPS) for rs in ref]
        print(f"| {k} | {m7.pair(o, '.4f')} | {m7.pair(a, '.4f')} | {m7.verdict(a, o)} |")


def curves(arm: list[list[dict]], ref: list[list[dict]]) -> None:
    print("\nD. relerr over training, 1000-step windows (H before 12,001: GENTLE batches)")
    keys = [f"relerr/k{k}" for k in KS]
    print("| steps | run | " + " | ".join(k.split("/")[1] for k in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for a in (1001, 3001, 5001, 11001, 12001, 13001, 17001, 23001, 35001, 47001):
        b = a + 999
        for name, runs in (("H", arm), ("ref", ref)):
            cells = [m7.pair([m7.window(rs, key, a, b) for rs in runs]) for key in keys]
            print(f"| {a:,}-{b:,} | {name} | " + " | ".join(cells) + " |")


def matched_steps(arm: list[dict], ref36: list[dict], ref: list[dict]) -> None:
    print("\nD. matched full-world steps: the reference at step 36,000 (s0 / s1)")
    print("| k | ref @36k | ref @48k | H @48k | H vs ref@36k | H vs ref@48k |")
    print("|---:|---|---|---|---|---|")
    for k in KS:
        a = [m7.ratio(r, "test")[k] for r in arm]
        o36 = [m7.ratio(r, "test")[k] for r in ref36]
        o48 = [m7.ratio(r, "test")[k] for r in ref]
        print(
            f"| {k} | {m7.pair(o36, '.4f')} | {m7.pair(o48, '.4f')} | {m7.pair(a, '.4f')} | "
            f"{m7.verdict(a, o36)} | {m7.verdict(a, o48)} |"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", type=Path, action="append", required=True)
    p.add_argument("--ref", type=Path, action="append", required=True)
    p.add_argument("--ref36", action="store_true")
    args = p.parse_args(argv)
    if len(args.ref) != 2 or not 1 <= len(args.arm) <= 2:
        raise SystemExit("give one or two --arm runs and the two --ref runs")
    arm_rows = [m7.rows(r) for r in args.arm]
    ref_rows = [m7.rows(r) for r in args.ref]
    print("A. gates")
    ok = all(m7.gates(rs, r.name) for rs, r in zip(arm_rows, args.arm, strict=True))
    print("\nthe switch")
    for run, rs in zip(args.arm, arm_rows, strict=True):
        switch_report(run, rs, ref_rows[0])
    print("\nevals")
    arm = [m7.eval_report(r, STEPS) for r in args.arm]
    ref = [m7.eval_report(r, STEPS) for r in args.ref]
    deciding(arm, ref)
    print("\nregimes files")
    regime_table([regimes(r) for r in args.arm], [regimes(r) for r in args.ref])
    latent_c(arm_rows, ref_rows)
    if args.ref36:
        print("\nreference step-36,000 evals")
        matched_steps(arm, [m7.eval_report(r, 36000) for r in args.ref], ref)
    print("\nD. (m7_ablation_check tables: arm = H)")
    m7.goal_tables(arm, ref)
    m7.training_tables(arm_rows, ref_rows)
    curves(arm_rows, ref_rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
