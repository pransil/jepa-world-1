"""Does a re-run repeat a recorded run? Compare two metrics.jsonl files, bitwise.

  uv run python experiments/compare_metrics.py --run <re-run dir> --ref <original dir>

Every step of --run is matched to the same step of --ref (which may be longer, e.g. a short re-run of
an 8000-step run). Prints the rows and keys compared, the number of differing values, and the largest
absolute difference (inf when one side is NaN and the other isn't). NaN equals NaN. Exit status 0 only
if --run has at least one row, every one of its steps is in --ref with the same keys, and every value
is identical.

Diagnostic columns (train.DIAGNOSTIC_PREFIXES: relerr/copy/, gradnorm/) are logged only on their
own steps, and runs from before they existed have none. A diagnostic key on only one side is
counted and reported, not a mismatch. Every other key must be on both sides, and every key both
sides have must match bitwise. No flag is needed: m7's re-run check against the pre-diagnostics
48k reference calls it as above.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from jepa_world.train import DIAGNOSTIC_PREFIXES


def rows(run: Path) -> list[dict]:
    return [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]


def diff(v, w) -> float | None:
    """None if identical (NaN == NaN), else the absolute difference (inf for NaN vs a number)."""
    v_nan = isinstance(v, float) and math.isnan(v)
    w_nan = isinstance(w, float) and math.isnan(w)
    if v_nan or w_nan:
        return None if v_nan and w_nan else math.inf
    return None if v == w else abs(v - w)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    args = p.parse_args(argv)
    ref = {r["step"]: r for r in rows(args.ref)}
    new = rows(args.run)
    if not new:
        print(f"{args.run}: metrics.jsonl has no rows: nothing to compare")
        return 1
    missing = [r["step"] for r in new if r["step"] not in ref]
    if missing:
        print(f"steps {missing[:5]}{'...' if len(missing) > 5 else ''} are not in {args.ref}")
        return 1
    keys, one_sided, diffs, worst = set(), set(), 0, 0.0
    for r in new:
        o = ref[r["step"]]
        odd = set(r) ^ set(o)
        if any(not k.startswith(DIAGNOSTIC_PREFIXES) for k in odd):
            print(f"step {r['step']}: different keys {sorted(odd)}")
            return 1
        one_sided |= odd
        for k in r.keys() & o.keys():
            keys.add(k)
            d = diff(r[k], o[k])
            if d is not None:
                diffs += 1
                worst = max(worst, d)
    print(
        f"{len(new)} rows x {len(keys)} keys compared; differing values: {diffs}; "
        f"max abs diff {worst}; diagnostic keys on one side only: {len(one_sided)}"
    )
    return 0 if diffs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
