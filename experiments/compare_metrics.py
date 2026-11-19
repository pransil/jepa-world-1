"""Does a re-run repeat a recorded run? Compare two metrics.jsonl files, bitwise.

  uv run python experiments/compare_metrics.py --run <re-run dir> --ref <original dir>

Every step of --run is matched to the same step of --ref (which may be longer, e.g. a short re-run of
an 8000-step run). Prints the rows and keys compared, the number of differing values, and the largest
absolute difference. NaN equals NaN. Exit status 0 only if every value is identical.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def rows(run: Path) -> list[dict]:
    return [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    args = p.parse_args(argv)
    ref = {r["step"]: r for r in rows(args.ref)}
    new = rows(args.run)
    keys, diffs, worst = set(), 0, 0.0
    for r in new:
        o = ref[r["step"]]
        if r.keys() != o.keys():
            raise SystemExit(f"step {r['step']}: different keys {sorted(set(r) ^ set(o))}")
        for k, v in r.items():
            keys.add(k)
            w = o[k]
            nan = isinstance(v, float) and isinstance(w, float) and math.isnan(v) and math.isnan(w)
            if v != w and not nan:
                diffs += 1
                worst = max(worst, abs(v - w))
    print(
        f"{len(new)} rows x {len(keys)} keys compared; differing values: {diffs}; "
        f"max abs diff {worst}"
    )
    return 0 if diffs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
