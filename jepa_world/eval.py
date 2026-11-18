"""Evaluation: baselines, metrics, probes, calibration. Baselines in M2, everything else in M6 (SPEC §7).

Headline numbers come from the held-out test terrain seeds only. Dev-set numbers are reported
separately and never used to claim generalisation (SPEC §7.3).
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run directory to evaluate")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.parse_args(argv)
    raise SystemExit(
        "eval.py is not implemented yet: baselines land in M2, metrics in M6 (SPEC §8)"
    )


if __name__ == "__main__":
    main()
