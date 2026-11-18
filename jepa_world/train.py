"""Training entry point. Built in M3 (near heads), extended in M4 and M5. See SPEC §5-§6.

Contract every version of this script keeps (the /experiment skill relies on it):
  uv run python -m jepa_world.train --config configs/<x>.yaml [--set key=value ...] [--name NAME]
  - runs under utils.locks.gpu_lock() on MPS
  - writes to utils.paths.new_run_dir(): config.yaml, meta.json (utils.repro.run_metadata),
    metrics.jsonl (every loss term, every step), checkpoints/, tensorboard/
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--name", default=None)
    p.parse_args(argv)
    raise SystemExit("train.py is not implemented yet: it is built in milestone M3 (SPEC §8)")


if __name__ == "__main__":
    main()
