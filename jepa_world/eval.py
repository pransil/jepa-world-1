"""Evaluation: baselines, metrics, probes, calibration. Baselines in M2, everything else in M6 (SPEC §7).

  uv run python -m jepa_world.eval --baselines --config configs/base.yaml --split test   (M2)
  uv run python -m jepa_world.eval --run <run dir> --split test                          (M6)

Headline numbers come from the held-out test terrain seeds only. Dev-set numbers are reported
separately and never used to claim generalisation (SPEC §7.3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from jepa_world import baselines
from jepa_world.data.build import split_dir
from jepa_world.data.dataset import load_split, valid_starts
from jepa_world.utils.config import load_config
from jepa_world.utils.paths import new_run_dir
from jepa_world.utils.repro import run_metadata


def run_baselines(cfg: dict, split: str) -> tuple[Path, dict]:
    """Score every position baseline on `split`, over the windows the model is scored on
    (C = model.context_frames, k_max = max(model.horizons)). Writes a fresh run dir."""
    es = load_split(cfg, split)
    horizons = list(cfg["model"]["horizons"])
    starts = valid_starts(es.T, cfg["model"]["context_frames"], max(horizons))
    scores = baselines.evaluate(es.ball_pos, es.ball_vel, 1 / es.fps, starts, horizons)
    result = {"split": split, **scores}
    run = new_run_dir(f"baselines-{split}")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {**run_metadata(), "dataset": str(split_dir(cfg, split))}
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    (run / "baselines.json").write_text(json.dumps(result, indent=2))
    return run, result


def format_table(result: dict) -> str:
    names = baselines.BASELINES
    head = (
        f"{'k':>4} "
        + " ".join(f"{n:>20}" for n in names)
        + "   (mean error m, ratio to persistence)"
    )
    lines = [head]
    for k in result["horizons"]:
        cells = []
        for n in names:
            mean = result["error_m"][n][str(k)]["mean"]
            r = result["ratio_to_persistence"][n][str(k)]
            cells.append(f"{mean:9.4f} ({'  n/a' if r is None else f'{r:5.3f}'})".rjust(20))
        lines.append(f"{k:>4} " + " ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baselines", action="store_true", help="position baselines (M2)")
    mode.add_argument("--run", help="run directory to evaluate (M6)")
    p.add_argument("--config", help="config for --baselines")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    if args.run:
        raise SystemExit(
            "evaluating a model run is not implemented yet: metrics land in M6 (SPEC §8)"
        )
    if not args.config:
        p.error("--baselines needs --config")
    cfg = load_config(args.config, args.set)
    run, result = run_baselines(cfg, args.split)
    print(
        f"run dir: {run}\nsplit {args.split}: {result['episodes']} episodes, "
        f"{result['windows']} windows (t in {result['starts']})"
    )
    print(format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
