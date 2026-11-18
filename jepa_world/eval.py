"""Evaluation: baselines, metrics, probes, calibration. Baselines in M2, everything else in M6 (SPEC §7).

  uv run python -m jepa_world.eval --baselines --config configs/base.yaml --split test   (M2)
      add --sim-params true mean for the simulator-aware baseline                      (M6)
  uv run python -m jepa_world.eval --run <run dir> --split test                          (M6)

Headline numbers come from the held-out test terrain seeds only. Dev-set numbers are reported
separately and never used to claim generalisation (SPEC §7.3).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import yaml

from jepa_world import baselines
from jepa_world.data.build import split_dir
from jepa_world.data.dataset import EpisodeSet, load_split, read_manifest, valid_starts
from jepa_world.sim.episode import simulate_from_state
from jepa_world.sim.io import load_episode
from jepa_world.utils.config import load_config
from jepa_world.utils.paths import new_run_dir
from jepa_world.utils.repro import run_metadata


@dataclasses.dataclass(frozen=True, eq=False)
class _Physics:
    """What the simulator baseline needs from an episode file, beyond EpisodeSet's arrays."""

    sim_cfg: dict
    terrain: object
    ball: object
    spin: np.ndarray  # (T,)
    cam_pos: np.ndarray  # (T, 2)


def _episode_physics(cfg: dict, split: str, es: EpisodeSet) -> list[_Physics]:
    """Each episode's terrain, true ball, spin and stored sim config, in `es`'s row order (the
    manifest's). Frames are read and dropped: EpisodeSet doesn't carry these fields."""
    path = split_dir(cfg, split)
    rows = read_manifest(path)["episodes"]
    if len(rows) != es.n:
        raise ValueError(f"{path}: manifest lists {len(rows)} episodes, the split holds {es.n}")
    out = []
    for i, row in enumerate(rows):
        ep = load_episode(path / row["file"])
        if not np.array_equal(ep.ball_pos, es.ball_pos[i]):
            raise ValueError(
                f"{path / row['file']}: positions differ from the loaded split's row {i}"
            )
        out.append(_Physics(ep.meta["sim_cfg"], ep.terrain, ep.ball, ep.ball_spin, ep.cam_pos))
    return out


HIDDEN = ("radius", "friction", "restitution")  # the only BallParams simulate_from_state reads


def _params_record(balls: Sequence) -> dict:
    """What the rollouts actually used, per hidden parameter: the value if every episode used the
    same one, else "per-episode" (the true values, which vary)."""
    out = {}
    for name in HIDDEN:
        values = {float(getattr(b, name)) for b in balls}
        out[name] = values.pop() if len(values) == 1 else "per-episode"
    return out


def _quantiles(a: np.ndarray) -> dict:
    q = np.quantile(a, [0.0, 0.5, 1.0])
    return {"min": float(q[0]), "median": float(q[1]), "max": float(q[2])}


def simulator_baselines(
    cfg: dict,
    split: str,
    es: EpisodeSet,
    starts: np.ndarray,
    horizons: Sequence[int],
    sim_params: Sequence[str],
    simulate: Callable = simulate_from_state,
) -> tuple[dict, dict]:
    """Predictions {simulator_<params>: {k: (M, 2)}} in `baselines.window_index` order, and a
    record per params:
    - the parameters used;
    - how many starts the true centre would have put inside the terrain (`inside_as_given`,
      against `inside_true` for the true ball);
    - how many were shifted (`match_clearance`), by how far, and how many didn't converge;
    - the time taken.

    One rollout per window, on the CPU. Repeated params run once."""
    physics = _episode_physics(cfg, split, es)
    extra, info = {}, {}
    for params in dict.fromkeys(sim_params):
        t0 = time.perf_counter()
        per_k = {k: [] for k in horizons}
        recs, balls = [], []
        for i, ph in enumerate(physics):
            ball = baselines.sim_ball(params, ph.ball, ph.sim_cfg)
            preds, rec = baselines.simulator_predict(
                ph.sim_cfg, ph.terrain, ball, ph.ball.radius, es.ball_pos[i], es.ball_vel[i],
                ph.spin, starts, horizons, cam_pos=ph.cam_pos, simulate=simulate,
            )  # fmt: skip
            for k in horizons:
                per_k[k].append(preds[k])
            recs.append(rec)
            balls.append(ball)
            if (i + 1) % 20 == 0:
                print(f"[simulator {params}] {i + 1}/{len(physics)} episodes", file=sys.stderr)
        r = {key: np.concatenate([rec[key] for rec in recs]) for key in recs[0]}
        extra[baselines.simulator_name(params)] = {k: np.concatenate(v) for k, v in per_k.items()}
        info[params] = {
            "params": _params_record(balls),
            "windows": int(len(r["shift"])),
            "inside_true": int((r["gap_true"] < 0).sum()),
            "inside_as_given": int((r["gap_as_given"] < 0).sum()),
            "gap_as_given_m": _quantiles(r["gap_as_given"]),
            "shifted": int(np.count_nonzero(np.abs(r["shift"]).sum(axis=1))),
            "abs_shift_m": _quantiles(np.linalg.norm(r["shift"], axis=1)),
            "unmatched": int(r["unmatched"].sum()),
            "clearance_miss_m": _quantiles(r["clearance_miss"]),
            "seconds": round(time.perf_counter() - t0, 1),
        }
    return extra, info


def run_baselines(
    cfg: dict,
    split: str,
    sim_params: Sequence[str] = (),
    simulate: Callable = simulate_from_state,
) -> tuple[Path, dict]:
    """Score every position baseline on `split`, at every model.horizons k, over one shared window
    set t in [C-1, T-1-max(model.horizons)]. M6 must score the model on the same windows for its
    ratios to persistence to be like-for-like (data/DESIGN.md). Writes a fresh run dir.

    `sim_params` picks the simulator baselines (baselines.SIM_PARAMS); empty runs none.
    Loads the frames too, which go unused here. Eval reads only dev or test (at most 200
    episodes, about 150 MB), and a positions-only reader would duplicate sim/io's format."""
    es = load_split(cfg, split)
    horizons = list(cfg["model"]["horizons"])
    c = cfg["model"]["context_frames"]
    starts = valid_starts(es.T, c, max(horizons))
    extra, sim_info = {}, {}
    if sim_params:
        extra, sim_info = simulator_baselines(
            cfg, split, es, starts, horizons, sim_params, simulate
        )
    scores = baselines.evaluate(
        es.ball_pos, es.ball_vel, 1 / es.fps, starts, horizons, c, es.out_of_view, extra=extra
    )
    result = {"split": split, **scores}
    if sim_info:
        result["simulator"] = sim_info
    run = new_run_dir(f"baselines-{split}")
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    meta = {**run_metadata(), "dataset": str(split_dir(cfg, split))}
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    (run / "baselines.json").write_text(json.dumps(result, indent=2))
    return run, result


def format_table(result: dict) -> str:
    names = list(result["error_m"])
    width = {n: max(20, len(n)) for n in names}
    head = (
        f"{'k':>4} "
        + " ".join(n.rjust(width[n]) for n in names)
        + "   (mean error m, ratio to persistence)"
    )
    lines = [head]
    for k in result["horizons"]:
        cells = []
        for n in names:
            mean = result["error_m"][n][str(k)]["mean"]
            r = result["ratio_to_persistence"][n][str(k)]
            cells.append(f"{mean:9.4f} ({'  n/a' if r is None else f'{r:5.3f}'})".rjust(width[n]))
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
    p.add_argument(
        "--sim-params",
        nargs="*",
        choices=baselines.SIM_PARAMS,
        default=[],
        help=(
            "also run the simulator-aware baseline with these ball parameters (minutes of CPU "
            f"each); the proposed set is {' '.join(baselines.PROPOSED_SIM_PARAMS)}"
        ),
    )
    args = p.parse_args(argv)

    if args.run:
        raise SystemExit(
            "evaluating a model run is not implemented yet: metrics land in M6 (SPEC §8)"
        )
    if not args.config:
        p.error("--baselines needs --config")
    cfg = load_config(args.config, args.set)
    # No GPU lock: the simulator runs MuJoCo on the CPU with render=False (no MPS, no GL), and
    # gpu_lock is for MPS jobs. Taking it would hold up GPU training for the whole eval.
    run, result = run_baselines(cfg, args.split, sim_params=args.sim_params)
    print(
        f"run dir: {run}\nsplit {args.split}: {result['episodes']} episodes, "
        f"{result['windows']} windows (t in {result['starts']})"
    )
    print(format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
