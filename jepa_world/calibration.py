"""Post-hoc calibration of the goal head: one variance scale per horizon (a proposal, not an eval
default).

  uv run python -m jepa_world.calibration --run <run dir> [--level 0.9] [--folds 5]

The goal head's 90% joint regions hold 83-88% of test truths at k >= 5 (experiments/
2026-11-19-headline-48k-new-defaults.md). Is that one scale per horizon? For each goal horizon k
this fits s_k, which multiplies the predicted variance (the std by sqrt(s_k); the mean is
unchanged), so that the `level` joint region holds `level` of the fit set's truths:

    s_k = quantile_level(r2) / chi2_2(level),   r2 = sum over x, z of (y - mu)^2 / var

It reports, per k, 50% and 90% joint coverage and NLL on test:
- before: the model as trained, which is eval --run's numbers;
- dev-fit: s_k fit on dev (eval's windows) and applied to test;
- cross-fit: as a check, s_k fit on the other folds of test terrain seeds, for each fold.
It also gives coverage per motion regime (regimes.py), and the NLL-optimal scale mean(r2)/2 as a
diagnostic. When that scale is far from s_k, no single scale fixes both the coverage and the NLL.

Dev is SPEC §3.5's debugging split, so the dev fit is reported here and not made a default; that is
the user's call. Writes <run>/eval/calibration/<time>-calibration-<hex>.json, outside eval/*.json,
which readers take to be eval --run reports. Design: jepa_world/DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from jepa_world import regimes
from jepa_world.data.dataset import load_split
from jepa_world.data.splits import terrain_seeds
from jepa_world.eval import (
    BASE_CONFIG,
    check_episode_run,
    check_eval_block,
    check_eval_split,
    eval_windows,
    goal_positions,
    load_run,
    model_outputs,
)
from jepa_world.metrics import coverage
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import write_new
from jepa_world.utils.repro import run_metadata

LOG_2PI = math.log(2 * math.pi)
FIT_SPLIT, SCORED_SPLIT = "dev", "test"


def chi2_2(p: float) -> float:
    """The p-quantile of chi-square with 2 degrees of freedom: the joint region metrics.coverage
    counts is r2 <= chi2_2(p)."""
    return -2.0 * math.log(1.0 - p)


def standardized_r2(mean, var, truth) -> np.ndarray:
    """(M,) sum over the two axes of (truth - mean)^2 / var, for (M, 2) arrays."""
    mean, var, truth = (np.asarray(a, dtype=np.float64) for a in (mean, var, truth))
    return ((truth - mean) ** 2 / var).sum(axis=-1)


def fit_scale(r2, level: float) -> float:
    """The variance scale s that puts `level` of the r2 inside the joint `level` region:
    quantile(r2, level) / chi2_2(level). 1 for a calibrated model; > 1 widens the regions."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level={level}: expected 0 < level < 1")
    r2 = np.asarray(r2, dtype=np.float64)
    if r2.size == 0 or not np.isfinite(r2).all():
        raise ValueError("fit_scale needs at least one finite r2")
    s = float(np.quantile(r2, level) / chi2_2(level))
    if not (s > 0.0 and math.isfinite(s)):
        raise ValueError(
            f"fitted scale {s}: at least {level:g} of the fit set's r2 are 0 (the predicted mean "
            "equals the truth), so no positive variance scale fits"
        )
    return s


def ml_scale(r2) -> float:
    """The variance scale that minimises the mean 2-D Gaussian NLL: mean(r2) / 2."""
    return float(np.mean(r2) / 2.0)


def nll(mean, var, truth) -> np.ndarray:
    """(M,) diagonal-Gaussian NLL per window, summed over the two axes: the formula of training's
    goal_nll (models/goal.py), in nats."""
    mean, var, truth = (np.asarray(a, dtype=np.float64) for a in (mean, var, truth))
    return 0.5 * ((truth - mean) ** 2 / var + np.log(var) + LOG_2PI).sum(axis=-1)


def terrain_folds(seeds, n_folds: int) -> np.ndarray:
    """(M,) fold per window: the sorted distinct terrain seeds go round-robin into n_folds, so a
    terrain's windows are always in one fold, and the folds depend only on the seeds."""
    seeds = np.asarray(seeds)
    uniq = np.unique(seeds)
    if not 2 <= n_folds <= len(uniq):
        raise ValueError(f"n_folds={n_folds}: expected 2 .. {len(uniq)} (the terrain seeds scored)")
    return (np.searchsorted(uniq, seeds) % n_folds).astype(np.int64)


def cross_fit_scales(r2, folds, level: float) -> np.ndarray:
    """(M,) per window: fit_scale on the windows of every other fold."""
    r2, folds = np.asarray(r2, dtype=np.float64), np.asarray(folds)
    out = np.empty_like(r2)
    for f in np.unique(folds):
        out[folds == f] = fit_scale(r2[folds != f], level)
    return out


def _score(mean, var, truth, scale, levels) -> dict:
    """Joint coverage at `levels` and mean NLL with the variance times `scale` (a number, or one per
    window)."""
    s = np.asarray(scale, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64) * (s[:, None] if s.ndim else s)
    cov = coverage(mean, var, truth, levels)
    return {
        "coverage": {p: v["joint"] for p, v in cov.items()},
        "nll": float(nll(mean, var, truth).mean()),
    }


def calibrate(
    dev: Mapping[int, tuple],
    test: Mapping[int, tuple],
    test_folds,
    test_labels: Mapping[int, np.ndarray],
    levels: Iterable[float],
    level: float,
) -> dict:
    """{str(k): {"dev_fit", "test", "by_regime"}} for every k in `test`. dev and test map k to
    (mean, var, truth), each (M, 2) over that split's windows; test_folds (M,) are the test
    windows' terrain folds (terrain_folds), test_labels {k: (M,)} their regimes
    (regimes.regime_labels). Coverage is scored at `levels` and always at `level`.

    The cross-fit is a CHECK of the dev fit, fit on test's other terrain folds: never a held-out
    result, and its scales are not for adoption."""
    levels = sorted({*(float(p) for p in levels), float(level)})
    folds = np.asarray(test_folds)
    n_folds = len(np.unique(folds))
    out = {}
    for k in sorted(test):
        mean, var, truth = test[k]
        r2_dev = standardized_r2(*dev[k])
        s = fit_scale(r2_dev, level)
        r2 = standardized_r2(mean, var, truth)
        cf = cross_fit_scales(r2, folds, level)
        rows = {
            "before": _score(mean, var, truth, 1.0, levels),
            "dev_scaled": _score(mean, var, truth, s, levels),
            "cross_fit": {
                **_score(mean, var, truth, cf, levels),
                "n_folds": n_folds,
                "fold_scales": [float(cf[folds == f][0]) for f in np.unique(folds)],
            },
        }
        by_regime = {}
        for i, name in enumerate(regimes.REGIMES):
            mask = np.asarray(test_labels[k]) == i
            n = int(mask.sum())
            sub = (mean[mask], var[mask], truth[mask])
            by_regime[name] = {
                "windows": n,
                "before": _score(*sub, 1.0, levels) if n else None,
                "dev_scaled": _score(*sub, s, levels) if n else None,
                "cross_fit": _score(*sub, cf[mask], levels) if n else None,
            }
        out[str(k)] = {
            "dev_fit": {
                "scale": s,
                "std_factor": math.sqrt(s),
                "ml_scale": ml_scale(r2_dev),
                "dev_before": _score(*dev[k], 1.0, levels),
                "dev_after": _score(*dev[k], s, levels),
            },
            "test": rows,
            "by_regime": by_regime,
        }
    return out


def calibrate_run(
    run_dir: str | Path,
    eval_cfg: dict,
    checkpoint: str | Path | None = None,
    level: float = 0.9,
    n_folds: int = 5,
) -> tuple[dict, Path]:
    """Fit the per-k variance scales on dev, score them on test (and cross-fit over test terrain
    folds), and write <run>/eval/calibration/<time>-calibration-<hex>.json: {"meta", "horizons":
    calibrate(...)}. The same checks as `eval --run` for both splits before the GPU; only the
    forward passes run under the GPU lock, and only on MPS."""
    run_dir, ev = Path(run_dir), eval_cfg["eval"]
    check_eval_block(ev)  # these checks come before any model, data or lock
    if not 0.0 < level < 1.0:
        raise ValueError(f"level={level}: expected 0 < level < 1")
    n_terrains = len(terrain_seeds(eval_cfg["data"], SCORED_SPLIT))
    if not 2 <= n_folds <= n_terrains:
        raise ValueError(
            f"n_folds={n_folds}: expected 2 .. {n_terrains} ({SCORED_SPLIT} terrain seeds)"
        )
    device = pick_device(ev["device"])
    run = load_run(run_dir, checkpoint, torch.device("cpu"))  # moved to the device under the lock
    cfg = run.cfg
    check_episode_run(cfg, run_dir)
    if getattr(run.model, "goal", None) is None:
        raise ValueError(f"{run_dir}: no goal head (train.stages lacks goal); this calibrates one")
    for split in (FIT_SPLIT, SCORED_SPLIT):
        check_eval_split(cfg, eval_cfg, split)
    goal_ks = list(cfg["model"]["goal_horizons"])
    data = {s: load_split(cfg, s) for s in (FIT_SPLIT, SCORED_SPLIT)}
    windows = {s: eval_windows(cfg, es, has_goal=True) for s, es in data.items()}
    with gpu_lock(f"calibration {run_dir.name}", enabled=device.type == "mps"):
        run.model.to(device)
        outs = {
            s: model_outputs(
                run.model, data[s], *windows[s], ev["batch_size"], device, features=False
            )
            for s in data
        }
        run.model.to("cpu")
    per_split = {}
    for s, es in data.items():
        g = goal_positions(
            es.ball_pos, *windows[s], outs[s]["goal_mean"], outs[s]["goal_logvar"], goal_ks
        )
        per_split[s] = {k: (g["pred"][k], g["var"][k], g["truth"][k]) for k in goal_ks}
    es, (ep, t) = data[SCORED_SPLIT], windows[SCORED_SPLIT]
    c = cfg["model"]["context_frames"]
    contact, bounce = regimes.episode_flags(cfg, SCORED_SPLIT, es)
    labels = {k: regimes.regime_labels(contact, bounce, ep, t, k, c) for k in goal_ks}
    seeds = es.terrain_seed[ep]
    folds = terrain_folds(seeds, n_folds)  # the one fold assignment: scored and recorded
    horizons = calibrate(
        per_split[FIT_SPLIT], per_split[SCORED_SPLIT], folds, labels, ev["coverage_levels"], level
    )
    meta = {
        **run_metadata(),
        "run_dir": str(run_dir),
        "checkpoint": str(run.checkpoint),
        "step": run.step,
        "fit_split": FIT_SPLIT,
        "scored_split": SCORED_SPLIT,
        "windows": {s: int(len(w[0])) for s, w in windows.items()},
        "starts": {s: [int(w[1].min()), int(w[1].max())] for s, w in windows.items()},
        "level": level,
        "n_folds": n_folds,
        "fold_terrain_seeds": [
            sorted(int(x) for x in np.unique(seeds[folds == f])) for f in range(n_folds)
        ],
        "device": str(device),
        "eval": ev,
        "note": "scale multiplies the goal head's predicted variance (std by sqrt(scale)); the mean "
        "is unchanged. Dev is SPEC §3.5's debugging split: a proposal, not an eval default.",
        "cross_fit_note": "cross_fit scales are fit on test's other terrain folds: a check of the "
        "dev fit, not a held-out result, and not for adoption.",
    }
    doc = {"meta": meta, "horizons": horizons}
    text = json.dumps(doc, indent=2, allow_nan=False)  # fails before any file exists
    out_dir = run_dir / "eval" / "calibration"  # not eval/*.json: readers take those to be reports
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = write_new(out_dir / f"{stamp}-calibration-{secrets.token_hex(3)}.json", text)
    return json.loads(text), path


def _f(x, fmt: str) -> str:
    return "n/a" if x is None else format(x, fmt)


def format_table(doc: dict) -> str:
    """Per k: the dev-fit scale, then test coverage and NLL before, with the dev-fit scale, and
    cross-fit; then 90% coverage per regime before and with the dev-fit scale."""
    lines = []
    key = f"{float(doc['meta']['level']):g}"  # the fitted level, as metrics keys it
    for k, r in doc["horizons"].items():
        fit = r["dev_fit"]
        levels = list(r["test"]["before"]["coverage"])
        lines.append(
            f"\nk={k}  dev-fit scale {fit['scale']:.3f} (std x{fit['std_factor']:.3f}), "
            f"NLL-optimal scale {fit['ml_scale']:.3f}; cross-fit fold scales "
            + " ".join(f"{s:.3f}" for s in r["test"]["cross_fit"]["fold_scales"])
        )
        lines.append("  test       " + "".join(f"{'cov' + p:>9}" for p in levels) + f"{'nll':>9}")
        for name, key in (
            ("before", "before"),
            ("dev-fit", "dev_scaled"),
            ("cross-fit*", "cross_fit"),
        ):
            row = r["test"][key]
            lines.append(
                f"  {name:10} "
                + "".join(f"{row['coverage'][p]:9.3f}" for p in levels)
                + f"{row['nll']:9.3f}"
            )
        cells = []
        for name, row in r["by_regime"].items():
            if row["windows"]:
                cells.append(
                    f"{name} {_f(row['before']['coverage'].get(key), '.3f')} -> "
                    f"{_f(row['dev_scaled']['coverage'].get(key), '.3f')}"
                )
        lines.append(f"  cov{key} by regime, before -> dev-fit: " + ", ".join(cells))
    lines.append(
        "\n* cross-fit: scales fit on test's other terrain folds, a check of the dev fit only"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run directory to calibrate")
    p.add_argument("--checkpoint", help="a checkpoint file (default: the run's latest)")
    p.add_argument("--level", type=float, default=0.9, help="the joint region the scale is fit to")
    p.add_argument("--folds", type=int, default=5, help="test terrain-seed folds for the cross-fit")
    p.add_argument("--config", help="the eval config, default configs/base.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    eval_cfg = load_config(args.config or BASE_CONFIG, args.set)
    doc, path = calibrate_run(args.run, eval_cfg, args.checkpoint, args.level, args.folds)
    meta = doc["meta"]
    print(f"calibration: {path}\ncheckpoint {meta['checkpoint']} (step {meta['step']})")
    print(
        f"fit on {meta['fit_split']} ({meta['windows']['dev']} windows), scored on "
        f"{meta['scored_split']} ({meta['windows']['test']} windows), level {meta['level']}"
    )
    print(format_table(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
