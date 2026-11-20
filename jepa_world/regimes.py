"""Scores by motion regime: is the model's error in the bounces, or in smooth rolling too?

  uv run python -m jepa_world.regimes --run <run dir> [--split test] [--checkpoint step_N.pt]

Every window of eval's set (t in [C-1, T-1-k_eval], the baselines' windows) gets one regime per
horizon k, from the episode files' contact and bounce flags (sim/episode.py):
  bounce    a bounce (a landing) in frames t+1 .. t+k
  rolling   no bounce in frames t-C+1 .. t+k, and contact in every one of them
  airborne  everything else: a flight, or a bounce in the context but none ahead
Per k, per regime and over all windows:
- the share of windows;
- the goal head's mean error (m) and MSE, and every baseline's, on the same windows;
- the goal head's ratio to each baseline: a ratio of mean errors, as eval reports (SPEC §7.2);
- joint coverage at eval.coverage_levels;
- latent relerr per head: the mean per-window squared error over the target spread of the FULL
  window set at that k. One denominator for every regime, so they are comparable, and their
  share-weighted mean is the full set's relerr.
Writes <run>/eval/<time>-regimes-<split>-<hex>.json. Design: jepa_world/DESIGN.md.
"""

from __future__ import annotations

import argparse
import json
import secrets
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from jepa_world import baselines, metrics
from jepa_world.data.dataset import EpisodeSet, load_split, window_frames
from jepa_world.eval import (
    BASE_CONFIG,
    HELD_OUT,
    check_episode_run,
    check_eval_block,
    check_eval_split,
    eval_windows,
    goal_positions,
    load_run,
    simulator_baselines,
    stored_episodes,
)
from jepa_world.models.jepa import JEPA
from jepa_world.sim.episode import simulate_from_state
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import write_new
from jepa_world.utils.repro import run_metadata

REGIMES = ("rolling", "airborne", "bounce")  # labels are indices into this
ROLLING, AIRBORNE, BOUNCE = range(len(REGIMES))
DEFINITIONS = {
    "rolling": "no bounce in frames t-C+1..t+k and contact in every one of them",
    "airborne": "everything else: a flight, or a bounce in the context but none in t+1..t+k",
    "bounce": "a bounce flag in frames t+1..t+k",
}
TABLE_BASELINES = ("persistence", "const_velocity", "simulator_mean")  # the printed ratios
SPREAD_EPS = 1e-12  # relerr's floor on the spread, as training's (models/jepa.py)


def regime_labels(contact, bounce, ep, t, k: int, context_frames: int) -> np.ndarray:
    """(M,) int8 index into REGIMES for windows (ep[i], t[i]) at horizon k. contact, bounce:
    (N, T) bool. Refuses a window whose frames t-C+1 .. t+k leave the episode."""
    contact, bounce = np.asarray(contact, dtype=bool), np.asarray(bounce, dtype=bool)
    ep, t = np.asarray(ep, dtype=int), np.asarray(t, dtype=int)
    if contact.shape != bounce.shape or contact.ndim != 2:
        raise ValueError(f"contact {contact.shape} and bounce {bounce.shape}: expected (N, T) each")
    if k < 1 or context_frames < 1:
        raise ValueError(f"k={k} and context_frames={context_frames} must be >= 1")
    first = t - context_frames + 1
    if len(t) and (first.min() < 0 or (t + k).max() > contact.shape[1] - 1):
        raise ValueError(
            f"windows reach outside the episode: frames {first.min()}..{(t + k).max()}, "
            f"T={contact.shape[1]}"
        )
    frames = first[:, None] + np.arange(context_frames + k)  # (M, C+k): t-C+1 .. t+k
    c, b = contact[ep[:, None], frames], bounce[ep[:, None], frames]
    ahead = b[:, context_frames:].any(axis=1)  # frames t+1 .. t+k
    rolling = c.all(axis=1) & ~b.any(axis=1)
    return np.where(ahead, BOUNCE, np.where(rolling, ROLLING, AIRBORNE)).astype(np.int8)


def episode_flags(cfg: dict, split: str, es: EpisodeSet) -> tuple[np.ndarray, np.ndarray]:
    """contact, bounce: (N, T) bool from the split's episode files, in `es`'s row order (each
    file checked against its row's positions). EpisodeSet doesn't carry them."""
    contact = np.zeros((es.n, es.T), dtype=bool)
    bounce = np.zeros((es.n, es.T), dtype=bool)
    for i, ep in enumerate(stored_episodes(cfg, split, es)):
        contact[i], bounce[i] = ep.contact, ep.bounce
    return contact, bounce


class _Moments:
    """Per-feature mean and variance (correction 0) over every row added, in float64 on the CPU:
    running sums of (x - shift) and its square, shifted by the first batch's mean."""

    def __init__(self) -> None:
        self.n, self.shift, self.s1, self.s2 = 0, None, None, None

    def add(self, z: torch.Tensor) -> None:
        z = z.detach().cpu().double()
        if self.shift is None:
            self.shift = z.mean(dim=0)
            self.s1, self.s2 = torch.zeros_like(self.shift), torch.zeros_like(self.shift)
        d = z - self.shift
        self.s1 += d.sum(dim=0)
        self.s2 += d.square().sum(dim=0)
        self.n += z.shape[0]

    def spread(self) -> float:
        """Σ_channels Var, averaged over positions: training relerr's denominator for (B, C, H, W)
        rows, with every row added as one batch."""
        mean = self.s1 / self.n
        var = (self.s2 / self.n - mean.square()).clamp_min(0.0)
        return float(var.sum(dim=0).mean())


def window_outputs(
    model, es: EpisodeSet, ep: np.ndarray, t: np.ndarray, batch_size: int, device: torch.device
) -> dict:
    """The JEPA core's forward (the latent heads and their targets, as training computes them)
    and the goal head on its context, over windows (ep, t). The pixel decoder isn't run: nothing
    here reads it.
      "sqerr":  {k: (M,) float64} per latent horizon, the head's mean against its target, squared,
                summed over channels and averaged over positions: training's sqerr/k per window
      "spread": {k: float}, the target spread over all M windows (_Moments.spread)
      "goal_mean", "goal_logvar": (M, K, 2) float64 (= eval.model_outputs), or None without a
                goal head
    mean(sqerr[k]) / spread[k] is training's relerr/k with the M windows as one batch (both
    tested against WorldModel.forward)."""
    c, k_max = model.context_frames, max(model.horizons)
    has_goal = getattr(model, "goal", None) is not None
    sq = {k: [] for k in model.horizons}
    moments = {k: _Moments() for k in model.horizons}
    means, logvars = [], []
    with torch.no_grad():
        for i in range(0, len(ep), batch_size):
            frames = window_frames(es, ep[i : i + batch_size], t[i : i + batch_size], c, k_max)
            out = JEPA.forward(model, frames.to(device))  # WorldModel's, minus goal and pixel
            for k in model.horizons:
                target = out["target"][k]
                err = (target - out["pred"][k]).square().sum(dim=1).mean(dim=(1, 2))
                sq[k].append(err.cpu().double().numpy())  # MPS has no float64: CPU first
                moments[k].add(target)
            if has_goal:  # on the context levels, as WorldModel.forward runs it
                mean, logvar = model.goal(*(out["context"][lv] for lv in model.goal_levels))
                means.append(mean.cpu().double().numpy())
                logvars.append(logvar.cpu().double().numpy())
    return {
        "sqerr": {k: np.concatenate(v) for k, v in sq.items()},
        "spread": {k: m.spread() for k, m in moments.items()},
        "goal_mean": np.concatenate(means) if has_goal else None,
        "goal_logvar": np.concatenate(logvars) if has_goal else None,
    }


def _row(mask, err: dict, goal_k, truth_k, sq_k, spread_k, names, levels) -> dict:
    n = int(mask.sum())
    row: dict = {"windows": n, "share": n / len(mask) if len(mask) else 0.0}
    row["error_m"] = {name: float(e[mask].mean()) if n else None for name, e in err.items()}
    row["mse_m2"] = {
        name: float(np.square(e[mask]).mean()) if n else None for name, e in err.items()
    }
    g = row["error_m"].get("goal_head")
    row["ratio"] = {
        name: g / row["error_m"][name] if g is not None and row["error_m"][name] else None
        for name in names
    }
    row["coverage"] = None
    if goal_k is not None and n:
        cov = metrics.coverage(goal_k["pred"][mask], goal_k["var"][mask], truth_k[mask], levels)
        row["coverage"] = {p: v["joint"] for p, v in cov.items()}
    row["relerr"] = (
        float(sq_k[mask].mean() / max(spread_k, SPREAD_EPS)) if sq_k is not None and n else None
    )
    return row


def _check_positions(what: str, a, m: int) -> None:
    a = np.asarray(a)
    if a.shape != (m, 2):
        raise ValueError(f"{what}: expected ({m}, 2), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{what} has non-finite values")


def score_regimes(
    labels: Mapping[int, np.ndarray],
    truth: Mapping[int, np.ndarray],
    baseline_preds: Mapping[str, Mapping[int, np.ndarray]],
    goal: Mapping[str, Mapping[int, np.ndarray]] | None,
    sqerr: Mapping[int, np.ndarray],
    spread: Mapping[int, float],
    levels: Iterable[float],
) -> dict:
    """{str(k): {"all" | regime: row}} for every k in `labels`, each row:
      windows, share      the regime's windows at k, and their share of all of them
      error_m, mse_m2     {"goal_head" (if scored at k), baseline: mean error (m), MSE (m²)}
      ratio               {baseline: goal_head mean error / the baseline's}, None if either is
                          missing or the baseline's is 0
      coverage            {level: joint coverage} of the goal head, None without one at k
      relerr              mean sqerr / spread[k] (the full set's), None without a latent head at k
    Every array is (M, ...) over the same windows. `goal` is {"pred", "var": {k: (M, 2)}}.
    A regime without windows at k has None everywhere but windows and share. Every prediction
    must be (M, 2) and finite (as baselines.evaluate requires of its `extra` ones)."""
    levels = list(levels)
    out = {}
    for k in sorted(labels):
        lab = np.asarray(labels[k])
        goal_k = (
            {"pred": goal["pred"][k], "var": goal["var"][k]}
            if goal is not None and k in goal["pred"]
            else None
        )
        named = [(f"truth at k={k}", truth[k])]
        named += [(f"{n} at k={k}", per_k[k]) for n, per_k in baseline_preds.items()]
        named += [(f"goal {n} at k={k}", goal_k[n]) for n in ("pred", "var")] if goal_k else []
        for what, a in named:
            _check_positions(what, a, len(lab))
        err = {}
        if goal_k is not None:
            err["goal_head"] = metrics.position_error(goal_k["pred"], truth[k])
        for name, per_k in baseline_preds.items():
            err[name] = metrics.position_error(per_k[k], truth[k])
        masks = [("all", np.ones(len(lab), dtype=bool))]
        masks += [(name, lab == i) for i, name in enumerate(REGIMES)]
        out[str(k)] = {
            name: _row(
                mask,
                err,
                goal_k,
                truth[k],
                sqerr.get(k),
                spread.get(k),
                list(baseline_preds),
                levels,
            )  # fmt: skip
            for name, mask in masks
        }
    return out


def evaluate_regimes(
    run_dir: str | Path,
    eval_cfg: dict,
    split: str = "test",
    checkpoint: str | Path | None = None,
    simulate=simulate_from_state,
) -> tuple[dict, Path]:
    """Score a run's goal head, the baselines and the latent heads per motion regime on held-out
    `split`, over eval's windows, and write <run>/eval/<time>-regimes-<split>-<hex>.json:
    {"meta": ..., "regimes": score_regimes(...)}.

    The same checks as `eval --run`, before the GPU: a held-out split (checked before anything is
    loaded), the eval: block, a run trained on the real episodes, no shared terrain seeds. Needs a
    goal head. The simulator baselines (eval.sim_params) run on the CPU outside the GPU lock;
    only the forward passes run under it, and only on MPS."""
    run_dir, ev = Path(run_dir), eval_cfg["eval"]
    if split not in HELD_OUT:
        raise ValueError(f"split {split!r}: regimes scores dev or test only (eval.HELD_OUT)")
    check_eval_block(ev)
    device = pick_device(ev["device"])
    run = load_run(run_dir, checkpoint, torch.device("cpu"))  # moved to the device under the lock
    cfg = run.cfg
    check_episode_run(cfg, run_dir)
    if getattr(run.model, "goal", None) is None:
        raise ValueError(f"{run_dir}: no goal head (train.stages lacks goal); this scores one")
    check_eval_split(cfg, eval_cfg, split)
    c = cfg["model"]["context_frames"]
    goal_ks = list(cfg["model"]["goal_horizons"])
    ks = sorted({*goal_ks, *run.model.horizons})
    es = load_split(cfg, split)
    ep, t = eval_windows(cfg, es, has_goal=True)  # eval --run's windows, in the same order
    starts = np.unique(t)
    contact, bounce = episode_flags(cfg, split, es)
    labels = {k: regime_labels(contact, bounce, ep, t, k, c) for k in ks}
    preds = {
        name: {
            k: baselines.predict(name, es.ball_pos, es.ball_vel, ep, t, k, 1 / es.fps) for k in ks
        }
        for name in baselines.applicable(c)
    }
    sim_info = {}
    if ev["sim_params"]:
        extra, sim_info = simulator_baselines(
            cfg, split, es, starts, ks, ev["sim_params"], simulate
        )
        preds.update(extra)
    with gpu_lock(f"regimes {run_dir.name}", enabled=device.type == "mps"):
        run.model.to(device)
        out = window_outputs(run.model, es, ep, t, ev["batch_size"], device)
        run.model.to("cpu")
    g = goal_positions(es.ball_pos, ep, t, out["goal_mean"], out["goal_logvar"], goal_ks)
    truth = {k: es.ball_pos[ep, t + k] for k in ks}
    scores = score_regimes(
        labels, truth, preds, {"pred": g["pred"], "var": g["var"]}, out["sqerr"], out["spread"],
        ev["coverage_levels"],
    )  # fmt: skip
    meta = {
        **run_metadata(),
        "run_dir": str(run_dir),
        "checkpoint": str(run.checkpoint),
        "step": run.step,
        "split": split,
        "windows": int(len(ep)),
        "starts": [int(starts[0]), int(starts[-1])],
        "context_frames": int(c),
        "goal_horizons": goal_ks,
        "latent_horizons": list(run.model.horizons),
        "regimes": dict(DEFINITIONS),
        "latent_spread": {str(k): v for k, v in out["spread"].items()},
        "device": str(device),
        "eval": ev,
    }
    if sim_info:
        meta["simulator"] = sim_info
    doc = {"meta": meta, "regimes": scores}
    text = json.dumps(doc, indent=2, allow_nan=False)  # fails before any file exists
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = write_new(out_dir / f"{stamp}-regimes-{split}-{secrets.token_hex(3)}.json", text)
    return json.loads(text), path


def _cell(x, fmt: str, width: int) -> str:
    return ("n/a" if x is None else format(x, fmt)).rjust(width)


def format_regime_table(doc: dict) -> str:
    """Per k: one line per regime (and all windows) with its share, the goal head's mean error,
    its ratio to persistence, constant velocity and the mean-params simulator (where scored),
    joint coverage, and latent relerr."""
    lines = []
    for k, rows in doc["regimes"].items():
        names = [n for n in TABLE_BASELINES if n in rows["all"]["ratio"]]
        levels = list(rows["all"]["coverage"] or {})
        lines.append(f"\nk={k}  ({rows['all']['windows']} windows)")
        lines.append(
            f"  {'regime':9} {'share':>6} {'goal m':>8}"
            + "".join(f" {'/' + n:>16}" for n in names)
            + "".join(f" {'cov' + p:>7}" for p in levels)
            + f" {'relerr':>7}"
        )
        for name in ("all", *REGIMES):
            r = rows[name]
            cov = r["coverage"] or {}
            lines.append(
                f"  {name:9} {r['share']:6.1%} {_cell(r['error_m'].get('goal_head'), '.4f', 8)}"
                + "".join(f" {_cell(r['ratio'][n], '.3f', 16)}" for n in names)
                + "".join(f" {_cell(cov.get(p), '.3f', 7)}" for p in levels)
                + f" {_cell(r['relerr'], '.3f', 7)}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="run directory to score")
    p.add_argument("--split", default="test", choices=list(HELD_OUT))
    p.add_argument("--checkpoint", help="a checkpoint file (default: the run's latest)")
    p.add_argument("--config", help="the eval config, default configs/base.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    eval_cfg = load_config(args.config or BASE_CONFIG, args.set)
    doc, path = evaluate_regimes(args.run, eval_cfg, args.split, checkpoint=args.checkpoint)
    meta = doc["meta"]
    print(f"regimes: {path}\ncheckpoint {meta['checkpoint']} (step {meta['step']})")
    print(f"{meta['split']}: {meta['windows']} windows (t in {meta['starts']})")
    print(format_regime_table(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
