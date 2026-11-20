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
import secrets
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from jepa_world import baselines, metrics, probes, train
from jepa_world.data.build import split_dir
from jepa_world.data.dataset import (
    EpisodeSet,
    load_split,
    read_manifest,
    valid_starts,
    window_frames,
)
from jepa_world.data.splits import LeakageError, ball_seeds, check_split_config, terrain_seeds
from jepa_world.models.jepa import context_stack
from jepa_world.sim.ball import sample_ball
from jepa_world.sim.episode import simulate_from_state
from jepa_world.sim.io import load_episode
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import new_run_dir, write_new
from jepa_world.utils.repro import run_metadata


@dataclasses.dataclass(frozen=True, eq=False)
class _Physics:
    """What the simulator baseline needs from an episode file, beyond EpisodeSet's arrays."""

    sim_cfg: dict
    terrain: object
    ball: object
    spin: np.ndarray  # (T,)
    cam_pos: np.ndarray  # (T, 2)


def stored_episodes(cfg: dict, split: str, es: EpisodeSet) -> Iterator:
    """The split's episode files (`sim.episode.Episode`), one at a time in `es`'s row order (the
    manifest's), each checked against its row's positions. For the fields EpisodeSet doesn't
    carry; frames are read and dropped by the caller."""
    path = split_dir(cfg, split)
    rows = read_manifest(path)["episodes"]
    if len(rows) != es.n:
        raise ValueError(f"{path}: manifest lists {len(rows)} episodes, the split holds {es.n}")
    for i, row in enumerate(rows):
        ep = load_episode(path / row["file"])
        if not np.array_equal(ep.ball_pos, es.ball_pos[i]):
            raise ValueError(
                f"{path / row['file']}: positions differ from the loaded split's row {i}"
            )
        yield ep


def _episode_physics(cfg: dict, split: str, es: EpisodeSet) -> list[_Physics]:
    """Each episode's terrain, true ball, spin and stored sim config, in `es`'s row order."""
    return [
        _Physics(ep.meta["sim_cfg"], ep.terrain, ep.ball, ep.ball_spin, ep.cam_pos)
        for ep in stored_episodes(cfg, split, es)
    ]


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


# --- the model eval command (--run) -------------------------------------------------------------

BASE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
HELD_OUT = ("dev", "test")  # the splits --run scores; never the one the run trained on


def _step_of(path: Path) -> int:
    try:
        return int(path.stem.removeprefix("step_"))
    except ValueError:
        return -1  # not a step checkpoint: never the latest


def latest_checkpoint(run_dir: Path) -> Path:
    """The run's highest-step checkpoints/step_<n>.pt, by number, not by name."""
    ckpts = [p for p in (Path(run_dir) / "checkpoints").glob("step_*.pt") if _step_of(p) >= 0]
    if not ckpts:
        raise FileNotFoundError(f"{run_dir}: no checkpoints/step_*.pt checkpoint to evaluate")
    return max(ckpts, key=_step_of)


@dataclasses.dataclass(frozen=True, eq=False)
class LoadedRun:
    run_dir: Path
    checkpoint: Path
    step: int
    cfg: dict  # the checkpoint's own config: it built the model and names the data
    model: torch.nn.Module


def load_run(
    run_dir: str | Path, checkpoint: str | Path | None = None, device: torch.device | None = None
) -> LoadedRun:
    """A run's model from its latest checkpoint (or `checkpoint`, which must be one of the run's
    own: a path, or a bare file name in the run's checkpoints/), built from the config saved in
    that checkpoint and loaded strictly, in eval mode on `device`."""
    run_dir = Path(run_dir)
    path = Path(checkpoint) if checkpoint else latest_checkpoint(run_dir)
    if checkpoint and not path.exists() and path.name == str(checkpoint):
        path = run_dir / "checkpoints" / path.name
    if path.resolve().parent != (run_dir / "checkpoints").resolve():
        raise ValueError(
            f"checkpoint {path} isn't in {run_dir / 'checkpoints'}: eval of one run's model is filed under that run"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = train.build_model(cfg)
    model.load_state_dict(ckpt["model"])  # strict: any missing or unexpected key raises
    model.eval().to(device or torch.device("cpu"))
    return LoadedRun(Path(run_dir), path, int(ckpt["step"]), cfg, model)


def check_eval_split(run_cfg: dict, eval_cfg: dict, split: str) -> None:
    """Refuse to score `split` if it is the split the run trained on, shares a terrain seed with it
    (data/DESIGN.md, "Not checked here: leakage across configs"), or has other seeds in the run's
    config than in the eval config (a different test set isn't comparable)."""
    check_split_config(run_cfg["data"])
    trained = run_cfg["data"]["split"]
    if split == trained:
        raise LeakageError(f"the run trained on {trained!r}: it can't be scored on it (SPEC §7.3)")
    seen = set(terrain_seeds(run_cfg["data"], trained))
    wanted = terrain_seeds(eval_cfg["data"], split)
    shared = seen & set(wanted)
    if shared:
        raise LeakageError(
            f"eval split {split!r} shares {len(shared)} terrain seed(s), e.g. {sorted(shared)[:5]}, "
            f"with the {trained!r} split the run trained on"
        )
    ours = (terrain_seeds(run_cfg["data"], split), ball_seeds(run_cfg["data"], split))
    theirs = (wanted, ball_seeds(eval_cfg["data"], split))
    if ours != theirs:
        raise ValueError(
            f"split {split!r} differs between the run's config and the eval config (terrain and "
            f"ball seeds {ours[0][:1]}.. x{len(ours[0])} vs {theirs[0][:1]}.. x{len(theirs[0])}): "
            "a different test set isn't comparable"
        )


def goal_positions(pos, ep, t, mean, logvar, horizons: Sequence[int]) -> dict:
    """The goal head's (M, K, 2) displacement mean and log-variance as {k: (M, 2)} positions:
    pred = pos[t] + mean_k, var = exp(logvar_k), truth = pos[t+k], persistence = pos[t]."""
    mean, logvar = np.asarray(mean, dtype=np.float64), np.asarray(logvar, dtype=np.float64)
    if mean.shape != logvar.shape or mean.ndim != 3 or mean.shape[1] != len(horizons):
        raise ValueError(
            f"goal mean {mean.shape} / logvar {logvar.shape} don't match horizons {list(horizons)}"
        )
    x = pos[ep, t]
    out = {"pred": {}, "var": {}, "truth": {}, "persistence": {}}
    for j, k in enumerate(horizons):
        out["pred"][k] = x + mean[:, j]
        out["var"][k] = np.exp(logvar[:, j])
        out["truth"][k] = pos[ep, t + k]
        out["persistence"][k] = x.copy()
    return out


def model_outputs(
    model,
    es: EpisodeSet,
    ep: np.ndarray,
    t: np.ndarray,
    batch_size: int,
    device: torch.device,
    *,
    goal: bool = True,
    features: bool = True,
    pixel: bool = False,
) -> dict:
    """The goal head's output and the pooled context levels for windows (ep, t), and with
    pixel=True the pixel decoder's frame (its prediction of frame t+1, SPEC §4.7).

    Only the context frames t−C+1 … t go to the device, never ground truth or a later frame. The
    goal head and the probes read nothing else, so the targets and horizon heads that
    `WorldModel.forward` would also compute are skipped. This is the same path as forward's goal
    (`test_context_path_matches_forward`). Returns {"goal_mean", "goal_logvar"}: (M, K, 2)
    float64, or None without a goal head (or goal=False), and {"features": {level: (M,
    channels)}}, every encoder level spatially pooled, empty with features=False, and
    {"pixel"}: (M, 3, S, S) float32 in [0, 1], or None without a pixel decoder (or pixel=False)."""
    c = model.context_frames
    goal = goal and getattr(model, "goal", None) is not None
    pixel = pixel and getattr(model, "pixel", None) is not None
    means, logvars, feats, pixels = [], [], {}, []
    with torch.no_grad():
        for i in range(0, len(ep), batch_size):
            frames = window_frames(es, ep[i : i + batch_size], t[i : i + batch_size], c, 0)
            ctx = model.context_encoder(context_stack(frames.to(device), c))  # every level
            if goal:
                mean, logvar = model.goal(*(ctx[lv] for lv in model.goal_levels))
                means.append(mean.cpu().double().numpy())  # MPS has no float64: CPU first
                logvars.append(logvar.cpu().double().numpy())
            if features:
                for level, f in ctx.items():  # pool on the device: copy (B, C), not the map
                    pooled = f.mean(dim=(2, 3)).cpu().double().numpy()
                    feats.setdefault(level, []).append(pooled)
            if pixel:
                pixels.append(model.pixel([ctx[lv] for lv in model.pixel_levels]).cpu().numpy())
    return {
        "goal_mean": np.concatenate(means) if means else None,
        "goal_logvar": np.concatenate(logvars) if logvars else None,
        "features": {level: np.concatenate(v) for level, v in feats.items()},
        "pixel": np.concatenate(pixels) if pixels else None,
    }


def _probe_data(split: str, es: EpisodeSet, ep, t, features: dict, sim_cfg: dict):
    """Rows for the probes: pooled features, and as targets the ball's world position at t, its
    position relative to the camera (what a tracking camera's frame shows), and the episode's
    hidden parameters, drawn from its seeds (`sample_ball`, exact by M1's determinism)."""
    balls = [
        sample_ball(sim_cfg, int(a), int(b))
        for a, b in zip(es.terrain_seed, es.ball_seed, strict=True)
    ]
    hidden = {name: np.array([getattr(b, name) for b in balls]) for name in HIDDEN}
    targets = {
        "pos": es.ball_pos[ep, t],
        "pos_cam": es.ball_pos[ep, t] - es.cam_pos[ep, t],
        **{name: v[ep] for name, v in hidden.items()},
    }
    return probes.ProbeData(split, features, targets, es.terrain_seed[ep])


SEED_CHECKS = 8  # stored episodes per split checked against their seeds' ball draw


def _check_seed_balls(cfg: dict, split: str, es: EpisodeSet) -> None:
    """The probes take ball parameters from seeds. Check that against stored episodes: the first,
    the last and a seeded sample (reading all of train's 2000 files would cost about 1.5 GB)."""
    path = split_dir(cfg, split)
    rows = read_manifest(path)["episodes"]
    rng = np.random.default_rng(0)
    pick = {
        0,
        len(rows) - 1,
        *rng.choice(len(rows), size=min(SEED_CHECKS, len(rows)), replace=False).tolist(),
    }
    for i in sorted(pick):
        stored = load_episode(path / rows[i]["file"]).ball
        drawn = sample_ball(cfg["sim"], int(es.terrain_seed[i]), int(es.ball_seed[i]))
        if stored != drawn:
            raise ValueError(
                f"{path / rows[i]['file']}: stored ball {stored} differs from its seeds' draw {drawn}"
            )


def _ratio_to(goal_err: dict, base_err: dict) -> dict:
    out = {}
    for name, per_k in base_err.items():
        out[name] = {}
        for k, s in goal_err.items():
            b = per_k[k]["mean"]
            out[name][k] = s["mean"] / b if b > 0 else None
    return out


def score_split(
    cfg: dict,
    eval_cfg: dict,
    split: str,
    es: EpisodeSet,
    ep: np.ndarray,
    t: np.ndarray,
    outputs: dict,
    fitted_probes=None,
    simulate: Callable = simulate_from_state,
) -> dict:
    """Every SPEC §7.2 family for one split, as {family: metrics} (goal_head, visibility, events
    only with a goal head; probes only with `fitted_probes`, which are scored, never refit)."""
    ev = eval_cfg["eval"]
    levels, c = list(ev["coverage_levels"]), cfg["model"]["context_frames"]
    starts = np.unique(t)
    has_goal = outputs["goal_mean"] is not None
    horizons = list(cfg["model"]["goal_horizons"]) if has_goal else list(cfg["model"]["horizons"])
    extra, sim_info = {}, {}
    if ev["sim_params"]:
        extra, sim_info = simulator_baselines(
            cfg, split, es, starts, horizons, ev["sim_params"], simulate
        )
    base = baselines.evaluate(
        es.ball_pos, es.ball_vel, 1 / es.fps, starts, horizons, c, es.out_of_view, extra=extra
    )
    if sim_info:
        base["simulator"] = sim_info
    families = {"baselines": base}
    if has_goal:
        g = goal_positions(
            es.ball_pos, ep, t, outputs["goal_mean"], outputs["goal_logvar"], horizons
        )
        goal = metrics.evaluate_positions(
            g["pred"], g["truth"], g["persistence"], var=g["var"],
            out_of_view={k: es.out_of_view[ep, t + k] for k in horizons}, levels=levels,
        )  # fmt: skip
        goal["ratio_to_baseline"] = _ratio_to(goal["error_m"], base["error_m"])
        flags_t = {"out_of_view": es.out_of_view[ep, t], "occluded": es.occluded[ep, t]}
        flags_nt = {"out_of_view": es.out_of_view, "occluded": es.occluded}
        families["goal_head"] = goal
        families["visibility"] = {
            str(k): metrics.visibility(g["pred"][k], g["var"][k], g["truth"][k], flags_t, levels)
            for k in horizons
        }
        families["events"] = {
            str(k): metrics.event_behaviour(
                metrics.to_series(metrics.predictive_std(g["var"][k]), ep, t, es.n, es.T),
                flags_nt,
                ev["event_window"],
            )  # fmt: skip
            for k in horizons
        }
    if fitted_probes is not None:
        held = _probe_data(split, es, ep, t, outputs["features"], cfg["sim"])
        families["probes"] = probes.score_probes(fitted_probes, held).metrics["probes"]
    return families


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def check_eval_block(ev: dict) -> None:
    """Validate the eval: block up front. An unquoted `true` in sim_params is YAML's bool."""
    bad = [p for p in ev["sim_params"] if not isinstance(p, str) or p not in baselines.SIM_PARAMS]
    if bad:
        raise ValueError(
            f"eval.sim_params {bad}: expected names from {list(baselines.SIM_PARAMS)} "
            '(quote them in YAML: ["true", "mean"])'
        )
    metrics.coverage(np.zeros((1, 2)), np.ones((1, 2)), np.zeros((1, 2)), ev["coverage_levels"])
    for key in ("batch_size", "event_window"):
        if not _positive_int(ev[key]):
            raise ValueError(f"eval.{key} = {ev[key]!r}: expected an integer >= 1")
    if ev["probes"]["enabled"] and not (
        _positive_int(ev["probes"]["train_windows"]) and ev["probes"]["train_windows"] >= 2
    ):
        raise ValueError(
            f"eval.probes.train_windows = {ev['probes']['train_windows']!r}: expected >= 2"
        )


def check_episode_run(cfg: dict, run_dir) -> None:
    """Eval scores runs trained on the real episode splits only."""
    if cfg["data"]["source"] != "episodes":
        raise ValueError(
            f"{run_dir} trained on data.source={cfg['data']['source']!r}: eval needs a run "
            "trained on data.source=episodes (the real splits)"
        )


def eval_k(cfg: dict, has_goal: bool) -> int:
    """The largest horizon a window must leave room for: model.horizons and, with a goal head,
    goal_horizons. The baselines are recomputed on these windows, and they match `eval
    --baselines`'s whenever max(goal_horizons) <= max(model.horizons), as in base.yaml."""
    goal_ks = list(cfg["model"]["goal_horizons"]) if has_goal else []
    return max([*cfg["model"]["horizons"], *goal_ks])


def eval_windows(cfg: dict, es: EpisodeSet, has_goal: bool) -> tuple[np.ndarray, np.ndarray]:
    """(ep, t) of every window `eval --run` scores on `es`, t in [C-1, T-1-eval_k] (t = 3..39 at
    base.yaml), in `baselines.window_index` order."""
    starts = valid_starts(es.T, cfg["model"]["context_frames"], eval_k(cfg, has_goal))
    return baselines.window_index(es.n, starts)


def evaluate_run(
    run_dir: str | Path,
    eval_cfg: dict,
    splits: Sequence[str],
    checkpoint: str | Path | None = None,
    simulate: Callable = simulate_from_state,
) -> tuple[dict, Path]:
    """Score a run on held-out `splits` (SPEC §7) and write <run>/eval/<time>-<splits>.json:
    {"meta": ..., "report": metrics.report(...)}, with test as the headline and dev kept separate.

    Everything that needs no GPU happens outside the GPU lock: loading data, the seed checks,
    fitting probes, the simulator and the metrics. Only the forward passes run under it, and only
    on MPS. Returns the document and its path."""
    run_dir, ev = Path(run_dir), eval_cfg["eval"]
    splits = list(dict.fromkeys(splits))
    if not splits or not set(splits) <= set(HELD_OUT):
        raise ValueError(f"splits {splits}: --run scores {list(HELD_OUT)} only")
    check_eval_block(ev)  # before any model, data or lock: a typo shouldn't cost a GPU pass
    device = pick_device(ev["device"])
    run = load_run(run_dir, checkpoint, torch.device("cpu"))  # moved to the device under the lock
    cfg = run.cfg
    check_episode_run(cfg, run_dir)
    for split in splits:
        check_eval_split(cfg, eval_cfg, split)
    has_goal = getattr(run.model, "goal", None) is not None
    use_probes = bool(ev["probes"]["enabled"])
    c = cfg["model"]["context_frames"]
    k_eval = eval_k(cfg, has_goal)
    data = {s: load_split(cfg, s) for s in splits}
    windows = {s: eval_windows(cfg, es, has_goal) for s, es in data.items()}
    tr = tr_ep = tr_t = None
    if use_probes:
        trained = cfg["data"]["split"]
        tr = load_split(cfg, trained)
        ep_all, t_all = baselines.window_index(tr.n, valid_starts(tr.T, c, k_eval))
        rng = np.random.default_rng(ev["probes"]["seed"])
        n = min(int(ev["probes"]["train_windows"]), len(ep_all))
        pick = np.sort(rng.choice(len(ep_all), size=n, replace=False))
        tr_ep, tr_t = ep_all[pick], t_all[pick]
        for s, es in [(trained, tr), *data.items()]:
            _check_seed_balls(cfg, s, es)
    empty = {"goal_mean": None, "goal_logvar": None, "features": {}, "pixel": None}
    outputs, tr_out = dict.fromkeys(splits, empty), None
    if has_goal or use_probes:
        with gpu_lock(f"eval {run_dir.name}", enabled=device.type == "mps"):
            run.model.to(device)
            outputs = {
                s: model_outputs(
                    run.model, data[s], *windows[s], ev["batch_size"], device, features=use_probes
                )
                for s in splits
            }
            if use_probes:
                tr_out = model_outputs(
                    run.model, tr, tr_ep, tr_t, ev["batch_size"], device, goal=False
                )
            run.model.to("cpu")
    fitted = None
    if use_probes:
        fitted = probes.fit_probes(
            _probe_data(cfg["data"]["split"], tr, tr_ep, tr_t, tr_out["features"], cfg["sim"])
        )
    results = []
    for s in splits:
        families = score_split(cfg, eval_cfg, s, data[s], *windows[s], outputs[s], fitted, simulate)
        results.append(metrics.SplitResult(s, families))
    meta = {
        **run_metadata(),
        "run_dir": str(run_dir),
        "checkpoint": str(run.checkpoint),
        "step": run.step,
        "splits": splits,
        "windows": {s: int(len(windows[s][0])) for s in splits},
        "starts": {s: [int(windows[s][1].min()), int(windows[s][1].max())] for s in splits},
        "device": str(device),
        "eval": ev,
    }
    if not has_goal:
        meta["note"] = (
            "no goal head in this run (train.stages lacks goal): goal_head, visibility and events "
            "are not reported"
        )
    doc = {"meta": meta, "report": metrics.report(results)}
    text = json.dumps(doc, indent=2)  # fails here, before any file exists, if not JSON-safe
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    name = f"{stamp}-{'-'.join(splits)}-{secrets.token_hex(3)}.json"
    path = write_new(out_dir / name, text)  # all or nothing, never over an earlier eval
    return doc, path


def format_goal_table(metrics_: dict) -> str:
    """k, the goal head's mean error, its ratio to each baseline, and joint coverage at each of
    eval.coverage_levels."""
    goal = metrics_["goal_head"]
    names = list(goal["ratio_to_baseline"])
    levels = list(goal["calibration"][str(goal["horizons"][0])])
    head = (
        f"{'k':>4} {'error m':>9} "
        + " ".join(f"{'/' + n:>22}" for n in names)
        + "".join(f"  {'cov' + p:>8}" for p in levels)
    )
    lines = [head]
    for k in goal["horizons"]:
        ks = str(k)
        cells = [
            f"{'n/a' if r is None else f'{r:.3f}':>22}"
            for r in (goal["ratio_to_baseline"][n][ks] for n in names)
        ]
        cal = goal["calibration"][ks]
        lines.append(
            f"{k:>4} {goal['error_m'][ks]['mean']:9.4f} "
            + " ".join(cells)
            + "".join(f"  {cal[p]['joint']:8.3f}" for p in levels)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baselines", action="store_true", help="position baselines (M2)")
    mode.add_argument("--run", help="run directory to evaluate (M6)")
    p.add_argument(
        "--config",
        help="--baselines: the config to score. --run: the eval config (its eval: block and "
        "canonical split seeds), default configs/base.yaml",
    )
    p.add_argument(
        "--split",
        action="append",
        choices=["dev", "test"],
        help="repeatable for --run (default: test). test is the headline, dev is kept separate",
    )
    p.add_argument("--checkpoint", help="--run: a checkpoint file (default: the run's latest)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument(
        "--sim-params",
        nargs="*",
        choices=baselines.SIM_PARAMS,
        default=None,
        help=(
            "the simulator-aware baseline's ball parameters, minutes of CPU each (proposed: "
            f"{' '.join(baselines.PROPOSED_SIM_PARAMS)}). --baselines: none unless given. --run: "
            "replaces eval.sim_params; give it with no values for none"
        ),
    )
    args = p.parse_args(argv)
    splits = args.split or ["test"]

    if args.run:
        eval_cfg = load_config(args.config or BASE_CONFIG, args.set)
        if args.sim_params is not None:
            eval_cfg["eval"]["sim_params"] = list(args.sim_params)
        doc, path = evaluate_run(args.run, eval_cfg, splits, checkpoint=args.checkpoint)
        meta, report = doc["meta"], doc["report"]
        print(f"eval: {path}\ncheckpoint {meta['checkpoint']} (step {meta['step']})")
        blocks = [("headline", report["headline"]["metrics"] if report["headline"] else None)]
        blocks += [(f"development {s}", m) for s, m in report["development"].items()]
        for title, m in blocks:
            if m is None:
                continue
            print(
                f"\n{title}: {m['baselines']['windows']} windows (t in {m['baselines']['starts']})"
            )
            print(format_goal_table(m) if "goal_head" in m else meta.get("note", ""))
        return 0
    if len(splits) > 1:
        p.error("--baselines scores one --split")
    if args.checkpoint:
        p.error("--checkpoint is for --run")
    if not args.config:
        p.error("--baselines needs --config")
    cfg = load_config(args.config, args.set)
    # No GPU lock: the simulator runs MuJoCo on the CPU with render=False (no MPS, no GL), and
    # gpu_lock is for MPS jobs. Taking it would hold up GPU training for the whole eval.
    run, result = run_baselines(cfg, splits[0], sim_params=args.sim_params or [])
    print(
        f"run dir: {run}\nsplit {splits[0]}: {result['episodes']} episodes, "
        f"{result['windows']} windows (t in {result['starts']})"
    )
    print(format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
