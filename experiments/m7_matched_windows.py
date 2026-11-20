"""Score a run's goal head on one dataset's frames, with the hidden windows flagged by another
dataset's ground truth (M7 §3 and §5: matched windows).

  uv run python experiments/m7_matched_windows.py --run <dir> --flags-config <yaml> \
      --flag out_of_view|occluded [--frames-config <yaml>] [--split test] [--checkpoint <ckpt>] \
      [--device auto|mps|cpu]

- **--frames-config** defaults to the run's own config, as saved in its checkpoint. Its split's frames
  are what the model sees.
- **--flags-config** names the dataset whose ground-truth flag marks the hidden windows.
- **Example (§3):** the track_full model scored on its own frames, at the windows where the ball
  is out of view under track_lagged.
- **The two datasets must hold the same episodes:** the same terrain and ball seeds, and
  bitwise-identical `ball_pos` and `ball_vel`. Camera mode and occluders don't change physics. If
  they differ, the script refuses.
- **Both configs must define the run's held-out split** with the run's seeds
  (`eval.check_eval_split`).
- **Windows and metrics are eval's:** every episode, at every t in `valid_starts(T, C, k_eval)`,
  through `eval.model_outputs`, `eval.goal_positions`, `metrics.visibility` (the flag at t, the
  context's last frame) and `metrics.event_behaviour`.
- **Self-consistency:** on the run's own dataset and flag, the output is exactly eval's
  `visibility[k][flag]` and `events[k][flag]`.
- **Output:** writes `<run>/matched/<time>-<split>-<flag>-<id>.json` (never into eval/), with the
  git state recorded at the start.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

import jepa_world  # noqa: F401  (sets PYTORCH_ENABLE_MPS_FALLBACK before torch work)
from jepa_world import baselines, metrics
from jepa_world import eval as eval_cli
from jepa_world.data.dataset import load_split, valid_starts
from jepa_world.utils.config import load_config
from jepa_world.utils.device import pick_device
from jepa_world.utils.locks import gpu_lock
from jepa_world.utils.paths import write_new
from jepa_world.utils.repro import run_metadata

FLAGS = ("out_of_view", "occluded")
SAME = ("terrain_seed", "ball_seed", "ball_pos", "ball_vel")


def check_same_episodes(frames_set, flags_set) -> None:
    """Refuse two episode sets that aren't the same episodes with the same physics."""
    for name in SAME:
        a, b = np.asarray(getattr(frames_set, name)), np.asarray(getattr(flags_set, name))
        if a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(
                f"{name} differs between the frames set and the flags set: they must be the same "
                "episodes with the same physics (only the camera or occluders may differ)"
            )


def score_windows(model, cfg, frames_set, flags_set, flag, device, batch_size) -> dict:
    """{"visibility": {k: metrics.by_visibility}, "events": {k: event_response}} for `flag`,
    taken from `flags_set`, on the goal head's predictions from `frames_set`'s frames."""
    if flag not in FLAGS:
        raise ValueError(f"flag {flag!r}: expected one of {list(FLAGS)}")
    check_same_episodes(frames_set, flags_set)
    ks = list(cfg["model"]["goal_horizons"])
    c = cfg["model"]["context_frames"]
    k_eval = max([*cfg["model"]["horizons"], *ks])
    ep, t = baselines.window_index(frames_set.n, valid_starts(frames_set.T, c, k_eval))
    out = eval_cli.model_outputs(model, frames_set, ep, t, batch_size, device, features=False)
    if out["goal_mean"] is None:
        raise ValueError("the run has no goal head: nothing to score")
    g = eval_cli.goal_positions(
        frames_set.ball_pos, ep, t, out["goal_mean"], out["goal_logvar"], ks
    )
    mask = np.asarray(getattr(flags_set, flag), dtype=bool)
    ev = cfg["eval"]
    visibility, events = {}, {}
    for k in ks:
        v = metrics.visibility(
            g["pred"][k], g["var"][k], g["truth"][k], {flag: mask[ep, t]}, ev["coverage_levels"]
        )
        visibility[str(k)] = v[flag]
        std = metrics.predictive_std(g["var"][k])
        series = metrics.to_series(std, ep, t, frames_set.n, frames_set.T)
        events[str(k)] = metrics.event_behaviour(series, {flag: mask}, ev["event_window"])[flag]
    return {
        "windows": int(len(ep)),
        "starts": [int(t.min()), int(t.max())],
        "visibility": visibility,
        "events": events,
    }


def score_run(
    run_dir, flags_cfg, flag, frames_cfg=None, split="test", checkpoint=None, device=None
):
    """Load the run, check both configs against it, score, and write the JSON. Returns
    (document, path)."""
    meta = run_metadata()  # at the start: the code that ran, even if the tree changes mid-run
    run_dir = Path(run_dir)
    run = eval_cli.load_run(run_dir, checkpoint, torch.device("cpu"))
    cfg = run.cfg
    frames_cfg = cfg if frames_cfg is None else frames_cfg
    for c in (frames_cfg, flags_cfg):
        eval_cli.check_eval_split(cfg, c, split)
    frames_set, flags_set = load_split(frames_cfg, split), load_split(flags_cfg, split)
    check_same_episodes(frames_set, flags_set)
    device = device or pick_device(cfg["eval"]["device"])
    with gpu_lock(f"matched {run_dir.name}", enabled=device.type == "mps"):
        run.model.to(device)
        scored = score_windows(
            run.model, cfg, frames_set, flags_set, flag, device, cfg["eval"]["batch_size"]
        )
        run.model.to("cpu")
    doc = {
        "meta": {
            **meta,
            "run_dir": str(run_dir),
            "checkpoint": str(run.checkpoint),
            "step": run.step,
            "split": split,
            "flag": flag,
            "frames_sim": frames_cfg["sim"],
            "flags_sim": flags_cfg["sim"],
            "device": str(device),
        },
        **scored,
    }
    text = json.dumps(doc, indent=2)
    out_dir = run_dir / "matched"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = write_new(out_dir / f"{stamp}-{split}-{flag}-{secrets.token_hex(3)}.json", text)
    return doc, path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True)
    p.add_argument("--flags-config", required=True)
    p.add_argument("--flag", required=True, choices=FLAGS)
    p.add_argument("--frames-config", default=None)
    p.add_argument("--split", default="test", choices=("test", "dev"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, choices=("auto", "mps", "cpu"))
    args = p.parse_args(argv)
    frames_cfg = load_config(args.frames_config) if args.frames_config else None
    device = pick_device(args.device) if args.device else None
    doc, path = score_run(
        args.run, load_config(args.flags_config), args.flag, frames_cfg, args.split,
        args.checkpoint, device,
    )  # fmt: skip
    print(f"wrote {path}", file=sys.stderr)
    print(f"windows {doc['windows']}, t in {doc['starts']}")
    print(
        "| k | n hidden | n visible | std hidden | std visible | std ratio | 90% hidden | error m hidden |"
    )
    print("|---:|---|---|---|---|---|---|---|")
    for k, v in doc["visibility"].items():
        cov = v["coverage_hidden"]["0.9"]["joint"] if v["has_hidden"] else float("nan")
        print(
            f"| {k} | {v['n_hidden']} | {v['n_visible']} | {v['std_hidden']} | "
            f"{v['std_visible']:.4f} | {v.get('std_ratio')} | {cov} | {v['error_m_hidden']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
