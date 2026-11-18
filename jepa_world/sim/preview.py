"""Render a few episodes for a human to look at, and measure the generator.

    uv run python -m jepa_world.sim.preview --episodes 8 [--config configs/base.yaml] [--set k=v ...]

Writes to a fresh run dir (utils.paths.new_run_dir): one contact-sheet PNG per episode, with
events tinted (red border: out of view, yellow: bounce), one sample .npz, and stats.json. The
stats hold event rates over the physics-only sample, timings, file size, the restitution
calibration and its speed spread, and run metadata. The M1 summary cites stats.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from jepa_world.sim import contact
from jepa_world.sim.episode import Episode, simulate_episode
from jepa_world.sim.io import save_episode
from jepa_world.utils.config import load_config
from jepa_world.utils.paths import new_run_dir
from jepa_world.utils.repro import run_metadata

_RED, _YELLOW, _BLUE = (255, 40, 40), (255, 220, 0), (60, 120, 255)


def _runs(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(start indices, lengths) of the True runs in a 1-D bool array."""
    edges = np.diff(np.r_[0, flags.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return starts, ends - starts


def _run_stats(lengths: list[int]) -> dict | None:
    if not lengths:
        return None
    return {
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "max": int(max(lengths)),
    }


def episode_stats(episodes: list[Episode]) -> dict:
    """Event rates and ranges over a sample of episodes. Every number the M1 summary cites, plus
    the occlusion rates M7 tunes the posts with."""
    n = len(episodes)
    oov = [ep.out_of_view for ep in episodes]
    occ = [ep.occluded for ep in episodes]
    lengths, occ_lengths, below, above = [], [], 0, 0
    for ep in episodes:
        starts, runs = _runs(ep.out_of_view)
        lengths += runs.tolist()
        occ_lengths += _runs(ep.occluded)[1].tolist()
        for s in starts:
            if ep.ball_pos[s, 1] < ep.cam_pos[s, 1]:
                below += 1
            else:
                above += 1
    return {
        "episodes": n,
        "contact_rate": float(np.mean([ep.contact.mean() for ep in episodes])),
        # contact[t] is False only if no contact acted anywhere in [t-1, t] (see episode.py)
        "free_flight_frame_rate": float(np.mean([(~ep.contact).mean() for ep in episodes])),
        "bounces_per_episode": float(np.mean([ep.bounce.sum() for ep in episodes])),
        "out_of_view_rate": float(np.mean([o.mean() for o in oov])),
        "episodes_with_out_of_view": int(sum(bool(o.any()) for o in oov)),
        "out_of_view_exits": {"below": below, "above": above},
        "out_of_view_run_frames": _run_stats(lengths),
        "occluded_rate": float(np.mean([o.mean() for o in occ])),
        "episodes_with_occlusion": int(sum(bool(o.any()) for o in occ)),
        "occluded_run_frames": _run_stats(occ_lengths),
        "max_speed_p90": float(
            np.percentile([np.linalg.norm(ep.ball_vel, axis=1).max() for ep in episodes], 90)
        ),
        "max_abs_x": float(max(np.abs(ep.ball_pos[:, 0]).max() for ep in episodes)),
    }


def contact_sheet(
    frames: np.ndarray, cols: int = 10, pad: int = 1, tint: np.ndarray | None = None
) -> np.ndarray:
    """(T, S, S, 3) uint8 -> one grid image. tint (T, 3) uint8 colours each tile's border."""
    T, S = frames.shape[0], frames.shape[1]
    rows = -(-T // cols)
    cell = S + 2 * pad
    sheet = np.zeros((rows * cell, cols * cell, 3), np.uint8)
    for i in range(T):
        r, c = divmod(i, cols)
        y, x = r * cell, c * cell
        if tint is not None:
            sheet[y : y + cell, x : x + cell] = tint[i]
        sheet[y + pad : y + pad + S, x + pad : x + pad + S] = frames[i]
    return sheet


def _tints(ep: Episode) -> np.ndarray:
    tint = np.full((len(ep.t), 3), 64, np.uint8)
    tint[ep.bounce] = _YELLOW
    tint[ep.out_of_view] = _RED
    tint[ep.occluded] = _BLUE
    return tint


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(Path(__file__).parents[2] / "configs" / "base.yaml"))
    ap.add_argument(
        "--set", action="append", default=[], help="override, e.g. sim.camera_mode=track_full"
    )
    ap.add_argument("--episodes", type=int, default=8, help="rendered episodes (contact sheets)")
    ap.add_argument(
        "--stats-episodes", type=int, default=200, help="physics-only episodes for event rates"
    )
    args = ap.parse_args(argv)
    if args.stats_episodes < 1:
        ap.error("--stats-episodes must be at least 1")
    if not 0 <= args.episodes <= args.stats_episodes:
        ap.error("--episodes must be between 0 and --stats-episodes (it renders a subset of them)")

    from PIL import Image

    sim_cfg = load_config(args.config, args.set)["sim"]
    run = new_run_dir("sim-preview")

    t0 = time.perf_counter()
    physics = [simulate_episode(sim_cfg, ts, 0, render=False) for ts in range(args.stats_episodes)]
    sim_ms = (time.perf_counter() - t0) * 1000 / args.stats_episodes

    # Show the episodes with hidden-ball frames (out of view or occluded) first, then the rest.
    order = sorted(
        range(args.stats_episodes),
        key=lambda i: not (physics[i].out_of_view.any() or physics[i].occluded.any()),
    )
    rendered_ms, size_kb = [], None
    for ts in order[: args.episodes]:
        t0 = time.perf_counter()
        ep = simulate_episode(sim_cfg, ts, 0, render=True)
        rendered_ms.append((time.perf_counter() - t0) * 1000)
        Image.fromarray(contact_sheet(ep.frames, tint=_tints(ep))).save(
            run / f"episode-t{ts:03d}.png"
        )
        if size_kb is None:
            size_kb = save_episode(run / f"episode-t{ts:03d}.npz", ep).stat().st_size / 1024

    damping, e = contact.calibration_table(sim_cfg)
    spread = {
        f"{target}": {
            f"{v}": round(
                contact.measure_restitution(contact.damping_for(target, sim_cfg), sim_cfg, v), 4
            )
            for v in (1.0, 3.0, 6.0, 9.0)
        }
        for target in (0.2, 0.5, 0.8)
    }
    stats = {
        "events": episode_stats(physics),
        "timing_ms_per_episode": {
            "physics_only": round(sim_ms, 2),
            "with_render": round(float(np.mean(rendered_ms)), 2) if rendered_ms else None,
        },
        "episode_npz_kb_with_frames": round(size_kb, 1) if size_kb else None,
        "restitution": {
            "reachable": [round(float(e.min()), 4), round(float(e.max()), 4)],
            "measured_at_speed_m_s": spread,
        },
        "sim_cfg": sim_cfg,
        "run": run_metadata(),
    }
    (run / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(
        json.dumps(
            {
                k: stats[k]
                for k in ("events", "timing_ms_per_episode", "episode_npz_kb_with_frames")
            },
            indent=2,
        )
    )
    print(f"run dir: {run}")
    return run


if __name__ == "__main__":
    main()
