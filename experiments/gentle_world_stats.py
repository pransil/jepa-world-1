"""The gentle world's justification (configs/world_gentle.yaml): event rates beside the full world's
and the candidates that lost. Writes stats.json to a fresh run dir and prints the tables.

    uv run python experiments/gentle_world_stats.py [--episodes 200]

Physics only, CPU, terrain seeds 0..N-1 with ball seed 0 (the episodes sim.preview's stats use).
For each world: preview's own event stats (contact rate, bounces per episode, out of view), and
what preview doesn't split out:
- the start settle. The ball starts at `drop` above the highest surface point under its footprint
  (sim/ball.py start_z), so on a slope it starts above the surface even at drop 0, and lands within
  a few frames. start_gap is the ball-surface clearance at frame 0. Bounces and airborne frames
  are also counted from frame SETTLE on, after the settle.
- wall frames: frames where the ball's footprint is inside an end-wall zone (|x| + r > the walls'
  inner edge). The walls sit outside the slope cap.
- slip: contact frames (contact at t-1 and t) where the ball's surface speed and its rolling
  speed (spin x r) differ by more than SLIP_REL of the speed (floor 0.2 m/s). A solid sphere
  rolls without slipping while tan(slope) <= 3.5 mu. MuJoCo's soft contact leaves a few percent
  of mismatch even then.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jepa_world.sim.episode import simulate_episode
from jepa_world.sim.preview import episode_stats
from jepa_world.utils.config import load_config
from jepa_world.utils.paths import new_run_dir
from jepa_world.utils.repro import run_metadata

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
WORLDS = {"full": CONFIGS / "base.yaml", "gentle": CONFIGS / "world_gentle.yaml"}
NO_LEDGE_NO_DROP = ["sim.terrain.ledges=[0, 0]", "sim.ball.drop=[0.0, 0.0]"]
CANDIDATES = {  # gentle-world candidates that lost, as overrides of base.yaml
    "no ledges, drop 0 only": NO_LEDGE_NO_DROP,
    "relief [0.5,1.5], slope 0.3": NO_LEDGE_NO_DROP
    + ["sim.terrain.relief=[0.5, 1.5]", "sim.terrain.max_slope=0.3"],
    "relief [0.5,1.5], slope 0.25": NO_LEDGE_NO_DROP
    + ["sim.terrain.relief=[0.5, 1.5]", "sim.terrain.max_slope=0.25"],
    "relief [0.5,1.5], slope 0.3, wavelength [5,20]": NO_LEDGE_NO_DROP
    + [
        "sim.terrain.relief=[0.5, 1.5]",
        "sim.terrain.max_slope=0.3",
        "sim.terrain.wavelength=[5.0, 20.0]",
    ],
    "relief [0.25,1.0], slope 0.2": NO_LEDGE_NO_DROP
    + ["sim.terrain.relief=[0.25, 1.0]", "sim.terrain.max_slope=0.2"],
}
SETTLE = 5
SLIP_REL = 0.05


def world_stats(sim: dict, n: int) -> dict:
    eps = [simulate_episode(sim, ts, 0, render=False) for ts in range(n)]
    st = episode_stats(eps)
    edge = sim["terrain"]["length"] / 2 - sim["terrain"]["wall_width"]
    wall = [np.abs(ep.ball_pos[:, 0]) + ep.ball.radius > edge for ep in eps]
    gap0 = [float(ep.terrain.gap(ep.ball_pos[0], ep.ball.radius)) for ep in eps]
    rel, slip_mu = [], []
    for ep in eps:
        v = ep.ball_vel
        surface = np.sign(v[:, 0]) * np.linalg.norm(v, axis=1)
        rolling = ep.ball_spin * ep.ball.radius
        held = ep.contact & np.r_[False, ep.contact[:-1]]
        r = np.abs(surface - rolling) / np.maximum(np.abs(surface), 0.2)
        rel.append(r[held])
        if (r[SETTLE:][held[SETTLE:]] > SLIP_REL).any():
            slip_mu.append(ep.ball.friction)
    rel = np.concatenate(rel)
    # Sliding on a slope would pick out the low-friction balls (tan(slope) > 3.5 mu); soft-contact
    # mismatch wouldn't. So: the friction of the episodes that slip after the settle.
    mu = np.percentile(slip_mu, [0, 50, 100]) if slip_mu else [np.nan] * 3
    return {
        "contact_rate": st["contact_rate"],
        "airborne_frame_rate": st["free_flight_frame_rate"],
        "bounces_per_episode": st["bounces_per_episode"],
        "out_of_view_rate": st["out_of_view_rate"],
        "episodes_with_out_of_view": st["episodes_with_out_of_view"],
        f"bounces_from_frame_{SETTLE}": sum(int(ep.bounce[SETTLE:].sum()) for ep in eps),
        f"airborne_frames_from_frame_{SETTLE}": sum(
            int((~ep.contact[SETTLE:]).sum()) for ep in eps
        ),
        "start_gap_median_m": float(np.median(gap0)),
        "start_gap_max_m": float(np.max(gap0)),
        "episodes_reaching_a_wall": sum(bool(w.any()) for w in wall),
        "wall_frame_rate": float(np.mean([w.mean() for w in wall])),
        "slip_contact_frame_rate": float(np.mean(rel > SLIP_REL)),
        "slip_rel_p99": float(np.percentile(rel, 99)),
        f"episodes_slipping_from_frame_{SETTLE}": len(slip_mu),
        "their_friction_min": float(mu[0]),
        "their_friction_median": float(mu[1]),
        "their_friction_max": float(mu[2]),
        "relief_median_m": float(np.median([ep.terrain.params["relief"] for ep in eps])),
        "max_speed_p90": st["max_speed_p90"],
    }


def table(stats: dict[str, dict]) -> str:
    names = list(stats)
    keys = list(stats[names[0]])
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(f"{stats[n][key]:.4g}" for n in names) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args(argv)
    worlds = {w: world_stats(load_config(p)["sim"], args.episodes) for w, p in WORLDS.items()}
    cands = {
        name: world_stats(load_config(WORLDS["full"], over)["sim"], args.episodes)
        for name, over in CANDIDATES.items()
    }
    run = new_run_dir("gentle-world-stats")
    out = {
        "episodes": args.episodes,
        "seeds": f"terrain 0..{args.episodes - 1}, ball 0",
        "worlds": worlds,
        "candidates": {n: {"overrides": CANDIDATES[n], **s} for n, s in cands.items()},
        "run": run_metadata(),
    }
    (run / "stats.json").write_text(json.dumps(out, indent=2))
    print(f"{args.episodes} episodes per world (terrain seeds 0..{args.episodes - 1}, ball seed 0)")
    print(table(worlds))
    print("\ncandidates that lost (overrides of base.yaml)")
    print(table(cands))
    print(f"run dir: {run}")
    return run


if __name__ == "__main__":
    main()
