"""The gentle world's justification (configs/world_gentle.yaml): event rates beside the full world's.

    uv run python experiments/gentle_world_stats.py [--episodes 200]

Physics only, CPU, terrain seeds 0..N-1 with ball seed 0 (the episodes sim.preview's stats use).
For each world it prints preview's own event stats (contact rate, bounces per episode, out of
view), and four things preview doesn't split out:
- the start settle: frames 0..SETTLE-1. The ball starts at drop above the highest surface point
  under its footprint (sim/ball.py start_z), so on a slope it starts a few cm up and lands within
  a few frames, even at drop 0. Bounces and airborne frames are counted before and after it.
- wall frames: frames where the ball's footprint is inside an end-wall zone (|x| + r > the walls'
  inner edge). The walls sit outside the slope cap.
- slip: contact frames (contact at t-1 and t) where the ball's surface speed and its rolling
  speed (spin x r) differ by more than SLIP_REL of the speed (floor 0.2 m/s). A solid sphere
  rolls without slipping while tan(slope) <= 3.5 mu; MuJoCo's soft contact leaves a few percent
  of mismatch even then.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jepa_world.sim.episode import simulate_episode
from jepa_world.sim.preview import episode_stats
from jepa_world.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
WORLDS = {"full": CONFIGS / "base.yaml", "gentle": CONFIGS / "world_gentle.yaml"}
SETTLE = 5
SLIP_REL = 0.05


def world_stats(sim: dict, n: int) -> dict:
    eps = [simulate_episode(sim, ts, 0, render=False) for ts in range(n)]
    st = episode_stats(eps)
    edge = sim["terrain"]["length"] / 2 - sim["terrain"]["wall_width"]
    late_bounce = sum(int(ep.bounce[SETTLE:].sum()) for ep in eps)
    late_air = sum(int((~ep.contact[SETTLE:]).sum()) for ep in eps)
    wall = [np.abs(ep.ball_pos[:, 0]) + ep.ball.radius > edge for ep in eps]
    rel = []
    for ep in eps:
        v = ep.ball_vel
        surface = np.sign(v[:, 0]) * np.linalg.norm(v, axis=1)
        rolling = ep.ball_spin * ep.ball.radius
        held = ep.contact & np.r_[False, ep.contact[:-1]]
        rel.append((np.abs(surface - rolling) / np.maximum(np.abs(surface), 0.2))[held])
    rel = np.concatenate(rel)
    relief = [ep.terrain.params["relief"] for ep in eps]
    return {
        "contact_rate": st["contact_rate"],
        "airborne_frame_rate": st["free_flight_frame_rate"],
        "bounces_per_episode": st["bounces_per_episode"],
        "out_of_view_rate": st["out_of_view_rate"],
        "episodes_with_out_of_view": st["episodes_with_out_of_view"],
        f"bounces_from_frame_{SETTLE}": late_bounce,
        f"airborne_frames_from_frame_{SETTLE}": late_air,
        "episodes_reaching_a_wall": sum(bool(w.any()) for w in wall),
        "wall_frame_rate": float(np.mean([w.mean() for w in wall])),
        "slip_contact_frame_rate": float(np.mean(rel > SLIP_REL)),
        "slip_rel_p99": float(np.percentile(rel, 99)),
        "relief_median_m": float(np.median(relief)),
        "max_speed_p90": st["max_speed_p90"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args(argv)
    stats = {w: world_stats(load_config(p)["sim"], args.episodes) for w, p in WORLDS.items()}
    print(f"{args.episodes} episodes per world (terrain seeds 0..{args.episodes - 1}, ball seed 0)")
    print(f"{'':36s} {'full':>10s} {'gentle':>10s}")
    for key in stats["full"]:
        print(f"{key:36s} " + " ".join(f"{stats[w][key]:10.4g}" for w in WORLDS))


if __name__ == "__main__":
    main()
