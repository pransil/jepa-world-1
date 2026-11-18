"""Ball: hidden per-episode parameters and initial state (SPEC §3.1). Pure."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from jepa_world.sim.terrain import Terrain

BALL_STREAM = 0xBA11  # SeedSequence stream id, distinct from the terrain's

# BallParams field <- its range in cfg["ball"], in draw order. sample_ball draws in this order, so
# reordering it changes every episode. mean_ball reads the same table.
RANGES = (
    ("radius", "radius"),
    ("friction", "friction"),
    ("restitution", "restitution"),
    ("x0", "start_x"),
    ("drop", "drop"),
    ("vx0", "vx0"),
)


@dataclass(frozen=True)
class BallParams:
    radius: float  # hidden
    friction: float  # hidden: sliding mu; rolling resistance = rolling_per_friction * friction
    restitution: float  # hidden: normal restitution at contact.restitution_ref_speed
    x0: float  # initial x of the centre
    drop: float  # initial height of the ball's bottom above the terrain under it
    vx0: float  # initial horizontal speed (spin starts at rolling, vx0 / radius)

    def to_dict(self) -> dict:
        return asdict(self)


def sample_ball(sim_cfg: dict, terrain_seed: int, ball_seed: int) -> BallParams:
    """Draws depend on (terrain_seed, ball_seed) only, through their own stream."""
    bc = sim_cfg["ball"]
    rng = np.random.default_rng(
        np.random.SeedSequence([BALL_STREAM, int(terrain_seed), int(ball_seed)])
    )
    return BallParams(**{field: float(rng.uniform(*bc[key])) for field, key in RANGES})


def mean_ball(sim_cfg: dict) -> BallParams:
    """The mean of `sample_ball`'s distribution: every draw is uniform, so each field is its range's
    midpoint. For re-simulating with unknown hidden parameters (`simulate_from_state` uses only
    radius, friction and restitution)."""
    bc = sim_cfg["ball"]
    return BallParams(
        **{field: (float(bc[key][0]) + float(bc[key][1])) / 2 for field, key in RANGES}
    )


def start_z(terrain: Terrain, ball: BallParams) -> float:
    """Initial z of the centre: `drop` above the highest terrain under the ball's footprint, so the
    ball never starts inside the surface."""
    # The polyline's maximum over [x0 - r, x0 + r] is at an end of the interval or at a node in it.
    ends = terrain.height_at(np.array([ball.x0 - ball.radius, ball.x0 + ball.radius]))
    nodes = terrain.heights[np.abs(terrain.x - ball.x0) <= ball.radius]
    under = float(np.concatenate([ends, nodes]).max())
    return under + ball.radius + ball.drop
