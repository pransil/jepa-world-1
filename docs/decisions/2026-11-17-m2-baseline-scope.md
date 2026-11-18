# M2 baselines: persistence and constant velocity; simulator-aware deferred to M6

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §7.1, §8 M2
- **Status:** accepted

## Question
SPEC §7.1 requires a persistence baseline and a physics baseline: "constant-velocity or
simulator-aware prediction, using known terrain". Which ones does M2 ship?

A simulator-aware baseline re-simulates the known terrain from the state at t. M1's `simulate`
starts only from `BallParams` (x0, drop, vx0), so this needs a new `jepa_world/sim/` entry point
that starts from (pos, vel, spin) at t. Under the coordination rule, M2 doesn't edit `sim/`.

## Options
- **A:** ship these three in M2, and defer the simulator-aware baseline to M6:
  - persistence;
  - constant velocity from the true v(t);
  - constant velocity from a finite difference of the last two context positions, which uses
    only what the model can see.
- **B:** also ship the simulator-aware baseline in M2. Ask the M1 lane for `simulate_from_state`,
  and re-simulate with the mean ball parameters. This blocks M2's merge on an M1 PR.

## Decision
A. It unblocks M3's dev-set test now. The simulator-aware bar matters for M6's headline claims,
which is when it is needed.

## Consequences
`jepa_world/baselines.py` has `persistence`, `const_velocity` and `const_velocity_fd`, and
`python -m jepa_world.eval --baselines` writes their errors per horizon to a run dir. The
`sim/` entry point is queued for M6. It is recorded as an M2 → M1 handshake in
docs/process/LANES.md, and the orchestrator session tracks it.
