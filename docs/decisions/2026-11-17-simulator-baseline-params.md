# Simulator-aware baseline: which ball parameters

- **Date:** 2026-11-17   **Decided by:** user (2026-11-18)   **SPEC ref:** §7.1, §3.1
- **Status:** accepted

## Question
SPEC §7.1's physics baseline re-simulates the known terrain from the ball's true state at t
(position, velocity, spin), using `sim.episode.simulate_from_state`. It also needs the hidden ball
parameters (radius, friction, restitution). Which values should it use? The answer decides which
numbers are "the bar the learned model must beat on known terrain".

## Measured on the test split
Run `20261118T065246Z-worktree-wt-sim-baseline-baselines-test-590f46`, `baselines.json`: 200
episodes, 7400 windows, t = 3..39. The table gives the ratio of mean position error to
persistence's.

| k | const_velocity | simulator `true` (A) | simulator `mean` (B) | simulator `true_radius` (C) |
|---|---|---|---|---|
| 1 | 0.174 | 0.000 | 0.014 | 0.010 |
| 5 | 0.568 | 0.000 | 0.055 | 0.048 |
| 20 | 1.211 | 0.000 (1.0e-5 m) | 0.080 (0.137 m) | 0.072 (0.124 m) |

## Options
- **A: true parameters** (`ep.ball`). This is an oracle: it reproduces the ground truth to solver
  tolerance (mean error 1.0e-5 m at k = 20 above), so no model can beat it. It is useful as a
  sanity check and a ceiling, not as a bar.
- **B: mean parameters** (`sim.ball.mean_ball`, each range's midpoint: r = 0.25, μ = 0.55,
  e = 0.5). It matches SPEC §3.1's definition of hidden parameters, since none of the three is
  given to the model. It is the weakest of the three bars.
- **C: true radius, mean friction and restitution** (`true_radius`). The radius is *visible*:
  under the orthographic camera the ball is 2r·64/3.6 = 5.3–12.4 px across. Friction and
  restitution are not visible in a frame. This is the strongest fair bar: physics plus
  everything observable in the frames. It doesn't follow §3.1's list, though, which counts the
  radius as hidden. In the table it is only slightly stronger than B.

## Recommendation
**Report A and B, with B as the headline physics bar and A as the oracle ceiling.** C is available
(`--sim-params true mean true_radius`) for a secondary comparison, to show how much knowing the
radius is worth. B follows the SPEC's definition of what the model isn't given. A learned model
that beats B only by reading the ball's size from the pixels would show up as beating B but not
C. Choose C instead if the bar should credit the model only for what it can't see.

Both B and C are far stronger bars than constant velocity: 0.08 against 1.21 of persistence at
k = 20.

## Policy that applies to every option: a different radius at the true state
Measured in the run above:
- the true ball starts inside the terrain at 2648 of 7400 windows (down to −3.4 cm, in impacts);
- a 0.25 m ball at the true centre would start inside at 2095 (down to −10.7 cm).

So "lift any penetrating start" would break option A, and using the state as given launches the
bigger ball. The baseline maps the state the way a ball of the other radius would see it:
- **Same clearance.** The centre moves along the outward surface normal until `Terrain.gap` with
  the baseline radius equals the true ball's gap at t. On smooth terrain that is Δr·n, beside a
  face it is horizontal, and a corner takes two steps. The shift is 0 exactly for the true
  radius. A first version shifted only vertically. It left 307 windows unmatched against ledge
  faces, which carried about half of B's error (run `…-9fa587`: 0.253 m at k = 20, against
  0.137 m now).
- **Same slip.** Spin is scaled by r_true / r, so the surface speed ω·r, and with it the rolling
  or slipping state, is unchanged.
- **Same frame.** Predictions are the rollout minus the shift, so the radius offset doesn't
  count as position error.

These mappings use the clearance and surface speed at t, which are part of the state and visible
in the frames. They use the radius only to translate between the two balls.

## Decision
**B, mean parameters, is the headline physics bar. A, true parameters, is reported as the oracle
ceiling. C, `true_radius`, stays available as a secondary comparison.** This is the
recommendation above, accepted by the user on 2026-11-18.

## Consequences
The code already implements this, with no change needed:
- `baselines.PROPOSED_SIM_PARAMS = ("true", "mean")`, the set this decision accepts. The name
  predates the decision and is kept, because a rename is a code change.
- `eval.sim_params` in `configs/base.yaml` holds the same set, so `python -m jepa_world.eval
  --run` reports `simulator_mean` (the bar) and `simulator_true` (the oracle) beside the model.
- `--sim-params true mean true_radius` adds C.

`eval --baselines` runs the simulator only when `--sim-params` is given. Each option is one
21-frame rollout per window: 46–58 s on the 7400 test windows in the run above, and 443–655 s at
load average ~265 (run `…-9fa587`).
