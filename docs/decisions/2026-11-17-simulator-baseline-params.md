# Simulator-aware baseline: which ball parameters

- **Date:** 2026-11-17   **Decided by:** pending (user)   **SPEC ref:** §7.1, §3.1
- **Status:** proposed

## Question
SPEC §7.1's physics baseline re-simulates the known terrain from the ball's true state at t
(position, velocity, spin), using `sim.episode.simulate_from_state`. It also needs the hidden ball
parameters (radius, friction, restitution). Which values should it use? The answer decides which
numbers are "the bar the learned model must beat on known terrain".

## Options
- **A: true parameters** (`ep.ball`). This is an oracle: it reproduces the ground truth to solver
  tolerance (max 1.4e-8 m on the fast test's impact start), so no model can beat it. It is useful
  as a sanity check and a ceiling, not as a bar.
- **B: mean parameters** (`sim.ball.mean_ball`, each range's midpoint: r = 0.25, μ = 0.55,
  e = 0.5). It matches SPEC §3.1's definition of hidden parameters, since none of the three is
  given to the model. It is the weakest of the three bars.
- **C: true radius, mean friction and restitution** (`true_radius`). The radius is *visible*:
  under the orthographic camera the ball is 2r·64/3.6 = 5.3–12.4 px across. Friction and
  restitution are not visible in a frame. This is the strongest fair bar: physics plus
  everything observable in the frames. It doesn't follow §3.1's list, though, which counts the
  radius as hidden.

## Recommendation
**Report A and B by default, with B as the headline physics bar and A as the oracle ceiling.**
C is available (`--sim-params true mean true_radius`) for a secondary comparison, to show how much
knowing the radius is worth. B follows the SPEC's definition of what the model isn't given. A
learned model that beats B only by reading the ball's size from the pixels would show up as
beating B but not C. Choose C instead if the bar should credit the model only for what it can't
see.

## Policy that applies to every option: a different radius at the true state
The true centre with a different radius can start inside the terrain. That happens at 56% of
contact frames for r = 0.25 (scratch, 20 test episodes), by up to 8.8 cm. Even the true ball
penetrates up to 1.8 cm during impacts, so "lift any penetrating start" would break option A. The
baseline therefore maps the state the way a ball of the other radius would see it:
- **Same clearance.** The centre is shifted vertically by dz so that `Terrain.gap` with the
  baseline radius equals the true ball's gap at t (bisection; dz = 0 exactly when the radius is
  the true one).
- **Same slip.** Spin is scaled by r_true / r, so the surface speed ω·r, and with it the rolling
  or slipping state, is unchanged.
- **Same frame.** Predictions are the rollout minus the shift, so the radius offset doesn't
  count as position error.

These mappings use the clearance and surface speed at t, which are part of the state and visible
in the frames. They use the radius only to translate between the two balls.

## Decision
Pending. The code ships A and B as the defaults, and C as an option.

## Consequences
`baselines.SIM_PARAMS` holds `true`, `mean` and `true_radius`, and `DEFAULT_SIM_PARAMS` is
`("true", "mean")`. `python -m jepa_world.eval --baselines --sim-params ...` picks which ones run.
Each costs one 21-frame rollout per window, about 0.03–0.1 s depending on machine load, so about
4–12 min per option on the 7400 test windows. If the user picks one, change
`DEFAULT_SIM_PARAMS` and mark this file accepted.
