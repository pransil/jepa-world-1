# Hidden "friction" parameter: sliding μ plus rolling drag

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §3.1 (hidden parameters), §7.2 (probes)
- **Status:** accepted

## Question
SPEC §3.1 lists friction as a hidden per-episode parameter, and M6 probes for it, but it doesn't
say which physical friction is meant. In MuJoCo, sliding friction alone barely affects a rolling
ball: on flat ground, 2.0 m/s drops only to 1.94 m/s over 4 s (condim 3; a scratch measurement on
4.0.1 during design).

## Options
- **A: sliding μ + rolling drag.** One draw f sets sliding μ = f and rolling resistance
  = `contact.rolling_per_friction` · f (condim 6). Friction visibly slows the ball on every
  terrain. In the same scratch measurement, a rolling coefficient of 0.005 took 2.0 m/s down to 1.44 m/s in
  4 s.
- **B: sliding μ only.** Standard Coulomb. It is visible only when the ball slips on steep slopes
  or at landings.
- **C: rolling drag only.** μ is fixed high, and the parameter sets rolling resistance only.

## Decision
A. A single scalar keeps SPEC's three hidden parameters, and its effect shows up in every episode,
so the M6 probes have something to recover.

## Consequences
- `jepa_world/sim/contact.py` maps f to the geom friction triple (f, 0, rolling_per_friction·f)
  with condim 6. Torsional friction is 0 because the ball cannot spin about the contact normal.
- A test asserts that a higher f gives a lower final speed on flat ground.
