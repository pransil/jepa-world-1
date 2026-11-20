# Where is the 48k model's error: in the bounces, or in smooth rolling too?

- **Date:** 2026-11-19 (evals 2026-11-20 07:41–07:45Z)   **Lane/branch:** worktree-wt-eval-regimes   **Commit:** 7e792f3 (clean, recorded in each eval file)
- **Config:** configs/base.yaml (eval block as committed: MPS, batch 64, coverage 0.5 / 0.9, sim_params true + mean)
- **Command:** `python -m jepa_world.regimes --run <run dir> --split test`, one run at a time under the GPU lock (about 85 s each, 1.7 GB peak)
- **Run dirs (step 48,000):**
  - `20261119T224253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s0-0f2411` (new defaults, seed 0), file `eval/regimes/20261120T074239Z-regimes-test-2dd8a9.json`
  - `20261120T003253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s1-ada237` (new defaults, seed 1), file `eval/regimes/20261120T074408Z-regimes-test-b231bd.json`
  - `20261119T070501Z-worktree-wt-exp-headline-48k-headline48k-s0-f5ccce` (old defaults, seed 0, for contrast), file `eval/regimes/20261120T074535Z-regimes-test-483811.json`
- **File location:** 7e792f3 wrote the three files into `eval/`. They were moved, unchanged, into
  `eval/regimes/` when the output moved there (the next commit on this branch), because other
  readers take every `eval/*.json` to be an `eval --run` report.

## Question

The user asked whether a gentle world would train better. The answer decides whether m5's
curriculum experiment (H) matters. So: does the model's error concentrate in bounces, or is the
model also weak on smooth rolling?

This is a measurement, not a test of a prediction. No hypothesis or deciding metric was
registered before the evals.

## Method

Test split: 200 episodes and 7400 windows, t in 3..39 (eval's own windows; C = 4, k_eval = 20).
Each (window, k) gets one regime from the episode files' flags:

- **bounce:** a `bounce` flag in frames t+1..t+k.
- **rolling:** no bounce in t−3..t+k, and `contact` in every one of those frames.
- **airborne:** everything else. This is a flight, or a bounce in the context but none ahead.

Per regime, everything is scored on the same windows:

- **The goal head:** mean error, and its ratio to each baseline (a ratio of mean errors, as
  SPEC §7.2's is).
- **Coverage:** the 50% and 90% joint coverage.
- **Latent relerr per head:** the regime's mean squared latent error over the full window set's
  target spread at that k. One denominator for all regimes, so share-weighted they add up to the
  full-set relerr.

**Cross-checks:**

- Every file's all-windows row equals that run's earlier `eval --run` headline for the same
  checkpoint: error, every ratio and both coverages at every k, with a max relative difference
  of 0.
- The orchestrator's baseline-only numbers reproduce exactly: the shares, and const-vel ÷
  persistence as a ratio of summed squared errors. At k5: rolling 42.3% 0.08, airborne 22.2%
  0.29, bounce 35.4% 0.92. At k20: rolling 39.2% 0.62, airborne 11.0% 0.67, bounce 49.8% 1.98.

**Which baselines are fair.** `const_velocity` reads the simulator's exact velocity at t. The
simulators restart from the exact state at t (position, velocity, spin): `simulator_mean` with
mid-range ball parameters, and `simulator_true` with the episode's own parameters.
`simulator_true` replays the episode, so its error is 0.0000 m at every k and its ratios mean
nothing. `const_velocity_fd` uses only the exact positions at t−1 and t. That is the closest to
what the model sees, though it is still exact where the model sees 64×64 pixels at 5.6 cm per
pixel.

## Results (test)

### Share of windows (the same for every run)

| k | rolling | airborne | bounce |
|---|---|---|---|
| 1 | 43.2% | 48.1% | 8.7% |
| 2 | 42.9% | 39.8% | 17.3% |
| 5 | 42.3% | 22.2% | 35.4% |
| 10 | 41.2% | 13.1% | 45.7% |
| 20 | 39.2% | 11.0% | 49.8% |

### Goal-head mean error (m), and each regime's share of the summed error

All windows / rolling / airborne / bounce. The share columns are for new s0; s1 and old s0 are
within 5 points of them.

| k | new s0 | new s1 | old s0 | share of summed error (abs): roll / air / bounce | share of summed squared error: roll / air / bounce |
|---|---|---|---|---|---|
| 1 | 0.028 / **0.010** / 0.040 / 0.048 | 0.031 / 0.012 / 0.044 / 0.054 | 0.043 / 0.013 / 0.064 / 0.073 | 16% / 69% / 15% | 3% / 80% / 17% |
| 2 | 0.053 / 0.019 / 0.074 / 0.087 | 0.056 / 0.020 / 0.079 / 0.095 | 0.073 / 0.025 / 0.109 / 0.112 | 16% / 56% / 29% | 3% / 65% / 33% |
| 5 | 0.127 / 0.049 / 0.168 / 0.195 | 0.132 / 0.045 / 0.178 / 0.209 | 0.152 / 0.061 / 0.208 / 0.225 | 16% / 29% / 54% | 4% / 36% / 61% |
| 10 | 0.251 / 0.110 / 0.286 / 0.368 | 0.255 / 0.101 / 0.281 / 0.386 | 0.282 / 0.137 / 0.305 / 0.407 | 18% / 15% / 67% | 6% / 17% / 78% |
| 20 | 0.563 / 0.285 / 0.444 / **0.808** | 0.567 / 0.278 / 0.460 / 0.818 | 0.605 / 0.324 / 0.488 / 0.852 | 20% / 9% / 71% | 11% / 6% / 82% |

### Goal head's ratio to each baseline, by regime (new s0)

All windows / rolling / airborne / bounce. Below 1 means the goal head is better. New s1's ratios
are within 13% of these at k = 1 and within 10% from k = 2 on.

| k | / persistence | / const_vel_fd (context positions) | / const_velocity (exact velocity) | / simulator_mean (exact state) |
|---|---|---|---|---|
| 1 | 0.21 / 0.12 / 0.24 / 0.30 | 0.72 / **0.99** / 0.75 / 0.50 | 1.21 / 1.81 / 1.67 / 0.46 | 15.6 / 11.5 / 40.4 / 4.5 |
| 2 | 0.21 / 0.11 / 0.22 / 0.29 | 0.52 / 0.65 / 0.54 / 0.44 | 0.68 / 0.93 / 0.90 / 0.42 | 8.2 / 8.1 / 27.3 / 3.4 |
| 5 | 0.22 / 0.12 / 0.23 / 0.28 | 0.35 / 0.36 / 0.36 / 0.35 | 0.38 / 0.42 / 0.44 / 0.35 | 4.0 / 6.3 / 15.5 / 2.6 |
| 10 | 0.25 / 0.14 / 0.28 / 0.30 | 0.29 / 0.27 / 0.34 / 0.29 | 0.30 / 0.29 / 0.37 / 0.29 | 3.3 / 5.8 / 8.3 / 2.6 |
| 20 | 0.33 / 0.22 / 0.28 / 0.40 | 0.27 / 0.24 / 0.31 / 0.28 | 0.27 / 0.25 / 0.32 / 0.27 | 4.1 / 5.8 / 5.9 / 3.7 |

Rolling against `const_velocity_fd` at k=1 is 0.99 for new s0, 1.09 for new s1 and 1.25 for old
s0. Old s0 is at or above 1 at k=1 in rolling and airborne windows, and below 1 from k=2 on.

### Calibration: 90% joint coverage (0.90 is calibrated)

All windows / rolling / airborne / bounce.

| k | new s0 | new s1 | old s0 |
|---|---|---|---|
| 1 | 0.90 / **0.96** / 0.87 / **0.81** | 0.91 / 0.98 / 0.87 / 0.83 | 0.91 / 0.96 / 0.88 / 0.85 |
| 2 | 0.86 / 0.89 / 0.85 / 0.81 | 0.90 / 0.96 / 0.86 / 0.82 | 0.87 / 0.90 / 0.84 / 0.85 |
| 5 | 0.83 / 0.81 / 0.84 / 0.84 | 0.87 / 0.90 / 0.85 / 0.85 | 0.82 / 0.81 / 0.81 / 0.83 |
| 10 | 0.85 / 0.84 / 0.80 / 0.88 | 0.87 / 0.90 / 0.82 / 0.86 | 0.79 / 0.77 / 0.75 / 0.82 |
| 20 | 0.83 / 0.84 / 0.83 / 0.83 | 0.88 / 0.91 / 0.88 / 0.86 | 0.82 / 0.82 / 0.81 / 0.83 |

50% coverage shows the same pattern. For new s0 at k=1 it is 0.73 rolling against 0.43 bounce;
at k=5 it is 0.48 / 0.48 / 0.44 (rolling / airborne / bounce).

### Latent heads: relerr over the full set's target spread

All windows / rolling / airborne / bounce, then each regime's share of the total.

| k | new s0 | new s1 | old s0 | new s0 share of the total: roll / air / bounce |
|---|---|---|---|---|
| 1 | 0.134 / 0.115 / 0.146 / 0.162 | 0.137 / 0.118 / 0.150 / 0.164 | 0.137 / 0.118 / 0.150 / 0.164 | 37% / 52% / 10% |
| 2 | 0.211 / 0.163 / 0.239 / 0.267 | 0.213 / 0.169 / 0.239 / 0.264 | 0.209 / 0.164 / 0.237 / 0.260 | 33% / 45% / 22% |
| 5 | 0.417 / 0.339 / 0.445 / 0.493 | 0.413 / 0.341 / 0.440 / 0.483 | 0.416 / 0.342 / 0.442 / 0.489 | 34% / 24% / 42% |
| 10 | 0.595 / 0.530 / 0.586 / 0.657 | 0.578 / 0.508 / 0.575 / 0.642 | 0.557 / 0.474 / 0.549 / 0.635 | 37% / 13% / 50% |
| 20 | 0.660 / 0.552 / 0.669 / 0.743 | 0.668 / 0.560 / 0.676 / 0.750 | 0.692 / 0.587 / 0.716 / 0.769 | 33% / 11% / 56% |

The test-set relerr here divides by the whole set's spread, not by a 32-window batch's as
training does, so it is not on the same scale as the training `relerr/k` columns in earlier
notes.

## Conclusion

**The goal head's error is concentrated in bounces at k ≥ 4, and in flight at short horizons.
Smooth rolling is not where it is weak, as far as these baselines can tell.**

- **Bounce windows hold most of the error at k ≥ 4.** At k=5 they are 35% of the windows and 54%
  of the summed error (61% of the squared error). At k=20 they are 50% of the windows and 71%
  (82%). At k=20 the error in bounce windows is 0.81 m, against 0.29 m in rolling.
- **At k ≤ 2 most of the error is in airborne windows:** 48% of the windows and 69% of the
  summed error at k=1.
- **Rolling windows are about 40% of the windows at every k, and hold 13–21% of the summed
  error.** At k=1 the rolling error is 1.05 cm, about 0.19 px. That is the same as the
  finite-difference baseline that gets the exact positions at t−1 and t (ratio 0.99, s1 1.09).
  From k=2 on the goal head beats that baseline in rolling (0.65 at k=2, 0.36 at k=5, 0.24 at
  k=20).
- **The baselines with exact state are still far better in rolling:** `simulator_mean` is
  5.8–11.5× better at every k, and the exact-velocity const-vel is 1.8× better at k=1. But that gap is
  about estimating the state from pixels, not about bounces. Those baselines get the simulator's
  velocity and spin, and the model sees 64×64 frames.
- **The latent heads' error is not concentrated in bounces.** At k=5, relerr in rolling windows
  is 0.34, against 0.42 overall and 0.49 in bounces. Rolling contributes 33–37% of the total at
  every k, close to its share of the windows. So the near-head error is broad.
- **The goal head's calibration miss depends on the regime mostly at k ≤ 2.** At k=1 the 90%
  region is too wide in rolling (0.96) and too narrow in bounces (0.81). For new s0 the three
  regimes' 90% coverages differ by 0.15 at k=1 and 0.08 at k=2. They differ by 0.03 at k = 3–5,
  0.07 at k=10 and 0.02 at k=20 (0.81–0.84 at k=5).
- **The seeds and the old reference agree on the shape.** The old-defaults run is worse
  everywhere, most at short k in airborne and bounce windows (k=1: 0.064 / 0.073 m, against
  0.040 / 0.048 for new s0).

**For H (a gentle world):** these numbers don't show a smooth-rolling weakness for a gentle
world to fix. The goal head is already at or better than the context-only physics baseline in
rolling, and the error it leaves is mostly in bounces and flights. Whether training on gentle
terrain first helps the model on bouncy test terrain is a different question, about transfer,
and only H itself can answer it. The user decides whether H goes ahead.

## Next (proposals; none run)

- **F (variance rescaling, approved, queued):** one scale per k fits the k ≥ 3 miss, which is
  close to even across regimes. At k = 1–2 it can't fix the regime split (too wide in rolling, too narrow
  in bounces); F should report coverage per regime too, using this module.
- **Report the share of error in bounce windows** beside the headline in later notes (one
  command). It moves when a change helps the bounces, while the all-window mean dilutes that.
- **The definition's edge:** the rolling span starts at the flag of frame t−3, which covers the
  interval before the first context frame (jepa_world/DESIGN.md). It moves a few windows from
  rolling to airborne, and was kept so the baseline cross-check reproduces.
