# Does one variance scale per horizon fix the goal head's coverage?

- **Date:** 2026-11-20 (evals 11:53Z)   **Lane/branch:** worktree-wt-exp-calibration   **Commit:** b7f4d57 (clean, recorded in each file)
- **Config:** configs/base.yaml (eval block as committed: MPS, coverage 0.5 / 0.9)
- **Command:** `python -m jepa_world.calibration --run <run dir>` (defaults: level 0.9, 5 folds),
  one run at a time under the GPU lock. s0 took 26 s wall time and s1 took 3 s; each peaked at 0.8 GB.
  s0's extra time was not CPU work in the process (4.3 s user, 0.9 s sys). s0 was the first job
  in a fresh worktree environment, so the likely cause is a cold start. I didn't instrument it.
- **Run dirs (step 48,000, new defaults):**
  - `20261119T224253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s0-0f2411` (seed 0), file `eval/calibration/20261120T115329Z-calibration-ab676c.json`
  - `20261120T003253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s1-ada237` (seed 1), file `eval/calibration/20261120T115335Z-calibration-88d73d.json`
- **Cross-check:** both runs used the full window sets: dev has 1480 windows and test has
  7400, with t in 3..39. Each run's before-scaling coverage equals its 48k `eval --run` report
  at both levels and every k, on test and on dev (s0 `eval/20261120T003237Z-test-dev-f36d50.json`,
  s1 `eval/20261120T022242Z-test-dev-8c61ea.json`). It also equals G's regimes file overall and
  per regime, with matching per-regime window counts (`eval/regimes/…-2dd8a9.json`,
  `…-b231bd.json`). The largest difference anywhere is 0.

## Question

For s0, the 48k goal head's 90% joint region holds only 82–86% at k ≥ 2 on test (its
`eval --run` report). G's note (`experiments/2026-11-19-error-by-motion-regime.md`) found that
from k = 3 on, the shortfall is about the same in every motion regime. Can one variance scale
per horizon fix that? The scale is fit on dev, which is held out from training, so that the 90% region holds
90%, then scored on test.

This is a measurement, not a test of a prediction. No hypothesis or deciding metric was
registered before the evals. Whether to make rescaling an eval default is the user's call.

## Method

These are the units in `jepa_world/calibration.py`; DESIGN.md's "Post-hoc calibration" section
has the details.

- **Score per window and k:** r² = Σ over x and z of (y − μ)² / σ², using the goal head's mean
  and diagonal variance. A calibrated Gaussian gives r² ~ χ²₂.
- **The fitted scale:** s_k = quantile(r², 0.9) / χ²₂(0.9). It is fit on dev: 10 terrains,
  40 episodes, 1480 windows. Multiplying σ² by s_k makes the 90% region hold exactly 90% on dev.
  "std ×" is √s_k. For comparison, the NLL-optimal scale on dev is mean(r²)/2.
- **Scored on test** (50 terrains, 200 episodes, 7400 windows), before and after scaling:
  50% and 90% joint coverage, and the mean NLL (training's goal_nll formula, in nats).
  Coverage is also scored per motion regime, using G's regime definitions.
- **The cross-fit (a check, not for adoption):** the 50 test terrains are dealt round-robin
  into 5 folds of 10. Each fold is scored with a scale fit on the other four folds: 40
  terrains, 5920 windows. This tells us what a scale fit on test-like data would achieve. It
  uses test for fitting, so it is not a headline number.

## Results

All numbers are on test. Each cell reads before → dev-fit → cross-fit.

#### s0: all test windows (7400)

| k | dev scale (std ×) | NLL-opt. dev scale | cross-fit fold scales | 90% cov. before → dev-fit → cross-fit | 50% cov. before → dev-fit → cross-fit | NLL before → dev-fit → cross-fit |
|---|---|---|---|---|---|---|
| 1 | 0.811 (×0.901) | 0.829 | 0.968–1.027 | 0.901 → 0.861 → 0.900 | 0.584 → 0.522 → 0.581 | -5.575 → -5.562 → -5.574 |
| 2 | 1.085 (×1.042) | 1.071 | 1.208–1.290 | 0.859 → 0.876 → 0.899 | 0.493 → 0.516 → 0.553 | -4.426 → -4.439 → -4.441 |
| 3 | 1.217 (×1.103) | 1.199 | 1.323–1.460 | 0.833 → 0.876 → 0.900 | 0.469 → 0.524 → 0.564 | -3.660 → -3.702 → -3.700 |
| 4 | 1.311 (×1.145) | 1.253 | 1.405–1.534 | 0.819 → 0.882 → 0.899 | 0.462 → 0.543 → 0.574 | -3.113 → -3.171 → -3.168 |
| 5 | 1.255 (×1.120) | 1.256 | 1.375–1.511 | 0.826 → 0.875 → 0.898 | 0.468 → 0.542 → 0.581 | -2.691 → -2.746 → -2.747 |
| 10 | 1.173 (×1.083) | 1.146 | 1.296–1.347 | 0.850 → 0.879 → 0.899 | 0.473 → 0.525 → 0.562 | -1.371 → -1.408 → -1.417 |
| 20 | 1.378 (×1.174) | 1.418 | 1.312–1.448 | 0.835 → 0.898 → 0.900 | 0.442 → 0.552 → 0.560 | 0.137 → 0.048 → 0.052 |

#### s1: all test windows (7400)

| k | dev scale (std ×) | NLL-opt. dev scale | cross-fit fold scales | 90% cov. before → dev-fit → cross-fit | 50% cov. before → dev-fit → cross-fit | NLL before → dev-fit → cross-fit |
|---|---|---|---|---|---|---|
| 1 | 0.830 (×0.911) | 0.849 | 0.906–0.942 | 0.913 → 0.884 → 0.900 | 0.612 → 0.550 → 0.586 | -5.351 → -5.348 → -5.353 |
| 2 | 0.953 (×0.976) | 0.976 | 1.005–1.035 | 0.897 → 0.887 → 0.900 | 0.544 → 0.527 → 0.553 | -4.354 → -4.350 → -4.355 |
| 3 | 0.996 (×0.998) | 1.015 | 1.049–1.098 | 0.885 → 0.884 → 0.901 | 0.527 → 0.525 → 0.551 | -3.685 → -3.684 → -3.689 |
| 4 | 1.021 (×1.010) | 1.034 | 1.080–1.153 | 0.876 → 0.881 → 0.900 | 0.518 → 0.525 → 0.556 | -3.198 → -3.200 → -3.203 |
| 5 | 1.054 (×1.027) | 1.032 | 1.071–1.161 | 0.873 → 0.886 → 0.900 | 0.518 → 0.535 → 0.559 | -2.797 → -2.802 → -2.801 |
| 10 | 0.953 (×0.976) | 0.979 | 1.137–1.204 | 0.870 → 0.861 → 0.898 | 0.529 → 0.512 → 0.589 | -1.385 → -1.376 → -1.395 |
| 20 | 1.038 (×1.019) | 1.010 | 1.043–1.214 | 0.881 → 0.886 → 0.900 | 0.562 → 0.575 → 0.611 | 0.130 → 0.125 → 0.129 |

#### Before any scaling: 90% joint coverage on dev (the fit set) vs test

| k | s0 dev | s0 test | s1 dev | s1 test |
|---|---|---|---|---|
| 1 | 0.933 | 0.901 | 0.930 | 0.913 |
| 2 | 0.886 | 0.859 | 0.912 | 0.897 |
| 3 | 0.861 | 0.833 | 0.901 | 0.885 |
| 4 | 0.846 | 0.819 | 0.897 | 0.876 |
| 5 | 0.853 | 0.826 | 0.889 | 0.873 |
| 10 | 0.870 | 0.850 | 0.906 | 0.870 |
| 20 | 0.845 | 0.835 | 0.895 | 0.881 |

#### s0: 90% joint coverage by regime, before → dev-fit → cross-fit

| k | rolling (windows) | airborne (windows) | bounce (windows) |
|---|---|---|---|
| 1 | 0.956 → 0.933 → 0.956 (3195) | 0.869 → 0.817 → 0.867 (3561) | 0.806 → 0.747 → 0.807 (644) |
| 2 | 0.892 → 0.906 → 0.920 (3177) | 0.845 → 0.862 → 0.892 (2943) | 0.809 → 0.833 → 0.864 (1280) |
| 3 | 0.842 → 0.882 → 0.905 (3162) | 0.837 → 0.880 → 0.904 (2404) | 0.812 → 0.860 → 0.885 (1834) |
| 4 | 0.811 → 0.876 → 0.895 (3148) | 0.840 → 0.886 → 0.902 (1982) | 0.812 → 0.885 → 0.902 (2270) |
| 5 | 0.807 → 0.860 → 0.884 (3133) | 0.836 → 0.875 → 0.897 (1644) | 0.842 → 0.892 → 0.915 (2623) |
| 10 | 0.836 → 0.867 → 0.889 (3050) | 0.804 → 0.843 → 0.860 (967) | 0.875 → 0.901 → 0.920 (3383) |
| 20 | 0.841 → 0.900 → 0.903 (2899) | 0.826 → 0.899 → 0.898 (814) | 0.832 → 0.896 → 0.897 (3687) |

#### s1: 90% joint coverage by regime, before → dev-fit → cross-fit

| k | rolling (windows) | airborne (windows) | bounce (windows) |
|---|---|---|---|
| 1 | 0.980 → 0.967 → 0.975 (3195) | 0.869 → 0.830 → 0.851 (3561) | 0.828 → 0.772 → 0.800 (644) |
| 2 | 0.957 → 0.951 → 0.958 (3177) | 0.863 → 0.852 → 0.867 (2943) | 0.825 → 0.809 → 0.830 (1280) |
| 3 | 0.939 → 0.938 → 0.951 (3162) | 0.861 → 0.860 → 0.877 (2404) | 0.824 → 0.822 → 0.846 (1834) |
| 4 | 0.919 → 0.922 → 0.934 (3148) | 0.847 → 0.852 → 0.880 (1982) | 0.844 → 0.849 → 0.871 (2270) |
| 5 | 0.901 → 0.911 → 0.925 (3133) | 0.850 → 0.865 → 0.879 (1644) | 0.853 → 0.869 → 0.883 (2623) |
| 10 | 0.901 → 0.893 → 0.930 (3050) | 0.824 → 0.811 → 0.853 (967) | 0.855 → 0.847 → 0.883 (3383) |
| 20 | 0.912 → 0.916 → 0.931 (2899) | 0.885 → 0.888 → 0.897 (814) | 0.857 → 0.862 → 0.876 (3687) |

## What the numbers say

1. **The dev-fit scale helps s0 at k ≥ 2, but not s1.**
   - s0 at k ≥ 2: the 90% region rises from 0.819–0.859 to 0.875–0.898, so its distance from
     0.9 shrinks from 0.041–0.081 to 0.002–0.025. NLL improves by 0.013–0.089 nats.
   - s1 was already close: 0.870–0.913 before. Its dev scales are 0.953–1.054, so the fit
     changes little: 0.861–0.887 after. It moves closer to 0.9 at k = 4, 5 and 20 and further
     away at k = 1, 2, 3 and 10. NLL moves by less than 0.01 nats either way.
   - The two seeds differ in calibration (0.82–0.86 vs 0.87–0.90 at k ≥ 2) by about as much as
     the s0 correction itself.
2. **At k = 1 the dev fit hurts both seeds.** Dev is over-covered at k = 1 (0.933 and 0.930),
   so it asks for a narrower region: scales of 0.81 and 0.83. Test was already near 0.9 (0.901
   and 0.913), and the narrower region drops it to 0.861 and 0.884. s0's NLL gets worse by
   0.013. Bounces take the largest drop: 0.806 → 0.747 for s0 and 0.828 → 0.772 for s1.
3. **Dev under-states test's spread, at every k and in both seeds.** Before any scaling, dev's
   90% coverage beats test's by 0.010–0.036. As a result, the cross-fit scales are 1.02–1.25×
   the dev scales, and the dev fit stops short of 0.9 at k ≥ 2 for s0. With test-like data the
   cross-fit reaches 0.898–0.901 at every k in both seeds.
   - Both seeds share the same 10 dev terrains. So their agreement can't tell a real dev/test
     difference apart from a 10-terrain sample that happens to be easy.
   - The cross-fit's fold-to-fold range doesn't settle it either. Each of those fits uses 40
     terrains, not 10.
4. **A single scale can't fix the shape.** When the scale is fit to put exactly 90% inside the
   90% region (the cross-fit), the 50% region holds 0.553–0.581 for s0 and 0.551–0.611 for s1,
   not 0.5. The standardized residuals are heavier-tailed than the head's Gaussian: too many
   small misses and too many large ones for any one width. On dev, the NLL-optimal scale is
   within 5% of the 90%-quantile scale (s0 k = 4: 1.253 vs 1.311; most k are closer). So
   choosing an NLL fit instead would change little.
5. **A single scale can't fix the regimes either.**
   - For s0 at k = 3–20, the regimes miss together (spread 0.015–0.072 before scaling), so
     the scale lifts all of them at once.
   - At k = 1 the regimes split for both seeds: rolling is over-covered (0.956 and 0.980) and
     bounces are under-covered (0.806 and 0.828). One scale moves both the same way.
   - s1 shows that split at every k ≥ 2: rolling holds 0.901–0.957 and bounces 0.824–0.857.
     s1's overall 0.87–0.90 is an average of a region that is too wide in rolling and too
     narrow in bounces, and rescaling keeps that mismatch.

## For the user to decide

Whether to adopt rescaling is your call. The options:

- **(a) Keep raw coverage as the headline, and use `python -m jepa_world.calibration` as a
  diagnostic.** (Recommended.) The dev-fit scale helps one of two seeds at k ≥ 2 and hurts both
  at k = 1. Its fit set is 10 terrains that under-state test's spread. And the remaining miss
  is in the shape and per regime, which one number per k can't fix.
- **(b) Report a dev-fit column next to raw coverage in eval.** This is cheap, and it is honest
  if it is labelled dev-fit. But point 3 says it will still read low at k ≥ 2, and point 1 says
  it can make k = 1 worse.
- **(c) Fix it in training.** For example, use a heavier-tailed likelihood for the goal head,
  or a variance that depends on the context's motion. That changes the model in SPEC, so it
  needs your decision. Points 4 and 5 are the evidence for it.
- **(d) If you want a dev fit, first make dev bigger** (more terrains), so that the fit set's
  sampling noise is small next to the 0.01–0.04 dev/test gap. That is a data-split change, so
  it is also your decision.

## Also in this lane

The CLI's per-regime line printed `n/a`. The row loop reused the variable that holds the level
key. The fix and its test, `test_the_table_prints_each_regimes_coverage_at_the_fitted_level`
(seen to fail first), are commit aed087d. The JSON files were never affected. Every number
above comes from the JSON.
