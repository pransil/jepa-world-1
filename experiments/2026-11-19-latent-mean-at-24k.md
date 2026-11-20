# Per-channel latent losses at 24k steps, with the pixel share held at 10%

- **Date:** 2026-11-19   **Lane/branch:** worktree-wt-exp-latent-mean-24k
- **Commit:** the commit that adds this note, which is pre-registered before the evals. The evals
  run at it, clean. The sha is filled in under Results.
- **Asked for:** proposal "Next 2" in [2026-11-19-headline-48k-new-defaults.md](2026-11-19-headline-48k-new-defaults.md),
  approved by the user 2026-11-19 (task E, relayed by the orchestrator session).
- **No training.** Three evals, each `uv run python -m jepa_world.eval --run <dir> --checkpoint
  <dir>/checkpoints/step_0024000.pt --split test --split dev`, and a JSON-only scorer
  (`experiments/latent_mean_24k_check.py`).

## Question
The 48k reference at the new defaults beat the old reference at every horizon, including the
deciding k = 20: 0.330 against 0.350 of persistence's error, on test. Three things changed at once:
1. per-channel latent losses (`latent_reduce: mean`);
2. a 10% pixel share;
3. a goal weight 4× smaller relative to L1's latent terms, and 2× relative to L2's
   (`w_goal_eff` = 0.5 / 256). This follows from change 1 in code.

The 24k pixel-share ablation isolated change 2: k = 20 stayed within the seed spread, and k = 1–5
improved. **Does changes 1 + 3, with the pixel share held at 10%, move the goal head at 24k?**

## The comparison: two runs that differ only in `latent_reduce`
| side | runs | `latent_reduce` | pixel | scored at |
|---|---|---|---|---|
| **sum** | `m7_pixel_share10` s0 / s1 (commit f1a7c03, [note](2026-11-18-pixel-share-ablation.md)): `20261119T115353Z-…-pixshare10-s0-aad2ee`, `20261119T134114Z-…-pixshare10-s1-880ec0` | `sum` (key absent) | share 0.10 | step 24,000, their final checkpoint; their existing evals |
| **mean** | the 48k reference at the new defaults (commit 6fd7059, [note](2026-11-19-headline-48k-new-defaults.md)): `20261119T224253Z-…-headline48k-new-s0-0f2411`, `20261120T003253Z-…-headline48k-new-s1-ada237` | `mean` | share 0.10 | `checkpoints/step_0024000.pt` |

- **Configs:** the recorded `config.yaml` of each seed pair differs in exactly two leaves:
  `loss.latent_reduce` (absent → `mean`) and `train.steps` (24,000 → 48,000). This was checked by
  diffing every leaf of the parsed files.
- **A step-24,000 checkpoint of a 48k run is a 24k run:**
  - there is no LR schedule;
  - `train.steps` sets only the loop length, the stage-start check and the final checkpoint
    (`jepa_world/train.py`);
  - training is deterministic on MPS. The old 48k runs matched the 24k headline runs bitwise
    through step 24,000 ([2026-11-18-headline-48k.md](2026-11-18-headline-48k.md)).
- **The training code is the same up to the switch.** The `sum` path at 6fd7059 is bitwise
  f1a7c03's (`experiments/latent_reduce_bitwise.py`,
  [decision](../docs/decisions/2026-11-19-latent-loss-normalization.md)).
- **Eval code:** `jepa_world/eval.py` changed between f1a7c03 and this commit only by an optional
  pixel output (off by default) and a refactored file write. The model path's `sum` branch is
  bitwise unchanged. A control eval checks this with an artifact instead of code reading
  (step 2 below).

## Hypothesis (pre-registered)
**H: the per-channel latent change (1 + 3) produced the reference's k = 20 gain, not the pixel
share.** If so, it shows at 24k with the pixel share fixed: the `mean` side beats the `sum` side at
k = 20.

- **Deciding metric:** goal-head position error at **k = 20**, as a ratio to persistence, on the
  **test** split (200 episodes, 7400 windows), at step 24,000.
- **Verdict rule** (per horizon, as in the reference's note):
  - **better** if both `mean` seeds are below both `sum` seeds;
  - **worse** if both are above;
  - otherwise **within the seed spread**.
- **Outcomes at k = 20:**
  - better → **H supported at 24k**;
  - worse → **H refuted at 24k**;
  - within → **inconclusive** (no difference claimed).
- **Reported without a prediction:**
  - k = 1–5 and 10, with the same rule;
  - dev, separately (SPEC §7.3);
  - against constant velocity and the mean-parameter simulator;
  - calibration (50% / 90%), visibility `std_ratio` and hidden coverage;
  - probes (`pos_cam` L1–L3 and pooled, `radius`, world `pos`);
  - training-batch `relerr`, `emb_std`, far `logvar`, `loss/pix`, `w_pix_eff`, `w_goal_eff` at
    23,001–24,000, and `goalratio` per 4k window.

**My expectation, stated separately from H:** k = 20 within the seed spread, or worse.
- **Disclosed: I had already seen the training-batch tables.** A dry run of the scorer printed them
  before I wrote this.
- **What they showed on training batches:** the `mean` runs escaped persistence later. Their k1
  `goalratio` over steps 12,001–16,000 was 0.927 / 0.925, against 0.854 / 0.845 for `sum`. Over
  20,001–24,000, k20 was 0.464 / 0.468 for `mean`, against 0.464 / 0.416.
- **Why that is a weak guide:** at 48k the reference's k = 20 gain showed on test and dev, but not
  in training-batch `goalratio` (its note).

**The limit of a 24k comparison.** At 24k the goal head is still on the steep part of its curve.
That is the reason M7 moved to 48k ([decision](../docs/decisions/2026-11-19-m7-ablations-at-48k.md)).
If `mean` learns more slowly early but ends better, a 24k comparison shows the timing, not the end
point. So an inconclusive or worse result at 24k does **not** rule out H at 48k. Only a 48k run of
`sum` + 10% would isolate the change there (≈ 4 GPU-h, a proposal).

## Procedure
Evals run one at a time, under the GPU lock, only with no other project job running and ≥ 10 GB
free. The worktree stays clean throughout: `eval.py` records the git state at the end.

1. Eval `mean` s0 and `mean` s1 at `step_0024000.pt`.
2. **Control:** re-eval `sum` s0 at its step-24,000 checkpoint, at this commit. The scorer compares
   every number in the two reports.
   - **Identical:** the original `sum` evals (f1a7c03) stand for both seeds.
   - **Any difference:** re-eval `sum` s1 too, and score both `sum` seeds from the re-evals
     (`--sum-eval latest`). Report the size of the difference.
3. Score with
   `uv run python experiments/latent_mean_24k_check.py --mean <mean s0> --mean <mean s1> --sum <sum s0> --sum <sum s1>`.

**Budget:** 3 evals (4 if the control differs), about 3 minutes each on MPS. No training.

## Results
**Commit:** every new eval ran at **e55f5ea**, clean (`meta.git.dirty: false` in each JSON), from this
lane's worktree, 2026-11-20 03:47–03:55 local, one at a time under the GPU lock.

**Evals scored** (all `--split test --split dev`, step 24,000):
| side | seed | run | eval file | sha |
|---|---|---|---|---|
| mean | 0 | `…-headline48k-new-s0-0f2411` | `eval/20261120T114925Z-test-dev-2edf26.json` (`--checkpoint step_0024000.pt`) | e55f5ea |
| mean | 1 | `…-headline48k-new-s1-ada237` | `eval/20261120T115052Z-test-dev-061b1b.json` (`--checkpoint step_0024000.pt`) | e55f5ea |
| sum | 0 | `…-pixshare10-s0-aad2ee` | `eval/20261120T115219Z-test-dev-4256b9.json` (re-eval; original `20261119T124735Z-test-dev-923920.json` at f1a7c03) | e55f5ea |
| sum | 1 | `…-pixshare10-s1-880ec0` | `eval/20261120T115544Z-test-dev-24855b.json` (re-eval; original `20261119T143443Z-test-dev-9450dc.json` at f1a7c03) | e55f5ea |

**Control (step 2).** The `sum` s0 re-eval at e55f5ea differed from its f1a7c03 eval. So by the
pre-registered rule `sum` s1 was re-evaluated too, and both `sum` seeds are scored from the
re-evals (`--sum-eval latest`).
- **The differences are only wall-clock timings, not a real discrepancy.** Each re-eval matches
  its original in every metric. In each report, 4 of 3,318 numbers differ, and all 4 are the
  simulator baselines' run times (`baselines/simulator/{mean,true}/seconds`, test and dev). On
  test, the mean-parameter time went from 33.6 to 34.1 s (s0) and from 33.0 to 35.3 s (s1).
- So the eval code is unchanged for these runs. The tables below are identical to scoring the
  original f1a7c03 evals: diffed, scorer output from the table onward.

**Scorer:** `uv run python experiments/latent_mean_24k_check.py --mean <0f2411> --mean <ada237> --sum <aad2ee> --sum <880ec0> --sum-eval latest`.
The tables below are its output.

### Deciding metric: goal head, TEST, error ÷ persistence at step 24,000 (s0 / s1)
| k | sum (m7_pixel_share10) | mean (step 24k) | sum mean | mean mean | change | verdict | mean runs at 48k |
|---:|---|---|---:|---:|---:|---|---|
| 1 | 0.3638 / 0.3569 | 0.3344 / 0.3915 | 0.360 | 0.363 | +0.7% | within | 0.2111 / 0.2310 |
| 2 | 0.3085 / 0.3031 | 0.2856 / 0.3371 | 0.306 | 0.311 | +1.8% | within | 0.2055 / 0.2184 |
| 3 | 0.2925 / 0.2841 | 0.2795 / 0.3178 | 0.288 | 0.299 | +3.6% | within | 0.2087 / 0.2208 |
| 4 | 0.2916 / 0.2807 | 0.2800 / 0.3151 | 0.286 | 0.298 | +4.0% | within | 0.2122 / 0.2234 |
| 5 | 0.2945 / 0.2885 | 0.2861 / 0.3187 | 0.292 | 0.302 | +3.7% | within | 0.2184 / 0.2277 |
| 10 | 0.3196 / 0.3088 | 0.3105 / 0.3411 | 0.314 | 0.326 | +3.7% | within | 0.2471 / 0.2506 |
| 20 | 0.4273 / 0.4042 | 0.4095 / 0.4349 | 0.416 | 0.422 | +1.6% | within | 0.3288 / 0.3312 |

The last column is the same `mean` runs at step 48,000, from their evals at 6fd7059
(`eval/20261120T003237Z-test-dev-f36d50.json`, `eval/20261120T022242Z-test-dev-8c61ea.json`).

**k = 20: within the seed spread, so the latent-mean hypothesis is inconclusive at 24k.** No
horizon is better or worse:
- `mean` s1 is above both `sum` seeds at every k;
- `mean` s0 is below both at k = 1–5, and between them at k = 10 and 20;
- the `mean` seeds differ more than the `sum` seeds (k = 20: 0.025 against 0.023).

### Against the other baselines, TEST (mean of seeds; the `sum` side in parentheses)
| k | / const-vel mean (sum) | / sim-mean mean s0 / s1 (sum mean) | error m mean s0 / s1 |
|---:|---|---|---|
| 1 | 2.08 (2.07) | 24.7 / 28.9 (26.6) | 0.045 / 0.052 |
| 2 | 1.03 (1.01) | 11.3 / 13.4 (12.1) | 0.074 / 0.087 |
| 3 | 0.73 (0.71) | 7.7 / 8.8 (8.0) | 0.104 / 0.119 |
| 4 | 0.60 (0.58) | 6.1 / 6.9 (6.2) | 0.135 / 0.151 |
| 5 | 0.53 (0.51) | 5.2 / 5.8 (5.3) | 0.166 / 0.185 |
| 10 | 0.39 (0.38) | 4.1 / 4.5 (4.2) | 0.316 / 0.347 |
| 20 | 0.35 (0.34) | 5.1 / 5.4 (5.2) | 0.702 / 0.745 |

### DEV (development only), error ÷ persistence (s0 / s1)
| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| sum | 0.354 / 0.357 | 0.300 / 0.307 | 0.280 / 0.283 | 0.275 / 0.273 | 0.275 / 0.275 | 0.287 / 0.287 | 0.384 / 0.379 |
| mean | 0.329 / 0.377 | 0.287 / 0.323 | 0.278 / 0.299 | 0.273 / 0.293 | 0.276 / 0.293 | 0.292 / 0.307 | 0.382 / 0.384 |

Dev verdicts: k1 within, k2 within, k3 within, k4 within, k5 worse, k10 worse, k20 within

### Calibration, TEST: joint fraction of truths inside the interval (s0 / s1)
| k | 50% sum | 50% mean | 90% sum | 90% mean |
|---:|---|---|---|---|
| 1 | 0.55 / 0.68 | 0.60 / 0.62 | 0.90 / 0.94 | 0.91 / 0.91 |
| 2 | 0.56 / 0.66 | 0.57 / 0.58 | 0.90 / 0.94 | 0.91 / 0.90 |
| 3 | 0.57 / 0.64 | 0.53 / 0.56 | 0.91 / 0.93 | 0.90 / 0.89 |
| 4 | 0.56 / 0.62 | 0.52 / 0.55 | 0.89 / 0.91 | 0.89 / 0.88 |
| 5 | 0.55 / 0.59 | 0.51 / 0.53 | 0.89 / 0.91 | 0.89 / 0.87 |
| 10 | 0.54 / 0.52 | 0.51 / 0.45 | 0.87 / 0.87 | 0.87 / 0.82 |
| 20 | 0.58 / 0.46 | 0.52 / 0.40 | 0.90 / 0.85 | 0.89 / 0.81 |

### Ball hidden, TEST: std ratio hidden ÷ visible, and 90% coverage when hidden (s0 / s1)
| k | std ratio sum | std ratio mean | 90% hidden sum | 90% hidden mean |
|---:|---|---|---|---|
| 1 | 4.0 / 4.4 | 4.7 / 3.5 | 0.73 / 0.81 | 0.78 / 0.67 |
| 2 | 4.8 / 4.7 | 5.6 / 4.0 | 0.74 / 0.78 | 0.81 / 0.68 |
| 3 | 4.9 / 4.9 | 5.8 / 4.2 | 0.77 / 0.79 | 0.85 / 0.69 |
| 4 | 4.9 / 4.8 | 5.6 / 4.3 | 0.76 / 0.79 | 0.85 / 0.71 |
| 5 | 4.7 / 4.9 | 5.3 / 4.3 | 0.76 / 0.81 | 0.83 / 0.74 |
| 10 | 3.8 / 4.4 | 4.2 / 3.8 | 0.78 / 0.85 | 0.82 / 0.74 |
| 20 | 2.5 / 3.2 | 2.8 / 3.1 | 0.92 / 0.90 | 0.92 / 0.90 |

### Probes, TEST R² (s0 / s1)
| target | level | sum | mean |
|---|---|---|---|
| pos_cam | L1 | 0.652 / 0.663 | 0.689 / 0.628 |
| pos_cam | L2 | 0.746 / 0.765 | 0.719 / 0.717 |
| pos_cam | L3 | 0.873 / 0.890 | 0.884 / 0.881 |
| pos_cam | pooled | 0.896 / 0.916 | 0.905 / 0.903 |
| radius | pooled | 0.907 / 0.918 | 0.918 / 0.916 |
| pos | pooled | 0.050 / 0.059 | 0.067 / 0.049 |

### Training batches, steps 23,001–24,000 (s0 / s1)
| key | sum | mean |
|---|---|---|
| relerr/k1 | 0.1062 / 0.1003 | 0.1196 / 0.1163 |
| relerr/k2 | 0.1718 / 0.1675 | 0.19 / 0.1845 |
| relerr/k3 | 0.2431 / 0.238 | 0.2622 / 0.2536 |
| relerr/k4 | 0.3022 / 0.299 | 0.3207 / 0.3117 |
| relerr/k5 | 0.3532 / 0.3509 | 0.367 / 0.3605 |
| relerr/k10 | 0.5322 / 0.5042 | 0.5369 / 0.5095 |
| relerr/k20 | 0.5849 / 0.5604 | 0.5909 / 0.5855 |
| emb_std/L1 | 1.779 / 1.782 | 1.795 / 1.798 |
| emb_std/L2 | 1.866 / 1.877 | 1.832 / 1.853 |
| emb_std/L3 | 1.865 / 1.835 | 1.85 / 1.812 |
| logvar/k10 | -0.1931 / -0.3466 | -0.1785 / -0.3109 |
| logvar/k20 | 0.115 / 0.01053 | 0.1174 / 0.06307 |
| loss/pix | 0.00136 / 0.001342 | 0.001284 / 0.00131 |
| w_pix_eff | 6.08e+04 / 5.949e+04 | 585.4 / 559.3 |
| pix_share | 0.1 / 0.1 | 0.1 / 0.1 |
| w_goal_eff | nan / nan | 0.001953 / 0.001953 |

### `goalratio` on training batches, mean over each 4k window (s0 / s1)
| steps | k1 sum | k1 mean | k5 sum | k5 mean | k20 sum | k20 mean |
|---|---|---|---|---|---|---|
| 4,001-8,000 | 1.064 / 1.056 | 1.051 / 1.053 | 0.991 / 0.990 | 0.988 / 0.989 | 0.988 / 0.988 | 0.988 / 0.989 |
| 8,001-12,000 | 0.988 / 1.000 | 0.999 / 1.003 | 0.963 / 0.973 | 0.976 / 0.976 | 0.973 / 0.975 | 0.977 / 0.978 |
| 12,001-16,000 | 0.854 / 0.845 | 0.927 / 0.925 | 0.826 / 0.794 | 0.918 / 0.889 | 0.893 / 0.841 | 0.944 / 0.908 |
| 16,001-20,000 | 0.529 / 0.504 | 0.537 / 0.587 | 0.438 / 0.403 | 0.444 / 0.498 | 0.564 / 0.500 | 0.599 / 0.572 |
| 20,001-24,000 | 0.429 / 0.423 | 0.400 / 0.460 | 0.347 / 0.327 | 0.332 / 0.372 | 0.464 / 0.416 | 0.464 / 0.468 |

**Training sanity:** every logged float is finite through step 24,000 in all four runs. `emb_std`
is 1.78–1.88 at every level in every run, with no sign of collapse. Far `logvar` is −0.35 to +0.12.
`pix_share` is 0.10 throughout on both sides. `w_goal_eff` is 0.5/256 = 0.001953 on the `mean`
side; the `sum` runs predate that column (nan).

## Conclusion
**Inconclusive at 24k:** the per-channel latent change, with the pixel share held at 10%, did not
move the goal head at k = 20. On test, `sum` is 0.427 / 0.404 and `mean` is 0.410 / 0.435 of
persistence's error, a mean-of-seeds change of +1.6% that is within the seed spread. No other
horizon on test is better or worse either. This is what I expected, stated before the evals
("within the seed spread, or worse").

- **What it does not show:** that the change is neutral at 48k. The goal head is still on the
  steep part of its curve at 24k: the same `mean` runs go from 0.410 / 0.435 at 24k to 0.329 /
  0.331 at 48k. A slower start that ends better would look like this.
- **So the source of the reference's k = 20 gain** (0.330 against 0.350 at 48k) **is still open.**
  At 24k, neither the pixel share
  ([2026-11-18-pixel-share-ablation.md](2026-11-18-pixel-share-ablation.md)) nor the latent change
  (this note) moves k = 20 on its own.
- **Differences reported, not decided on:**
  - **dev k = 5 and 10 are "worse"** by the both-seeds rule. At k = 5 the margin is about 0.001
    (0.276 / 0.293 against 0.275 / 0.275).
  - **Training-batch latent `relerr`** at k1–5 is higher on the `mean` side in both seeds (k1
    0.120 / 0.116 against 0.106 / 0.100).
  - **`mean` s1 is over-confident at long horizons:** 90% coverage at k = 20 is 0.81, against
    0.85–0.90 in the other runs. When the ball is hidden at k1–10 it is 0.67–0.74, against
    0.73–0.85.
  - L2 `pos_cam` probe R² is lower in both `mean` seeds (0.719 / 0.717 against 0.746 / 0.765). L3
    and pooled R² are within.

## Next
- **Proposal, not run:** a 48k run of `sum` + `pix_share` 0.10, seeds 0 and 1 (≈ 4 GPU-h). It is the
  only comparison that isolates the latent change at the step count M7 uses. The user's call.
- No change to the defaults is proposed from this result.
