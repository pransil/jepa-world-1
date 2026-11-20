# The M7 reference at the new defaults: the full model at 48k with per-channel latents and a 10% pixel share

- **Date:** 2026-11-19   **Lane/branch:** worktree-wt-exp-headline-48k-new
- **Commit:** 6fd7059 (clean), for training and eval of both seeds. That commit added this note with
  the pass rules and nothing else. The results below were added after the runs.
- **Config:** `configs/exp_headline_48k.yaml`, unchanged (seed 0), plus `--set seed=1`
- **Run dirs** (each holds `eval/<time>-test-dev-<id>.json` from
  `uv run python -m jepa_world.eval --run <dir> --split test --split dev`, at step 48,000):
  - `20261119T224253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s0-0f2411` (seed 0; eval `20261120T003237Z-test-dev-f36d50.json`)
  - `20261120T003253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s1-ada237` (seed 1; eval `20261120T022242Z-test-dev-8c61ea.json`)
- **Scored by** `experiments/headline_48k_new_check.py` (every table below). Pointed at the old
  runs on both sides, it reproduces the old note's tables.
- **Old reference:** [2026-11-18-headline-48k.md](2026-11-18-headline-48k.md), commit 5ed1564:
  `20261119T070501Z-worktree-wt-exp-headline-48k-headline48k-s0-f5ccce` and
  `20261119T090343Z-worktree-wt-exp-headline-48k-headline48k-s1-5006f4`
- **Asked for:** the user approved the run on 2026-11-19 ("run the new 48k reference", relayed by the
  orchestrator session). It is the M7 reference
  ([decision](../docs/decisions/2026-11-19-m7-ablations-at-48k.md)), re-run because base.yaml's
  defaults changed.

## What changed since the old reference: three things at once

The resolved config differs from the `config.yaml` the old runs recorded in exactly two keys:
`loss.latent_reduce` (absent, meaning `sum` → `mean`) and `loss.pix_share` (absent → 0.10). Those
two keys make three changes to the objective:

1. **Per-channel latent losses** (`latent_reduce: mean`,
   [decision](../docs/decisions/2026-11-19-latent-loss-normalization.md)). Every per-level latent
   term and its variance guard is divided by the level's channel count (L1 64, L2 128, L3 256).
2. **A 10% pixel share** (`pix_share: 0.10`,
   [decision](../docs/decisions/2026-11-19-pixel-share-default.md)), in place of the fixed
   `w_pix` 0.05, whose share was at most 3e-5.
3. **A smaller effective goal weight:** `w_goal_eff` = 0.5 / 256 = 0.001953125, which follows from
   change 1 in code. Its ratio to L3's latent terms is unchanged. Its ratio to L1's terms is 4×
   smaller than before, and to L2's 2× smaller.

Nothing else on the training path changed between 5ed1564 and this commit: the fixed-`w_pix` and
`sum` paths are bitwise main's (`experiments/latent_reduce_bitwise.py`, and the pixel-share runs
matched the headline runs through step 6000).

Resolved loss block (from `load_config("configs/exp_headline_48k.yaml")`): `latent_distance: mse`,
`latent_reduce: mean`, `target_norm: none`, `collapse_guard: true`, `var_gamma: 1.0`,
`var_eps: 1e-4`, `var_reduce: sum`, `w_var: 25.0`, `w_k: 1.0`, `w_goal: 0.5`
(`w_goal_eff` 0.001953125), `w_pix: 0.05` (unused: `pix_share` is set), `pix_share: 0.1`,
`w_td: 0.5`. Train: B = 32, AdamW at lr 1e-3, weight decay 0.01, fp32, stages near / far / goal /
pixel from steps 0 / 2000 / 4000 / 6000, 48,000 steps. Model: SPEC §4.2 levels, U-Net skips on.

## Pass rules (committed before either run)

### A. Is it a usable reference? Each must hold in both seeds.
- **A1.** Every logged term is finite at every step, 1–48,000.
- **A2. No collapse.** `emb_std` at L1, L2 and L3 never falls below 50% of its step-2000 value over
  steps 2001–48,000. This is the collapse check's rule, extended to every level and the whole run.
  The old runs' lowest values were 71–74% (L1) and 99–102% (L2, L3).
- **A3. The far log-variance doesn't run away.** `logvar/k10` and `logvar/k20` average at least −3.0
  over steps 47,501–48,000, and no step after 2000 falls below −8. Again the collapse check's
  rule; the old runs had −0.36…+0.06 and minima of −1.8 or above.
- **A4. The pixel share holds:** `pix_share` is within 1e-4 of 0.10 at every step after 6000.
- **A5. The goal weight is as designed:** `w_goal_eff` is 0.001953125 at every step after 4000.

A failure of any of these is reported, and the run is not offered as the reference.

**A6, a reproducibility check (it doesn't disqualify):** steps 1–4000, before the goal head joins,
match the collapse check's `mean` run with the same seed bitwise, on every key both log
(`20261119T162336Z-…-latent-mean-s0-e82d5b`, `20261119T163737Z-…-latent-mean-s1-de9b34`, commit
00a4e78). The old 48k runs matched the first real run the same way, first differing at step 4001.

### B. Goal head vs the old reference: TEST, ratio to persistence, step 48,000
Per horizon:
- **better** if both new seeds are below both old seeds;
- **worse** if both new seeds are above both old seeds;
- otherwise **within the seed spread** (no difference claimed).

The old values are read from their eval JSONs by the scoring script. Rounded, s0 / s1:

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| old 48k | 0.3201 / 0.2914 | 0.2851 / 0.2597 | 0.2660 / 0.2464 | 0.2594 / 0.2442 | 0.2609 / 0.2484 | 0.2775 / 0.2650 | 0.3533 / 0.3470 |

- **Deciding metric: k = 20. Prediction: not worse.** Refuted if worse.
  - For: at 24k, the 10% share was within the seed spread at k = 20 (0.416 against 0.430).
  - Against: in the collapse check, `mean` made the far latent head's k = 20 relerr 2.6% worse on
    dev at 8k, and the goal head reads L3.
- **Secondary: k = 1–5. Prediction: better at each of them.**
  - At 24k the 10% share was better beyond the seed spread at every k from 1 to 5 (by 10–15%).
  - Whether that adds to the 24k → 48k gain is open (pixel-share note, "Next" 3).
  - Supported if better at all five, refuted if better at none, mixed otherwise.
- **k = 10:** reported, no prediction.

### C. Latent heads vs the old reference: train batches, `relerr` over steps 47,001–48,000
`relerr` is computed from channel sums under either reduction, so it compares directly.

- **Prediction: near k5 relerr is lower**, with both new seeds below both old seeds (0.398 / 0.406).
  - At 8k, the collapse check's `mean` arm regressed less after the far heads joined.
  - At 24k, the 10% share drifted slightly less (0.35 against 0.36).
- **Far k20 relerr:** reported. The collapse check expects a few percent worse.

### D. Reported without a prediction
- Every horizon against constant velocity and the mean-parameter simulator (test), and dev
  separately (SPEC §7.3).
- Calibration: joint coverage of the 50% and 90% intervals. The old reference was overconfident:
  its 90% interval held 79–82% of truths at k ≥ 5. Also the visibility `std_ratio` and hidden-ball
  coverage.
- Near k1–5 relerr over the whole run, `emb_std` per level, the far log-variance, `loss/pix`,
  `w_pix_eff`, and training-batch `goalratio` per 4k window.
- Eval's linear probes (`pos_cam`, `radius`, world `pos`).

### Attribution (written before the results)
No run isolates the three changes at 48k. Two runs isolate parts of them at shorter lengths:
- **The pixel-share ablation** ([note](2026-11-18-pixel-share-ablation.md); 24k, `sum`,
  `w_goal` 0.5) isolates 10% against ≈ 0.
  - Goal k = 1–5 were better beyond the seed spread, and k = 20 was within it.
  - Calibration and the visibility response didn't change.
  - The `pos_cam` probe rose at L2/L3, near drift was slightly less, and `loss/pix` fell 22%.
- **The collapse check** ([note](2026-11-19-latent-mean-collapse-check.md); 8k, near + far only,
  dev latent relerr) isolates `mean` against `sum` on the latent heads.
  - Near k5 was better, far k20 was 2.6% worse, and there was no collapse.
- **Nothing isolates the smaller goal weight** relative to L1 and L2.

So a change in the goal head can be called *consistent with* one of those results, never *caused
by* one change. A change neither run showed (for example in calibration) is assigned to none of
the three.

**Budget:** 2 seeds × 48,000 steps, about 2 h each on MPS, then one eval per seed. The order is
train s0, eval s0, train s1, eval s1, one at a time. Evals start only with ≥ 10 GB free.

## Results

Command (all six run dirs under `$JEPA_ARTIFACTS/runs/`):
`uv run python experiments/headline_48k_new_check.py --new <new s0> --new <new s1> --old <old s0>
--old <old s1> --mean-ref <latent-mean-s0> --mean-ref <latent-mean-s1>`. It exited 0 (A1–A5 pass).

### A. Gates: PASS in both seeds; the run is a usable reference

| | seed 0 | seed 1 |
|---|---|---|
| A1 every term finite, steps 1–48,000 | yes | yes |
| A2 lowest `emb_std` ÷ step 2000: L1 / L2 / L3 | 0.83 / 1.02 / 1.00 | 0.86 / 1.00 / 0.99 |
| A2 final `emb_std` L1 / L2 / L3 | 1.95 / 1.99 / 1.93 | 1.90 / 2.02 / 1.92 |
| A3 `logvar/k10`: last-500 mean, lowest | −0.19, −1.74 | −0.31, −1.70 |
| A3 `logvar/k20`: last-500 mean, lowest | −0.04, −1.26 | −0.05, −1.28 |
| A4 `pix_share` after step 6000 | 0.1000 (within 2e-8) | 0.1000 (within 2e-8) |
| A5 `w_goal_eff` after step 4000 | 0.001953125 exactly | 0.001953125 exactly |
| A6 first step differing from the collapse check's mean run | 4001 | 4001 |

- **A6 holds:** steps 1–4000 are bitwise identical to the collapse check's `mean` runs, on all 26
  shared keys. So that check's 8k results describe these runs' first 4000 steps exactly.
- **Each run took 1 h 48 min** on MPS (start to the step-48,000 checkpoint), and each eval about
  3 min. Training and evals ran strictly one at a time.

### B. Goal head on TEST (headline): 200 episodes, 7400 windows, t ∈ 3..39, step 48,000
Ratio to persistence, s0 / s1. "Better" means both new seeds are below both old seeds.

| k | old 48k | **new 48k** | old mean | **new mean** | change | verdict |
|---:|---|---|---:|---:|---:|---|
| 1 | 0.3201 / 0.2914 | 0.2111 / 0.2310 | 0.306 | **0.221** | −28% | better |
| 2 | 0.2851 / 0.2597 | 0.2055 / 0.2184 | 0.272 | **0.212** | −22% | better |
| 3 | 0.2660 / 0.2464 | 0.2087 / 0.2208 | 0.256 | **0.215** | −16% | better |
| 4 | 0.2594 / 0.2442 | 0.2122 / 0.2234 | 0.252 | **0.218** | −14% | better |
| 5 | 0.2609 / 0.2484 | 0.2184 / 0.2277 | 0.255 | **0.223** | −12% | better |
| 10 | 0.2775 / 0.2650 | 0.2471 / 0.2506 | 0.271 | **0.249** | −8% | better |
| **20** | 0.3532 / 0.3470 | **0.3288 / 0.3312** | 0.350 | **0.330** | −6% | **better** |

- **Deciding metric, k = 20: better.** The prediction was "not worse"; the result is better beyond
  the seed spread. The error is 0.563 / 0.567 m, against 0.605 / 0.594 m.
- **Secondary, k = 1–5: supported.** Better at all five, by 12–28% of the ratio.
- **k = 10:** better too.
- **The gain shrinks with the horizon:** −28% at k = 1, down to −6% at k = 20.

Against the other baselines (mean of seeds, test):

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| ÷ constant velocity, new (old) | 1.27 (1.75) | 0.70 (0.90) | 0.53 (0.63) | 0.44 (0.51) | 0.39 (0.45) | 0.30 (0.33) | 0.27 (0.29) |
| ÷ mean-parameter simulator, new s0 / s1 (old mean) | 15.6 / 17.1 (22.6) | 8.2 / 8.7 (10.8) | 5.8 / 6.1 (7.1) | 4.6 / 4.9 (5.5) | 4.0 / 4.2 (4.7) | 3.3 / 3.3 (3.6) | 4.1 / 4.1 (4.4) |

- **k = 1 is still worse than constant velocity** (1.27×), though much closer than before (1.75×).
  From k = 2 on, the model beats it.
- **The physics bar is still far off:** the error is 3.3–4.1× the mean-parameter simulator's at
  k = 10–20, and 16–17× at k = 1.

### Goal head on DEV (development only, never a headline; SPEC §7.3): 40 episodes, 1480 windows

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| old 48k s0 / s1 | 0.322 / 0.295 | 0.288 / 0.266 | 0.271 / 0.250 | 0.263 / 0.245 | 0.259 / 0.244 | 0.257 / 0.258 | 0.331 / 0.330 |
| new 48k s0 / s1 | 0.216 / 0.231 | 0.212 / 0.217 | 0.215 / 0.218 | 0.216 / 0.221 | 0.220 / 0.224 | 0.246 / 0.243 | 0.330 / 0.319 |

Dev agrees at every horizon. At k = 20 the margin is thin for seed 0: 0.3295, against 0.3305 for
the better old seed.

### Calibration and visibility (test)
Joint fraction of truths inside the predicted interval, s0 / s1:

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| 50%, old | 0.65 / 0.63 | 0.55 / 0.55 | 0.51 / 0.51 | 0.48 / 0.48 | 0.47 / 0.46 | 0.43 / 0.43 | 0.43 / 0.40 |
| 50%, new | 0.58 / 0.61 | 0.49 / 0.54 | 0.47 / 0.53 | 0.46 / 0.52 | 0.47 / 0.52 | 0.47 / 0.53 | 0.44 / 0.56 |
| 90%, old | 0.91 / 0.92 | 0.87 / 0.89 | 0.84 / 0.87 | 0.83 / 0.84 | 0.82 / 0.81 | 0.79 / 0.79 | 0.82 / 0.81 |
| 90%, new | 0.90 / 0.91 | 0.86 / 0.90 | 0.83 / 0.88 | 0.82 / 0.88 | 0.83 / 0.87 | 0.85 / 0.87 | 0.83 / 0.88 |

- **Less overconfident at long horizons.** At k = 10 the 90% interval holds 85–87% of truths,
  against 79%. At k = 20 it holds 83–88%, against 81–82%. It is still below 90% at every k ≥ 3.
- **Seed 1 is better calibrated than seed 0** at k ≥ 3, by 2–6 points at 90%.
- **The 50% interval at k = 1–2 holds fewer truths than before:** 0.49–0.61, against 0.55–0.65,
  which is closer to nominal. At k = 3–4, seed 0 is just under 50% (0.46–0.47) and seed 1 just
  over (0.52–0.53).
- **Uncertainty responds more strongly to a hidden ball.** Predicted std is 6.2–8.4× higher when
  the ball is out of view at k ≤ 5, against 4.1–6.5×.
- **Hidden-ball coverage at 90%:** 78–90% of truths, against 68–88%. Seed 0 rose at k = 2–10;
  seed 1 is mixed. Only 156 of the 7400 test windows have the ball hidden.

### Probes (eval's linear probes, test R²)

| target, level | old s0 / s1 | new s0 / s1 |
|---|---|---|
| `pos_cam`, L1 | 0.691 / 0.703 | 0.681 / 0.649 |
| `pos_cam`, L2 | 0.719 / 0.758 | 0.771 / 0.759 |
| `pos_cam`, L3 | 0.870 / 0.877 | 0.925 / 0.907 |
| `pos_cam`, pooled | 0.902 / 0.911 | 0.936 / 0.923 |
| `radius`, pooled | 0.914 / 0.928 | 0.935 / 0.931 |
| world `pos`, pooled | 0.058 / 0.042 | 0.049 / 0.044 |

- **Position is more readable at L3** (+0.03 to +0.06), and pooled.
- **It is less readable at L1** (−0.01 to −0.05).
- World position stays unreadable, as before (the camera tracks the ball).

### C. Latent heads (train batches, `relerr`, steps 47,001–48,000)
- **Near k5: within the seed spread. The prediction ("lower") is not supported.** It is
  0.402 / 0.397, against 0.397 / 0.406.
- **Far k20:** 0.553 / 0.551, against 0.560 / 0.539, also within the spread.
- **Far k10 is worse in both seeds:** 0.567 / 0.549, against 0.527 / 0.507 (+8% on the mean).

`relerr` over the run (1.0 = no better than the batch-mean target), new s0 / s1, with the old runs
below:

| steps | k1 | k2 | k3 | k4 | k5 | k10 | k20 |
|---|---|---|---|---|---|---|---|
| 1,001–2,000 new | 0.047 / 0.047 | 0.077 / 0.078 | 0.115 / 0.114 | 0.156 / 0.153 | 0.196 / 0.195 | 2.28 / 2.11 | 2.36 / 2.27 |
| old | 0.047 / 0.047 | 0.078 / 0.078 | 0.115 / 0.114 | 0.156 / 0.154 | 0.197 / 0.197 | 2.28 / 2.11 | 2.37 / 2.27 |
| 5,001–6,000 new | 0.066 / 0.065 | 0.122 / 0.120 | 0.180 / 0.176 | 0.230 / 0.223 | 0.272 / 0.266 | 0.469 / 0.473 | 0.673 / 0.691 |
| old | 0.071 / 0.060 | 0.135 / 0.118 | 0.200 / 0.178 | 0.256 / 0.231 | 0.304 / 0.278 | 0.483 / 0.499 | 0.655 / 0.696 |
| 11,001–12,000 new | 0.092 / 0.094 | 0.155 / 0.155 | 0.218 / 0.218 | 0.271 / 0.270 | 0.314 / 0.317 | 0.496 / 0.480 | 0.644 / 0.660 |
| old | 0.084 / 0.081 | 0.151 / 0.150 | 0.219 / 0.218 | 0.276 / 0.273 | 0.325 / 0.322 | 0.500 / 0.498 | 0.628 / 0.642 |
| 23,001–24,000 new | 0.120 / 0.116 | 0.190 / 0.184 | 0.262 / 0.254 | 0.321 / 0.312 | 0.367 / 0.360 | 0.537 / 0.509 | 0.591 / 0.585 |
| old | 0.111 / 0.107 | 0.177 / 0.177 | 0.247 / 0.249 | 0.309 / 0.310 | 0.362 / 0.362 | 0.504 / 0.486 | 0.583 / 0.564 |
| 35,001–36,000 new | 0.131 / 0.128 | 0.204 / 0.199 | 0.278 / 0.270 | 0.339 / 0.330 | 0.389 / 0.381 | 0.556 / 0.535 | 0.579 / 0.568 |
| old | 0.126 / 0.121 | 0.194 / 0.193 | 0.265 / 0.267 | 0.328 / 0.331 | 0.382 / 0.384 | 0.513 / 0.500 | 0.574 / 0.554 |
| 47,001–48,000 new | 0.138 / 0.137 | 0.214 / 0.211 | 0.290 / 0.284 | 0.351 / 0.345 | 0.402 / 0.397 | 0.567 / 0.549 | 0.553 / 0.551 |
| old | 0.138 / 0.137 | 0.209 / 0.212 | 0.281 / 0.287 | 0.344 / 0.352 | 0.397 / 0.406 | 0.527 / 0.507 | 0.560 / 0.539 |

- **The near heads' early advantage under `mean` doesn't last.**
  - At steps 5,001–6,000, k5 is 0.272 / 0.266 against 0.304 / 0.278. That is the collapse check's
    effect, and these steps are bitwise that run's up to step 4000.
  - By 12k the near heads are level with the old runs, and k1–k2 are slightly worse from 11k to 36k.
  - By 48k they are level again.
- **The near heads drift worse over the run, as before:** k5 0.27 → 0.40 from 6k to 48k.

### Training: embedding spread, far log-variance, pixel term, goal head

| steps | `emb_std` L1 / L2 / L3, new s0 | same, old s0 | `logvar/k10` s0 / s1, then `logvar/k20` s0 / s1 (new) |
|---|---|---|---|
| 11,001–12,000 | 1.68 / 1.58 / 1.50 | 1.71 / 1.59 / 1.54 | −0.32 / −0.38, −0.04 / −0.04 |
| 23,001–24,000 | 1.79 / 1.83 / 1.85 | 1.84 / 1.82 / 1.88 | −0.18 / −0.31, +0.12 / +0.06 |
| 47,001–48,000 | 1.87 / 1.97 / 1.90 | 1.91 / 1.94 / 1.99 | −0.20 / −0.29, −0.04 / −0.04 |

The spread grows at every level as before. L3 ends about 5% lower (1.89–1.90 against 1.99–2.00).
The far log-variance is flat and near zero, as in the old runs.

| steps | `loss/pix` new s0 / s1 | `loss/pix` old s0 / s1 | `w_pix_eff` new s0 / s1 |
|---|---|---|---|
| 6,001–6,250 | 0.0532 / 0.0469 | 0.0670 / 0.0582 | 34 / 33 |
| 23,001–24,000 | 0.00128 / 0.00131 | 0.00175 / 0.00173 | 585 / 559 |
| 47,001–48,000 | 0.00092 / 0.00097 | 0.00135 / 0.00135 | 953 / 862 |

- **The decoder reconstructs better:** `loss/pix` ends 30% lower.
- **`w_pix_eff` keeps growing as `loss/pix` falls.** The share holds at 0.1000 throughout.

Goal head on training batches (`goalratio`, the error ÷ persistence's, mean over each 4k window),
new s0 / s1 against old s0 / s1:

| steps | k1 new | k1 old | k5 new | k5 old | k20 new | k20 old |
|---|---|---|---|---|---|---|
| 16,001–20,000 | 0.537 / 0.587 | 0.603 / 0.591 | 0.444 / 0.498 | 0.501 / 0.476 | 0.599 / 0.572 | 0.603 / 0.549 |
| 24,001–28,000 | 0.332 / 0.400 | 0.442 / 0.437 | 0.291 / 0.321 | 0.339 / 0.325 | 0.399 / 0.414 | 0.423 / 0.393 |
| 36,001–40,000 | 0.245 / 0.298 | 0.384 / 0.375 | 0.239 / 0.266 | 0.283 / 0.278 | 0.347 / 0.354 | 0.359 / 0.346 |
| 44,001–48,000 | 0.227 / 0.263 | 0.359 / 0.347 | 0.227 / 0.247 | 0.266 / 0.262 | 0.332 / 0.340 | 0.340 / 0.330 |

- **The goal head's escape from persistence starts at the same point** (12k–16k), in both
  versions.
- **The short horizons pull ahead from 16k–20k,** and the gap keeps widening to the end.
- **On training batches, k20 does not separate:** 0.332 / 0.340 against 0.340 / 0.330 over the
  last 4k steps. The k = 20 gain shows on test and dev, not here.
- **It is still improving at 48k:** over the last 4k steps k1 fell 3–6% and k20 about 2%.

### What can be attributed (the pre-registered limits)
Three things changed at once, and no run isolates them at 48k.

- **Consistent with the 10% pixel share** (its 24k ablation showed the same direction):
  - the short-horizon gain at k = 1–5. At 24k, 10% against ≈ 0 gave 10–15%; here it is 12–28%,
    on top of the old run's own 24k → 48k gain;
  - the `pos_cam` probe rising at L3 and falling at L1;
  - `loss/pix` falling (−22% at 24k, −30% here).
- **Consistent with the collapse check** (`mean` against `sum`, 8k):
  - near relerr lower at 5–6k. These steps are that run up to step 4000, bitwise;
  - no collapse, and a flat far log-variance.
- **Assigned to none of the three** (neither isolating run showed them):
  - the k = 20 gain beyond the seed spread. At 24k the pixel share was within the spread at
    k = 20, and the collapse check put far-latent k20 2.6% worse. The gain also doesn't show in
    training-batch `goalratio`;
  - the k = 10 gain;
  - the better calibration at k ≥ 10. The pixel share didn't change calibration at 24k;
  - the stronger response to a hidden ball. The pixel share didn't change it at 24k;
  - the far k10 latent head 8% worse. The collapse check had k10 within the spread on dev at 8k.

## Conclusion
**Supported, and better than predicted.** At 48,000 steps on the test terrains, the full model at
the new defaults has a goal-head k = 20 error of 0.329 / 0.331 of persistence's (mean **0.330**).
The old reference had 0.353 / 0.347 (mean 0.350). Both new seeds are below both old seeds, so the
prediction "not worse" holds with a margin. Every other horizon is better too, most at k = 1
(0.221 against 0.306, −28%). The secondary prediction (better at k = 1–5) is supported.

Gates and the latent-head prediction:
- **Every sanity gate passes in both seeds:** finite, no collapse, flat far log-variance, the share
  at 0.10, and `w_goal_eff` as designed. So these runs are the M7 reference.
- **The latent-head prediction (near k5 lower) is not supported.** The near heads end where the
  old ones did. The far k10 head is 8% worse.

Calibration improved at long horizons, but stays below nominal: 83–88% inside the 90% interval at
k ≥ 5.

**Three things changed at once:** per-channel latent losses, a 10% pixel share, and a 4× / 2×
smaller goal weight relative to L1 / L2. The short-horizon gain fits the pixel share's 24k result.
The k = 10–20 gain, the calibration change and the far k10 cost are not attributed to any one of them.

## Next (proposals for the user; none run)
1. **M7 ablations can use these runs as their reference.** They inherit `exp_headline_48k.yaml`
   ([decision](../docs/decisions/2026-11-19-m7-ablations-at-48k.md)), and these runs are that
   config at the current defaults.
2. **A nearly free attribution step:** evaluate these runs' step-24,000 checkpoints
   (`jepa_world.eval --run <dir> --checkpoint <dir>/checkpoints/step_0024000.pt --split test
   --split dev`).
   - There is no LR schedule and training is deterministic, so those checkpoints are a 24k run at
     the new defaults.
   - The 24k `m7_pixel_share10` runs differ from them only in `latent_reduce` (and so
     `w_goal_eff`).
   - So the comparison isolates changes 1 + 3 at 24k, for two evals of about 3 minutes and no
     training.
3. **Calibration:** the old note's post-hoc variance rescaling (one scale per horizon, fit on dev)
   still applies. The gap is smaller now: 83–88% inside the 90% interval at k ≥ 5, against 79–82%.
