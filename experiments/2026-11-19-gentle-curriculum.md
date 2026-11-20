# Gentle-world curriculum (H)

**Status:** done, two seeds: **refuted** (worse at k = 20, within at every other k). The rules below were committed before any curriculum run.

## Question

Does training first on a gentle world (smooth rolling, no ledges, the ball starting on the surface)
and then on the full world beat training on the full world throughout, at matched total steps
(48k), on the full-world test set?

The user approved it (relayed by the orchestrator, 2026-11-19).

**Background:**
- **Where the baseline error is.** A baseline-only split of the test windows by regime (the
  orchestrator's `regime_split.py`) gives:
  - at k = 5, smooth rolling is 42% of windows but only 4% of the const-velocity error, while
    bounces are 35% of windows and 78% of it;
  - at k = 20, bounces are 50% of windows and 85% of the error.
- **Where the model's error is** (m6's G read of the reference,
  [error by motion regime](2026-11-19-error-by-motion-regime.md), PR #35):
  - In smooth rolling the goal head is about at par with the fair, context-only
    finite-difference const-velocity baseline at k = 1 (0.99 for s0, 1.09 for s1; 1.05 cm, about
    0.2 px). It beats that baseline from k = 2 on (0.65 at k = 2, 0.36 at k = 5, 0.24 at k = 20).
  - The baselines with the simulator's exact velocity and spin do better in rolling. They are not
    fair comparisons: the model sees only frames.
  - The error lives in bounces and flight. Bounce windows hold 54% of the summed goal error at
    k = 5 and 71% at k = 20, and airborne windows hold 69% at k = 1.

The full world mixes all regimes from step 1. So the premise is a transfer question: does
learning smooth rolling first, where the future is predictable, give the model a better start for
bounces and flight, where its error is?

*(Background corrected 2026-11-20, after seed 0 started. The first version quoted the rolling
error against baselines that are given the exact velocity, which overstated how weak the model is
in rolling. The rules below are unchanged.)*

## What runs

| | reference (done) | H |
|---|---|---|
| config | `configs/exp_headline_48k.yaml` | `configs/exp_gentle_curriculum.yaml` |
| steps 1–12,000 | full world | **gentle world** (`configs/world_gentle.yaml`) |
| steps 12,001–48,000 | full world | full world |
| total steps, stage starts, loss weights, seeds | — | identical to the reference |

- **The reference** is the two runs of
  [the new-defaults 48k note](2026-11-19-headline-48k-new-defaults.md):
  - s0 `20261119T224253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s0-0f2411` (eval
    `20261120T003237Z-test-dev-f36d50.json`);
  - s1 `20261120T003253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s1-ada237` (eval
    `20261120T022242Z-test-dev-8c61ea.json`).
- **The gentle world** (built in PR #38): relief [0.5, 1.5], max_slope 0.2, no ledges, drop 0.
  The ball, camera, contact model and walls are the full world's. Measured beside the full world
  over 200 episodes (`jepa_world/sim/DESIGN.md`, "The gentle world"; stats run
  `20261120T074426Z-worktree-wt-gentle-curriculum-gentle-world-stats-266e1a`):
  - contact rate 0.98 against 0.66;
  - 0.11 bounces per episode against 4.1;
  - after the start settle (from frame 5), 1 bounce and 1 airborne frame in total.

  Its train split is `datasets/episodes-train-0408ce614735` (2000 episodes).
- **The mechanism** is `data.curriculum` (PR #38):
  - Steps 1–12,000 read the gentle world's train split, which has the same terrain and ball seeds
    as the full world's.
  - From step 12,001 the run reads the full world's train split, starting from its first batch.
    So steps 12,001–48,000 see exactly the windows the reference saw at steps 1–36,000, and steps
    1–12,000 see the reference's first 12,000 windows by episode index, in the gentle world.
  - The run's config.yaml keeps the full world at the top level, so eval scores the full-world
    test split.
- **Why 12,000 gentle steps:**
  - All four stage starts (far 2000, goal 4000, pixel 6000) fall inside the gentle phase, so every
    head first learns on smooth rolling.
  - The reference's goal head sits at persistence on training batches until about step 12,000
    (goalratio k20 0.97 over steps 11,001–12,000 in both seeds), then breaks away (0.42 at 24k).
    The gentle phase covers that whole plateau.
  - It leaves 36,000 full-world steps, 1.5× the old 24k headline.
  - The cost is 25% fewer full-world steps. Over its last 12k steps, the reference's
    training-batch goalratio k20 went from 0.353 / 0.359 (steps 35,001–36,000) to 0.328 / 0.338
    (steps 47,001–48,000), about −6%. That is roughly the gain the gentle phase has to beat.
- **Budget:** seed 0 first (about 2 GPU-h). Seed 1 only under the trigger below. At most about
  4 GPU-h, plus the dataset build (done, 92 s).

## Rules (committed before the runs)

Each curriculum run is scored at step 48,000 with
`uv run python -m jepa_world.eval --run <dir> --split test --split dev`, from a clean tree.
`experiments/gentle_curriculum_check.py` prints every table below from the runs' JSON. It reuses
`experiments/m7_ablation_check.py`'s gates and tables, and is committed before the first eval.

### A. Gates

Every curriculum seed must pass these, or nothing else is read. They are m7_ablation_check's A1–A5,
the thresholds of the new-defaults note:
- **A1:** every logged term is finite.
- **A2:** emb_std at L1, L2 and L3 never falls below 50% of its step-2000 value, including after
  the switch.
- **A3:** far logvar k10 and k20 have a last-500-step mean ≥ −3.0, and no step after 2000 is
  below −8.
- **A4:** pix_share is within 1e-4 of 0.10 after step 6000.
- **A5:** w_goal_eff = 0.5/256 after step 4000.

**The switch (reported, not gated):**
- the run's config.yaml has `data.curriculum.steps: 12000`, and its gentle sim merges to
  `configs/world_gentle.yaml`'s;
- the jump in loss/total and near relerr k1–5 at step 12,001;
- the steps until near relerr k1–5 is back within 10% of its level over the reference's first
  steps on the same windows (steps 1–1,000 of the reference, which see the same batches).
- *Amendment (2026-11-20, written after H s0's training metrics were seen, before any eval).* The
  recovery measure above compares against the reference's steps 1–1,000, when the reference is an
  untrained model, so it cannot show recovery. Reported instead: the first 100-step window after
  step 12,000 in which H's near relerr k1–5 mean is within 10% of reference s0's over the same
  steps. The switch is reported, not gated, so no verdict depends on this change.

### B. Deciding: goal head, full-world TEST, ratio to persistence at step 48,000

Reference values (s0 / s1, and the spread):

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| s0 / s1 | 0.2111 / 0.2310 | 0.2055 / 0.2184 | 0.2087 / 0.2208 | 0.2122 / 0.2234 | 0.2184 / 0.2277 | 0.2471 / 0.2506 | 0.3288 / 0.3312 |
| spread | 0.0199 | 0.0129 | 0.0120 | 0.0112 | 0.0093 | 0.0035 | 0.0023 |

The verdict is given per k:
- **better:** every curriculum seed is below both reference seeds;
- **worse:** every curriculum seed is above both;
- **within:** anything else.

With seed 0 alone the same rule applies to one value, and the verdict is labelled
**provisional**.

**The seed-1 trigger.** Run seed 1 if s0 differs from the reference mean by more than the
reference's s0/s1 spread at either:
- k = 20 (spread 0.0023), or
- the mean over k = 1–5 (reference 0.2112 / 0.2243, spread 0.0131).

If neither holds, stop at one seed, because there is no effect beyond seed noise. The hypothesis
is then refuted, since "beats" needs better.

**The hypothesis verdict** (on two seeds if seed 1 ran):
- **supported:** better at k = 20, and worse at no k;
- **refuted:** worse at k = 20, or within at every k;
- **mixed:** anything else (for example better at k = 1–5 but within or worse at k = 20),
  reported per k.

**My prediction:** mixed. I expect better at k = 1–5, because smooth rolling is most of the
windows there and is the regime the gentle world teaches. I expect within or worse at k = 20,
because bounces are 85% of the k = 20 error and the run has 25% fewer full-world steps.

### B2. Co-deciding: per regime (m6's G breakdown, smooth rolling, airborne, bounce)

The orchestrator's question is whether the curriculum improves the rolling rows without hurting
the bounce rows.
- **Metric:** the goal head's per-regime test score at each k, read from the per-regime eval
  files of PR #35, in `<run>/eval/regimes/`. The reference runs already have them.
- **Rule:** the B rule within each regime.
- **Result:**
  - "rolling improved without hurting bounces" holds if smooth rolling is better at a majority of
    k (at least 4 of 7) and bounce is worse at no k;
  - otherwise each row is reported as it falls.
- **Fixed before scoring:** PR #35 isn't merged yet, so the exact field (ratio to persistence if
  it is there, else error in metres) is written into this note before H's first eval, without
  having seen H's per-regime numbers.
- *The field, fixed 2026-11-20, before H's first eval or regimes run.* The field is
  `regimes[k][regime]["ratio"]["persistence"]` from the file written by
  `uv run python -m jepa_world.regimes --run <dir> --split test`, at step 48,000. Regimes are
  `rolling` (the smooth-rolling row above), `airborne` and `bounce`. The reference files are:
  - s0: `eval/regimes/20261120T074239Z-regimes-test-2dd8a9.json`
  - s1: `eval/regimes/20261120T074408Z-regimes-test-b231bd.json`

  Scored by `experiments/gentle_curriculum_check.py`.
- **My prediction:** rolling better at most k, and bounce within.

### C. Secondary: latent heads, full-world training batches, steps 47,001–48,000

Steps 12,001–48,000 are full-world batches. Near relerr k1–5 and far relerr k10/k20 are compared
against the reference's same window with the B rule. There is no prediction.

### D. Reported without a verdict

- The test ratio to const-velocity and to the mean-parameter simulator, and the error in metres.
- Dev ratios, labelled dev.
- Calibration (50% and 90% joint coverage).
- Visibility (std ratio, hidden coverage).
- Probes.
- emb_std and far logvar over training.
- Relerr over training for both phases. Before step 12,001 it is on gentle batches, so it is not
  comparable with the reference's.
- **The reference at step 36,000 (matched full-world steps).** Two evals of the reference's
  step-36,000 checkpoints (about 3 min each), run only when the GPU queue allows. They show
  whether the gentle phase is worth more or less than 12k full-world steps:
  - H below the reference at 48k: worth more;
  - H between the reference at 36k and at 48k: it helps, but less;
  - H above the reference at 36k: it hurts.

## Results

**Verdict: REFUTED, on two seeds.** The pre-registered rule is "worse at k = 20". At matched total
steps (48k), the gentle-world curriculum is **worse at k = 20** on the full-world test set (both
seeds above both reference seeds, +4.5%), and **within at every other k**. It does not improve the
smooth-rolling rows. They are worse at 6 of 7 k.

### Runs and code

| | seed 0 | seed 1 |
|---|---|---|
| run | `20261120T075255Z-worktree-wt-exp-gentle-curriculum-gentle-curriculum-s0-ea4fa1` | `20261120T095402Z-worktree-wt-exp-gentle-curriculum-gentle-curriculum-s1-f8bb0b` |
| trained at | 1cd2d43, clean tree, MPS | ad3b9a4, clean tree, MPS (`--set seed=1`, as the reference's s1) |
| eval (test + dev) | `eval/20261120T094709Z-test-dev-8c79ad.json` (ad3b9a4) | `eval/20261120T114525Z-test-dev-a2352e.json` (38473a9) |
| regimes (test) | `eval/regimes/20261120T094833Z-regimes-test-e1c19f.json` (ad3b9a4) | `eval/regimes/20261120T114645Z-regimes-test-7aa2fc.json` (38473a9) |

- **The training code is the same at both training shas.** From 1cd2d43 to ad3b9a4, nothing
  changes in `train.py`, `models/`, `data/` or `configs/`. 38473a9 differs from ad3b9a4 only in
  this note.
- **The evals compare like for like.** The reference's 48k evals ran at 6fd7059, before an
  `eval.py` refactor from main: helpers extracted, a pixel output off by default, and
  `write_new`. The windows, goal outputs and ratios are computed the same way.
- **The reference's step-36,000 evals** (D) are `eval/20261120T095228Z-test-dev-d1bdc2.json` (s0)
  and `eval/20261120T095352Z-test-dev-57e49d.json` (s1), both at ad3b9a4.
- **GPU time:** about 3.7 h for the two trainings (1h47 each) and their evals, plus 3 min for the
  step-36k evals. That is within the ~4 GPU-h budget.
- **Every table below** is printed by:
  `uv run python experiments/gentle_curriculum_check.py --arm <H s0> --arm <H s1> --ref <ref s0> --ref <ref s1> --ref36`

### A. Gates: PASS, both seeds

- No NaN or inf in any logged term.
- emb_std's lowest value after step 2000 is the step-2000 value itself, in both seeds:
  - s0: 0.76, 0.97 and 0.91 (L1, L2, L3);
  - s1: 0.70, 0.97 and 0.96.
- Over the last 1000 steps, emb_std is 1.69–1.93.
- Far logvar's last-500-step mean is −0.36 / −0.38 (k10) and −0.07 / −0.07 (k20). Its lowest
  value after step 2000 is −3.50 / −3.28.
- pix_share stays within 1.6e-8 of 0.10, and w_goal_eff = 0.5/256 at every step after 4000.

**The switch** (reported, not gated). Both seeds' configs pass the check: steps 12000, and the
merged gentle sim equals `configs/world_gentle.yaml`'s.

| | s0 | s1 | reference s0, same steps |
|---|---|---|---|
| loss/total, 11,901–12,000 → 12,001–12,100 | 3.04 → 7.63 | 3.50 → 7.57 | 6.73 |
| near relerr k1–5 mean, same windows | 0.090 → 0.242 | 0.090 → 0.240 | 0.218 |
| far relerr k20, 12,001–13,000 | 0.83 | 0.80 | 0.64 / 0.67 |
| far relerr k20, 17,001–18,000 | 0.60 | 0.62 | 0.60 / 0.61 |

- Near relerr is within 10% of the reference's from the next 100-step window (12,101) on, in both
  seeds. This is the amended measure.
- Far relerr k20 takes about 5k steps to come level.

### B. Deciding: goal head, full-world TEST, ratio to persistence at step 48,000

| k | reference s0 / s1 | H s0 / s1 | change (mean) | verdict |
|---:|---|---|---:|---|
| 1 | 0.2111 / 0.2310 | 0.2551 / 0.2286 | +9.4% | within |
| 2 | 0.2055 / 0.2184 | 0.2318 / 0.2085 | +3.9% | within |
| 3 | 0.2087 / 0.2208 | 0.2244 / 0.2136 | +2.0% | within |
| 4 | 0.2122 / 0.2234 | 0.2214 / 0.2204 | +1.4% | within |
| 5 | 0.2184 / 0.2277 | 0.2243 / 0.2296 | +1.7% | within |
| 10 | 0.2471 / 0.2506 | 0.2444 / 0.2617 | +1.7% | within |
| 20 | 0.3288 / 0.3312 | 0.3371 / 0.3527 | +4.5% | **worse** |

- **Hypothesis: REFUTED** (worse at k = 20).
- **My prediction was mixed** (better at k1–5, within or worse at k20). k20 came out as
  predicted, but k1–5 did not improve.
- **H's mean is above the reference mean at every k** (+1.4% to +9.4%). Only k = 20 separates
  from the reference's seed range.

### B2. Per regime, TEST, goal ratio to persistence

| k | rolling: ref s0 / s1 → H s0 / s1 | airborne: ref → H | bounce: ref → H |
|---:|---|---|---|
| 1 | 0.122 / 0.135 → 0.209 / 0.157 **worse** | 0.236 / 0.256 → 0.254 / 0.242 within | 0.303 / 0.342 → 0.384 / 0.339 within |
| 2 | 0.115 / 0.117 → 0.152 / 0.121 **worse** | 0.222 / 0.236 → 0.241 / 0.222 within | 0.290 / 0.316 → 0.321 / 0.297 within |
| 3 | 0.115 / 0.111 → 0.133 / 0.115 **worse** | 0.219 / 0.234 → 0.230 / 0.225 within | 0.288 / 0.310 → 0.307 / 0.295 within |
| 4 | 0.116 / 0.109 → 0.124 / 0.118 **worse** | 0.220 / 0.237 → 0.228 / 0.235 within | 0.283 / 0.304 → 0.294 / 0.290 within |
| 5 | 0.118 / 0.108 → 0.120 / 0.124 **worse** | 0.232 / 0.246 → 0.236 / 0.251 within | 0.280 / 0.300 → 0.291 / 0.290 within |
| 10 | 0.142 / 0.131 → 0.135 / 0.152 within | 0.276 / 0.272 → 0.268 / 0.279 within | 0.300 / 0.314 → 0.301 / 0.320 within |
| 20 | 0.215 / 0.210 → 0.225 / 0.238 **worse** | 0.278 / 0.288 → 0.299 / 0.319 **worse** | 0.396 / 0.400 → 0.401 / 0.417 **worse** |

- **Rolling** is better at 0 of 7 k and worse at 6.
- **Bounce** is worse only at k = 20.
- So "rolling improved without hurting bounces" is **False**.
- **Rolling is the regime where the curriculum loses most.** At k1 in rolling, H's error is 1.48×
  the context-only finite-difference const-velocity baseline's (mean over seeds), against the
  reference's 1.04×.
- **My prediction was wrong here** ("rolling better at most k, bounce within"). The bounce half
  held except at k20.

### C. Latent relerr, full-world training batches, steps 47,001–48,000 (no prediction)

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| ref s0 / s1 | 0.1376 / 0.1375 | 0.2136 / 0.2114 | 0.2896 / 0.2837 | 0.3509 / 0.3455 | 0.4020 / 0.3966 | 0.5666 / 0.5490 | 0.5532 / 0.5512 |
| H s0 / s1 | 0.1361 / 0.1365 | 0.2091 / 0.2093 | 0.2802 / 0.2837 | 0.3406 / 0.3438 | 0.3911 / 0.3925 | 0.5636 / 0.5443 | 0.5470 / 0.5645 |
| verdict | better | better | better | better | better | within | within |

- The near latent heads are 1–3% lower in both seeds.
- So the curriculum does not hurt the encoder's near-horizon latent prediction. If anything it
  helps.
- The goal head's test score doesn't follow (B).

### D. Reported without a verdict

- **Matched full-world steps.** Does H at 48k (36k full-world steps) beat the reference at 36k?

  | k | ref @36k s0 / s1 | ref @48k s0 / s1 | H @48k s0 / s1 | vs ref@36k | vs ref@48k |
  |---:|---|---|---|---|---|
  | 1 | 0.2361 / 0.3193 | 0.2111 / 0.2310 | 0.2551 / 0.2286 | within | within |
  | 2 | 0.2203 / 0.2680 | 0.2055 / 0.2184 | 0.2318 / 0.2085 | within | within |
  | 3 | 0.2217 / 0.2582 | 0.2087 / 0.2208 | 0.2244 / 0.2136 | within | within |
  | 4 | 0.2264 / 0.2603 | 0.2122 / 0.2234 | 0.2214 / 0.2204 | better | within |
  | 5 | 0.2328 / 0.2677 | 0.2184 / 0.2277 | 0.2243 / 0.2296 | better | within |
  | 10 | 0.2529 / 0.2871 | 0.2471 / 0.2506 | 0.2444 / 0.2617 | within | within |
  | 20 | 0.3444 / 0.3663 | 0.3288 / 0.3312 | 0.3371 / 0.3527 | within | worse |

  - At k = 20 the means are:
    - reference at 36k: 0.355;
    - H at 48k: 0.345;
    - reference at 48k: 0.330.
  - So the 12k gentle steps moved k = 20 by −3.0% against the reference at 36k, inside its seed
    range. 12k more full-world steps moved it by −7.1%.
  - That is the pre-registered "between: it helps, but less", with the "helps" not separable from
    seed spread.
  - At k = 4–5, H is better than the reference at 36k and level with it at 48k.
  - The reference's 36k seeds are far apart at k1 (0.236 vs 0.319), so k1–3 can't separate.
- **Calibration, TEST, joint coverage** (an observation, not a cause):

  | k | 50%: ref s0 / s1 | 50%: H s0 / s1 | 90%: ref s0 / s1 | 90%: H s0 / s1 |
  |---:|---|---|---|---|
  | 1 | 0.58 / 0.61 | 0.29 / 0.53 | 0.90 / 0.91 | 0.84 / 0.89 |
  | 2 | 0.49 / 0.54 | 0.33 / 0.54 | 0.86 / 0.90 | 0.81 / 0.88 |
  | 5 | 0.47 / 0.52 | 0.44 / 0.50 | 0.83 / 0.87 | 0.86 / 0.87 |
  | 10 | 0.47 / 0.53 | 0.58 / 0.52 | 0.85 / 0.87 | 0.90 / 0.87 |
  | 20 | 0.44 / 0.56 | 0.65 / 0.53 | 0.83 / 0.88 | 0.92 / 0.89 |

  - Seed 0's goal head is overconfident at k1–2, with 50% coverage of 0.29 and 0.33.
  - Seed 1's isn't (0.53 and 0.54). So it is not a consistent effect of the curriculum.
- **Dev** (development only), ratio to persistence:

  | k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
  |---|---|---|---|---|---|---|---|
  | ref s0 / s1 | 0.216 / 0.231 | 0.212 / 0.217 | 0.215 / 0.218 | 0.216 / 0.221 | 0.220 / 0.224 | 0.246 / 0.243 | 0.330 / 0.319 |
  | H s0 / s1 | 0.264 / 0.230 | 0.246 / 0.212 | 0.238 / 0.213 | 0.234 / 0.215 | 0.237 / 0.221 | 0.257 / 0.250 | 0.340 / 0.349 |

  Dev agrees with test: H is above both reference seeds at k = 10 and 20, and seed 0 is the high
  one at k = 1–5.
- **Test, other baselines.**
  - Ratio to const-velocity: H's mean is 1.39 at k1 (the reference's mean is 1.27), 0.40 at k5
    (0.39) and 0.28 at k20 (0.27).
  - The error in metres at k20 is 0.577 / 0.604.
- **Probes,** TEST:
  - Pooled pos_cam RMSE is within at every level.
  - Friction and restitution are better in both seeds:
    - restitution R² 0.126 / 0.099, against 0.047 / 0.055;
    - friction RMSE 0.2687 / 0.2657, against 0.2696 / 0.2700.
  - Both stay near chance (R² ≤ 0.13), so this is a small shift in a weak probe.
- **Goal ratio on training batches,** steps 44,001–48,000:
  - k1: 0.217 / 0.234, against 0.227 / 0.263;
  - k20: 0.342 / 0.352, against 0.332 / 0.340.

  The k20 deficit on test is already present on the training batches.

### What seed 0 alone said (superseded)

Seed 0 alone read worse at k1–3 as well, with rolling worse and bounce worse at k1, k2 and k20,
and k1–2 overconfidence. Seed 1 is close to the reference at k1–3 (0.229 / 0.209 / 0.214), so
those three verdicts became within, and the overconfidence did not repeat. k = 20 is worse in both
seeds, and rolling is worse in both.

## Conclusion

**Refuted.** On the full-world test set, 12k gentle-world steps followed by 36k full-world steps do
not beat 48k full-world steps:
- **k = 20 is worse** in both seeds (0.337 / 0.353 against 0.329 / 0.331).
- **Every other k is within** the reference's seed range.
- **The smooth-rolling rows** are worse at 6 of 7 k, which is the opposite of the premise.
- **Against the reference at 36k** (the same full-world step count), H is better at k4–5 and
  level elsewhere. So the gentle phase is worth something at the middle horizons, but less than
  the same number of full-world steps at k = 20.
- **The latent near heads** come out 1–3% better on training batches. The goal head's test score
  doesn't follow.

## Next step

No follow-up is proposed from this note. The curriculum stays off (`data.curriculum: null`, the
default).

The matched-steps table (D) points to the cause of the k = 20 cost: it is mostly the 25% of
full-world steps the curriculum gives up, not harm from the gentle phase. That is a reading of
two seeds plus two reference checkpoints, not a test.

If gentle data is revisited, the direct test is 12k gentle steps followed by the full 48k
full-world steps (60k total), scored against the reference at 48k. That is a new experiment and
the user's call.
