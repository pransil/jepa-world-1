# Gentle-world curriculum (H)

**Status:** seed 0 scored (provisional, one seed); seed 1 running. The rules below were committed before any curriculum run.

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

### Seed 0: provisional, one seed (seed 1 is running)

**Runs and code:**
- **Run:** `20261120T075255Z-worktree-wt-exp-gentle-curriculum-gentle-curriculum-s0-ea4fa1`, trained
  at 1cd2d43 (clean tree, MPS, 48,000 steps).
- **Its eval files** are both at ad3b9a4 (clean tree):
  - eval: `eval/20261120T094709Z-test-dev-8c79ad.json`
  - regimes: `eval/regimes/20261120T094833Z-regimes-test-e1c19f.json`
- **The training code is the same at both shas.** From 1cd2d43 to ad3b9a4, nothing changes in
  `train.py`, `models/`, `data/` or `configs/`. ad3b9a4 adds main's regimes and calibration
  modules and an `eval.py` refactor: helpers extracted, a pixel output off by default, and
  `write_new`.
- **The reference's 48k evals ran at 6fd7059,** before that refactor. The windows, the goal
  outputs and the ratios are computed the same way, so the evals compare like for like.
- **The reference's step-36,000 evals** (D) ran at ad3b9a4:
  - s0: `eval/20261120T095228Z-test-dev-d1bdc2.json`
  - s1: `eval/20261120T095352Z-test-dev-57e49d.json`
- **Every table below** is printed by:
  `uv run python experiments/gentle_curriculum_check.py --arm <H s0> --ref <ref s0> --ref <ref s1> --ref36`

**A. Gates: PASS.**
- No NaN or inf in any logged term.
- After step 2000, emb_std's lowest values are 0.76, 0.97 and 0.91 (L1, L2, L3), the step-2000
  values themselves. Over the last 1000 steps it is 1.73, 1.93 and 1.87.
- Far logvar's last-500-step mean is −0.36 at k10 and −0.07 at k20. Its lowest value after step
  2000 is −3.50.
- pix_share stays within 1.6e-8 of 0.10, and w_goal_eff = 0.5/256 at every step after 4000.

**The switch** (reported, not gated):
- **Config:** the config check passes (steps 12000; the merged gentle sim equals
  `configs/world_gentle.yaml`'s).
- **loss/total:** 3.04 over steps 11,901–12,000, then 7.63 over 12,001–12,100. The reference has
  6.73 over the same steps.
- **Near relerr k1–5 mean:** 0.090, then 0.242, against the reference's 0.218 over the same
  steps. It is within 10% of the reference's from the next window (12,101–12,200) on (the
  amended measure).
- **Far relerr k20** takes longer to recover: 0.83 over steps 12,001–13,000, against the
  reference's 0.64 / 0.67. It is level with it by steps 17,001–18,000 (0.60 against 0.60 / 0.61).

**B. Deciding: goal head, full-world TEST, ratio to persistence at step 48,000** (provisional):

| k | reference s0 / s1 | H s0 | change vs ref mean | verdict |
|---:|---|---:|---:|---|
| 1 | 0.2111 / 0.2310 | 0.2551 | +15.4% | worse |
| 2 | 0.2055 / 0.2184 | 0.2318 | +9.4% | worse |
| 3 | 0.2087 / 0.2208 | 0.2244 | +4.5% | worse |
| 4 | 0.2122 / 0.2234 | 0.2214 | +1.7% | within |
| 5 | 0.2184 / 0.2277 | 0.2243 | +0.6% | within |
| 10 | 0.2471 / 0.2506 | 0.2444 | −1.8% | better |
| 20 | 0.3288 / 0.3312 | 0.3371 | +2.1% | worse |

- **Seed-1 trigger: fired on both conditions.**
  - At k20, |s0 − ref mean| = 0.0071, against a spread of 0.0023.
  - On the k1–5 mean (0.2314), |s0 − ref mean| = 0.0137, against a spread of 0.0131.
  - Seed 1 is `20261120T095402Z-worktree-wt-exp-gentle-curriculum-gentle-curriculum-s1-f8bb0b`,
    started at ad3b9a4 (clean tree).
- **Hypothesis, provisional: REFUTED** (worse at k = 20). My prediction was mixed (better at
  k1–5); it was wrong at k1–3 too.

**B2. Per regime, TEST, goal ratio to persistence** (provisional; `share` is the regime's share of
the windows at that k):

| k | rolling: ref s0 / s1 → H | airborne: ref → H | bounce: ref → H |
|---:|---|---|---|
| 1 | 0.122 / 0.135 → 0.209 **worse** | 0.236 / 0.256 → 0.254 within | 0.303 / 0.342 → 0.384 **worse** |
| 2 | 0.115 / 0.117 → 0.152 **worse** | 0.222 / 0.236 → 0.241 **worse** | 0.290 / 0.316 → 0.321 **worse** |
| 3 | 0.115 / 0.111 → 0.133 **worse** | 0.219 / 0.234 → 0.230 within | 0.288 / 0.310 → 0.307 within |
| 4 | 0.116 / 0.109 → 0.124 **worse** | 0.220 / 0.237 → 0.228 within | 0.283 / 0.304 → 0.294 within |
| 5 | 0.118 / 0.108 → 0.120 **worse** | 0.232 / 0.246 → 0.236 within | 0.280 / 0.300 → 0.291 within |
| 10 | 0.142 / 0.131 → 0.135 within | 0.276 / 0.272 → 0.268 better | 0.300 / 0.314 → 0.301 within |
| 20 | 0.215 / 0.210 → 0.225 **worse** | 0.278 / 0.288 → 0.299 **worse** | 0.396 / 0.400 → 0.401 **worse** |

- **Rolling** is better at 0 of 7 k.
- **Bounce** is worse at k = 1, 2 and 20.
- So "rolling improved without hurting bounces" is **False**.
- My prediction ("rolling better at most k") was wrong. Rolling is where H loses most: k1 rolling
  is 0.209 against 0.122 / 0.135. Against the context-only finite-difference const-velocity
  baseline, H's k1 rolling ratio is 1.69 (the reference's mean is 1.04).

**C. Latent relerr, full-world training batches, steps 47,001–48,000** (no prediction). H is
below both reference seeds at k = 1–5 and k = 20, and within at k = 10:

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| ref s0 / s1 | 0.1376 / 0.1375 | 0.2136 / 0.2114 | 0.2896 / 0.2837 | 0.3509 / 0.3455 | 0.4020 / 0.3966 | 0.5666 / 0.5490 | 0.5532 / 0.5512 |
| H s0 | 0.1361 | 0.2091 | 0.2802 | 0.3406 | 0.3911 | 0.5636 | 0.5470 |

The margins are 1–3%. Each reference value is one seed's mean over 1000 steps.

**D. Reported without a verdict:**
- **Matched full-world steps.** The question: is H at 48k (36k full-world steps) better than the
  reference at 36k?

  | k | ref @36k s0 / s1 | ref @48k s0 / s1 | H @48k | vs ref@36k | vs ref@48k |
  |---:|---|---|---:|---|---|
  | 1 | 0.2361 / 0.3193 | 0.2111 / 0.2310 | 0.2551 | within | worse |
  | 2 | 0.2203 / 0.2680 | 0.2055 / 0.2184 | 0.2318 | within | worse |
  | 3 | 0.2217 / 0.2582 | 0.2087 / 0.2208 | 0.2244 | within | worse |
  | 4 | 0.2264 / 0.2603 | 0.2122 / 0.2234 | 0.2214 | better | within |
  | 5 | 0.2328 / 0.2677 | 0.2184 / 0.2277 | 0.2243 | better | within |
  | 10 | 0.2529 / 0.2871 | 0.2471 / 0.2506 | 0.2444 | better | better |
  | 20 | 0.3444 / 0.3663 | 0.3288 / 0.3312 | 0.3371 | better | worse |

  By the pre-registered reading:
  - at k = 20 the gentle phase helps, but less than 12k more full-world steps;
  - at k = 10 it is worth more;
  - at k = 1–3 H is inside the reference's 36k seed range, which is wide (0.236–0.319 at k = 1).
- **Calibration, TEST, joint coverage** (an observation, not a cause):

  | k | 50%: ref s0 / s1 | 50%: H | 90%: ref s0 / s1 | 90%: H |
  |---:|---|---:|---|---:|
  | 1 | 0.58 / 0.61 | 0.29 | 0.90 / 0.91 | 0.84 |
  | 2 | 0.49 / 0.54 | 0.33 | 0.86 / 0.90 | 0.81 |
  | 3 | 0.47 / 0.53 | 0.38 | 0.83 / 0.88 | 0.83 |
  | 5 | 0.47 / 0.52 | 0.44 | 0.83 / 0.87 | 0.86 |
  | 10 | 0.47 / 0.53 | 0.58 | 0.85 / 0.87 | 0.90 |
  | 20 | 0.44 / 0.56 | 0.65 | 0.83 / 0.88 | 0.92 |

  - At k = 1–2, H's goal head is overconfident.
  - At k = 20 it leans the other way.
- **Dev** (development only) has H above both reference seeds at every k. So dev agrees with test
  at k1–3 and k20, but not at k10, where test had H better:
  - k1 0.264 against 0.216 / 0.231;
  - k10 0.257 against 0.246 / 0.243;
  - k20 0.340 against 0.330 / 0.319.
- **Test ratio to const-velocity:** 1.46 at k1 (the reference's mean is 1.27); 0.28 at k20
  (0.27).
- **Probes** (pooled pos_cam RMSE): 0.163, against 0.158 / 0.174 (within). The hidden-parameter
  probes (radius, friction, restitution) are better than both reference seeds. Restitution R²
  is 0.126, against 0.047 / 0.055.
- **Goal ratio on training batches,** steps 47,001–48,000:
  - k1: 0.213, against 0.224 / 0.259;
  - k20: 0.340, against 0.328 / 0.338.

  That is better at k1 on the training batches, but worse at k1 on test.
