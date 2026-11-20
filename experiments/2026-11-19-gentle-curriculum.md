# Gentle-world curriculum (H)

**Status:** pre-registered. The rules below were committed before any curriculum run.

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
- **Where the model is weak.** m6's first per-regime read of the reference (G; test, s0, and s1
  agrees) found the model weakest in the easy regime:
  - In smooth-rolling windows its k = 1 error is 1.05 cm. That is 1.8× const-velocity with the
    true velocity, and 11× the mean-parameter simulator; it stays 5.8–8× the simulator at every
    k.
  - In bounce windows it beats const-velocity at every k.

The full world mixes all regimes from step 1. The premise is that a model which first learns
smooth rolling, where the future is predictable, gets a better start, and that the easy regime
isn't mastered now.

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

(After the runs.)
