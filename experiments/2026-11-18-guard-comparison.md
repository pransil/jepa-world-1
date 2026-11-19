# Does the variance guard (sum ×25) cause the near-head regression after far joins?

- **Date:** 2026-11-18   **Lane/branch:** worktree-wt-exp-guard
- **Commits** (all clean):
  - mean ×1 training, and every dev score and readout below: c54bcef;
  - the sum ×25 runs: e95576c, from the first real run, reused (see Reproducibility);
  - the reproducibility check: 5151179.
- **Configs:** sum ×25 arm `configs/exp_first_real_run.yaml`; mean ×1 arm `configs/exp_guard_mean.yaml`.
  Seed 0 is each file, seed 1 is `--set seed=1`.
- **Run dirs** (under `$JEPA_ARTIFACTS/runs/`):
  - sum ×25: `20261118T060158Z-worktree-wt-exp-first-real-run-first-real-s0-f3a0d2` (seed 0) and `…-first-real-s1-72abd4` (seed 1);
  - mean ×1: `20261119T063237Z-worktree-wt-exp-guard-guard-mean-s0-a110c8` (seed 0) and `20261119T064851Z-worktree-wt-exp-guard-guard-mean-s1-0c52c6` (seed 1);
  - reproducibility check: `20261119T062853Z-worktree-wt-exp-guard-guard-repro-s0-9a7b4a`.
  - Each trained run dir holds `latent_dev.json` (the deciding metric) and `readout.json`.
- **Asked for:** proposal 1 in [2026-11-17-first-real-run.md](2026-11-17-first-real-run.md),
  approved by the user 2026-11-18 (via the orchestrator session). The default guard does not
  change here: this note proposes, the user decides.

## Hypothesis
In [2026-11-17-first-real-run.md](2026-11-17-first-real-run.md), the near heads got worse once the
far heads joined at step 2000: k5 relerr 0.17–0.18 → 0.29–0.32 on train batches, in both seeds.
The suspect is the variance guard, which M4 raised from a channel mean at weight 1 to a channel
sum at weight 25 ([decision](../docs/decisions/2026-11-17-far-collapse-guard-sum.md)) to stop the
far NLL collapsing L2/L3. At sum ×25 the guard is about 150 when far joins, so it dominates the
loss, and it pushes on the shared encoder that the near heads read.

**Hypothesis:** with the guard back at mean ×1, and everything else the same (near + far, 8000
steps, far joins after 2000, B = 32, train split), the near heads regress less: k5 relerr on dev
at step 8000 is lower than under sum ×25.

**Deciding metric:** near-head `relerr/k5` on the **dev** split at step 8000.
- Scored with `experiments/latent_relerr.py`: the training loss code on 200 dev batches of 32,
  seed 0, so both arms see the same windows.
- Training only logs relerr on train batches, so "dev over the last 500 steps" isn't available.
  The step-8000 checkpoint on dev is the deciding number. The train-batch mean over steps
  7501–8000 is reported beside it.

**Supported if** both mean ×1 seeds have a lower dev k5 relerr than both sum ×25 seeds.
**Refuted if** the mean ×1 arm's mean is at or above the sum ×25 arm's. Anything in between
(the arms overlap across seeds) is inconclusive.

**Collapse is a result, not a failure.** M4 saw mean ×1 collapse L2/L3 under the far NLL. A run
counts as collapsed if any level's embedding std ends below 10% of its value at step 2000, or the
far log-variance runs toward the −10 bound. A collapsed run's relerr isn't read as a win: relerr
is scale-free, but a near-constant target is easy to predict.

**Secondary:**
- Velocity R² from the 4×4-grid readout (`experiments/probe_readout.py`, `g4` pooled, dev),
  against the existing random-init floors (0.79 dev, both seeds: the floor depends on the
  architecture and seed only, not the loss).
- Far-head health: dev relerr k10/k20, far log-variance, embedding std per level.

**Baselines:** persistence and physics score predictions in world metres. This run has no goal
head, so they are N/A. The in-latent reference is relerr = 1 (predicting the batch mean), and
each arm's own near-only phase (steps 1–2000).

**Budget:** 2 arms × 2 seeds × 8000 steps, about 19 min each on MPS. If the sum ×25 config still
reproduces the first-real-run seeds exactly on current main, those two runs are reused as the
sum ×25 arm and only the mean ×1 runs are trained (see Reproducibility below).

## Reproducibility: the sum ×25 arm is the first real run's two seeds
The first real run trained at e95576c. Main has moved a long way since (M5's aux heads, M6's eval,
M7's top-down and obstacle switches, all off here). So before reusing its runs, I re-ran
`configs/exp_first_real_run.yaml` on current main (5151179, clean) for 2100 steps, seed 0
(`--set train.steps=2100`; there is no learning-rate schedule, so the length changes nothing else).
- **Every logged term of all 2100 steps is bitwise identical** to the original seed 0's
  `metrics.jsonl`: 27 keys × 2100 rows, maximum absolute difference 0.
- That includes steps 2001–2100, after the far heads join.
- Training on MPS is deterministic here, and the code changes since e95576c don't touch this
  configuration.

So the two first-real-run seeds are the sum ×25 arm, and only the mean ×1 arm was trained. Seed 1
wasn't re-run; it is the same code with a different seed.

## Results

### Deciding metric: near-head relerr on DEV at step 8000
`experiments/latent_relerr.py`, 200 dev batches of 32 (seed 0, the same windows for every run), CPU.
The ± is the per-batch standard deviation.

DEV_TABLE

The script is checked against training's own numbers. Scored on **train** batches instead, the
sum ×25 seed 0 checkpoint gives k1 / k5 / k20 relerr 0.082 / 0.315 / 0.628, against 0.079 / 0.318 /
0.637 averaged over its last 500 training steps (`latent_train.json` in that run dir; same script
as c54bcef, run just before committing it).

### Training (train batches, from `metrics.jsonl`)

TRAIN_TABLE

k5 relerr through each run (mean of the last 250 steps of each 1000):

| arm | seed | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k |
|---|---|---|---|---|---|---|---|---|---|
| sum ×25 | 0 | 0.21 | 0.20 | 0.24 | 0.26 | 0.29 | 0.30 | 0.31 | 0.32 |
| sum ×25 | 1 | 0.22 | 0.19 | 0.23 | 0.23 | 0.26 | 0.29 | 0.29 | 0.29 |
| mean ×1 | 0 | 0.43 | 0.49 | 0.52 | 0.53 | 0.53 | 0.53 | 0.53 | 0.54 |
| mean ×1 | 1 | 0.45 | 0.50 | 0.55 | 0.54 | 0.55 | 0.55 | 0.56 | 0.56 |

Sanity, all four runs:
- Every term is finite at every step.
- **sum ×25:** no collapse. L2/L3 std never drops below its step-2000 value; L1 dips to 0.71–0.75 of
  it when far joins, then grows to 1.6–1.7. Far log-variance stays in −1.8…+0.1.
- **mean ×1: M4's collapse comes back, then recovers.**
  - Within 10 steps of far joining, L3's std falls from 0.39–0.40 to 0.065 (seed 0, 17%) and
    0.036 (seed 1, **9%**), and L2's to 44–48%.
  - L3 is back above 0.3 (50-step mean) by step 2966 / 3358, and ends at 0.34–0.35.
  - L1 shrinks too, from 0.43–0.44 to 0.31–0.32.
  - Far log-variance runs to −3.4…−3.9 and stays there (minimum −5.1 / −5.3), against −0.5…+0.1
    under sum ×25.
  - By the pre-set rule (*ends* below 10% of the step-2000 std), neither run counts as collapsed.
    Seed 1's L3 passed through 9% on the way.
- `loss/var` under mean ×1 sits at about 2.0 from step 2000 to the end: the guard is active the
  whole run, at weight 1. Under sum ×25 it falls to 0.17–0.24 by the end.

### Secondary: readouts (4×4 grid and others), R² on dev (test in parentheses)
`experiments/probe_readout.py --device cpu`, the same protocol as the first real run's addendum.
The random-init floors are that addendum's, reused. They depend only on the architecture and the
run's seed, which both arms share.

READOUT_TABLES

## Conclusion
(filled in after the runs)

## Next
(filled in after the runs)
