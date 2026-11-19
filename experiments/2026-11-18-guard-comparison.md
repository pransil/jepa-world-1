# Does the variance guard (sum ×25) cause the near-head regression after far joins?

- **Date:** 2026-11-18   **Lane/branch:** worktree-wt-exp-guard
- **Commits** (all clean):
  - mean ×1 training, and every dev score and readout below: c54bcef. The exceptions are mean ×1
    seed 1's dev score and readout, run at 8f2e4f8, whose only change from c54bcef is this note.
    Their first run was flagged dirty: its metadata was recorded while an uncommitted draft of
    this note sat in the tree. The clean re-run gives identical numbers. `latent_relerr.py` now
    records its metadata at the start of a run, as `probe_readout.py` does;
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

| arm | seed | relerr k1 | k2 | k3 | k4 | **k5** | k10 | k20 | far logvar k10 / k20 | emb std L1 / L2 / L3 |
|---|---|---|---|---|---|---|---|---|---|---|
| sum ×25 | 0 | 0.075 | 0.142 | 0.206 | 0.258 | **0.302** ± 0.054 | 0.467 | 0.666 | −0.53 / +0.09 | 1.64 / 1.46 / 1.40 |
| sum ×25 | 1 | 0.067 | 0.133 | 0.196 | 0.245 | **0.291** ± 0.051 | 0.462 | 0.672 | −0.51 / −0.17 | 1.68 / 1.49 / 1.38 |
| mean ×1 | 0 | 0.188 | 0.299 | 0.395 | 0.472 | **0.536** ± 0.054 | 0.742 | 0.746 | −3.89 / −3.54 | 0.31 / 0.31 / 0.34 |
| mean ×1 | 1 | 0.180 | 0.289 | 0.395 | 0.484 | **0.555** ± 0.054 | 0.728 | 0.725 | −3.60 / −3.65 | 0.31 / 0.34 / 0.33 |

**mean ×1 is worse at every horizon, near and far, in both seeds.** k5: 0.536 / 0.555 against
0.302 / 0.291, about 1.8×.

The script is checked against training's own numbers. Scored on **train** batches instead, the
sum ×25 seed 0 checkpoint gives k1 / k5 / k20 relerr 0.082 / 0.315 / 0.628, against 0.079 / 0.318 /
0.637 averaged over its last 500 training steps (`latent_train.json` in that run dir; same script
as c54bcef, run just before committing it).

### Training (train batches, from `metrics.jsonl`)

Window means. "2001–2010" is the first 10 steps after far joins.

| arm | seed | steps | relerr k1 | k5 | k10 | k20 | far logvar k10 / k20 | emb std L1 / L2 / L3 | loss/var |
|---|---|---|---|---|---|---|---|---|---|
| sum ×25 | 0 | 1501–2000, near only | 0.047 | 0.199 | (inactive) | (inactive) | ≈0 / ≈0 | 1.28 / 0.39 / 0.38 | 0.18 |
| | | 2001–2010 | 0.186 | 0.320 | 1.438 | 1.480 | −0.18 / −0.35 | 1.19 / 0.55 / 0.56 | 186 |
| | | 3501–4000 | 0.065 | 0.263 | 0.476 | 0.685 | −0.77 / −0.10 | 1.47 / 1.27 / 1.25 | 0.81 |
| | | 7501–8000 | 0.079 | 0.318 | 0.486 | 0.637 | −0.46 / −0.03 | 1.62 / 1.45 / 1.40 | 0.17 |
| sum ×25 | 1 | 1501–2000, near only | 0.047 | 0.195 | (inactive) | (inactive) | ≈0 / ≈0 | 1.28 / 0.41 / 0.39 | 0.18 |
| | | 2001–2010 | 0.115 | 0.218 | 1.396 | 1.408 | −0.11 / −0.30 | 1.23 / 0.55 / 0.55 | 184 |
| | | 3501–4000 | 0.048 | 0.222 | 0.541 | 0.746 | −0.68 / +0.01 | 1.48 / 1.29 / 1.27 | 1.01 |
| | | 7501–8000 | 0.069 | 0.291 | 0.479 | 0.664 | −0.42 / +0.03 | 1.66 / 1.48 / 1.38 | 0.24 |
| mean ×1 | 0 | 1501–2000, near only | 0.170 | 0.483 | (inactive) | (inactive) | ≈0 / ≈0 | 0.44 / 0.37 / 0.39 | 0.58 |
| | | 2001–2010 | 0.174 | 0.506 | 1.578 | 1.507 | −0.22 / −0.40 | 0.44 / 0.24 / 0.14 | 2.21 |
| | | 3501–4000 | 0.185 | 0.528 | 0.669 | 0.694 | −3.61 / −3.54 | 0.36 / 0.33 / 0.34 | 2.03 |
| | | 7501–8000 | 0.188 | 0.540 | 0.732 | 0.679 | −3.85 / −3.63 | 0.31 / 0.32 / 0.35 | 2.06 |
| mean ×1 | 1 | 1501–2000, near only | 0.182 | 0.498 | (inactive) | (inactive) | ≈0 / ≈0 | 0.43 / 0.41 / 0.40 | 0.59 |
| | | 2001–2010 | 0.192 | 0.492 | 1.457 | 1.454 | −0.11 / −0.35 | 0.42 / 0.27 / 0.12 | 2.22 |
| | | 3501–4000 | 0.186 | 0.547 | 0.735 | 0.645 | −3.35 / −3.81 | 0.36 / 0.35 / 0.31 | 2.03 |
| | | 7501–8000 | 0.181 | 0.555 | 0.733 | 0.653 | −3.61 / −3.74 | 0.32 / 0.35 / 0.34 | 2.04 |

Before step 2001 the far heads exist and are logged, but they don't train, so their numbers there
are left out ("inactive").

k5 relerr through each run (mean of the last 250 steps of each 1000):

| arm | seed | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k |
|---|---|---|---|---|---|---|---|---|---|
| sum ×25 | 0 | 0.21 | 0.20 | 0.24 | 0.26 | 0.29 | 0.30 | 0.31 | 0.32 |
| sum ×25 | 1 | 0.22 | 0.19 | 0.23 | 0.23 | 0.26 | 0.29 | 0.29 | 0.29 |
| mean ×1 | 0 | 0.43 | 0.49 | 0.52 | 0.53 | 0.53 | 0.53 | 0.53 | 0.54 |
| mean ×1 | 1 | 0.45 | 0.50 | 0.55 | 0.54 | 0.55 | 0.55 | 0.56 | 0.56 |

Sanity, all four runs:
- Every term is finite at every step.
- **sum ×25:** no collapse.
  - L2/L3 std stays at or above 0.98 of its step-2000 value.
  - L1 dips to 0.71–0.75 of its step-2000 value when far joins, then grows to 1.6–1.7.
  - Far log-variance stays in −1.8…+0.1.
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

Each row is the pooled probe (L1–L3) for g1, g2 and g4, and the single level for g8.

**vel** (the ball's world velocity)

| encoder | seed | g1 | g2 | **g4** | g8 L1 | g8 L2 |
|---|---|---|---|---|---|---|
| sum ×25 | 0 | 0.63 (0.63) | 0.73 (0.75) | **0.77** (0.79) | 0.76 (0.78) | 0.72 (0.72) |
| sum ×25 | 1 | 0.63 (0.61) | 0.73 (0.74) | **0.77** (0.79) | 0.77 (0.78) | 0.73 (0.72) |
| mean ×1 | 0 | 0.75 (0.78) | 0.81 (0.82) | **0.81** (0.83) | 0.79 (0.82) | 0.78 (0.80) |
| mean ×1 | 1 | 0.75 (0.78) | 0.80 (0.82) | **0.81** (0.83) | 0.79 (0.82) | 0.79 (0.81) |
| random init | 0 | 0.72 (0.72) | 0.79 (0.81) | **0.79** (0.81) | 0.79 (0.81) | 0.76 (0.79) |
| random init | 1 | 0.72 (0.73) | 0.78 (0.80) | **0.79** (0.80) | 0.78 (0.80) | 0.77 (0.79) |

**vel_rel** (the ball's velocity relative to the camera, i.e. its motion in the image)

| encoder | seed | g1 | g2 | g4 | g8 L1 | g8 L2 |
|---|---|---|---|---|---|---|
| sum ×25 | 0 | 0.48 (0.50) | 0.60 (0.62) | 0.64 (0.68) | 0.61 (0.67) | 0.56 (0.60) |
| sum ×25 | 1 | 0.49 (0.50) | 0.58 (0.62) | 0.63 (0.67) | 0.61 (0.65) | 0.55 (0.58) |
| mean ×1 | 0 | 0.58 (0.64) | 0.67 (0.71) | 0.69 (0.73) | 0.66 (0.71) | 0.63 (0.67) |
| mean ×1 | 1 | 0.56 (0.62) | 0.66 (0.69) | 0.67 (0.71) | 0.65 (0.70) | 0.63 (0.68) |
| random init | 0 | 0.60 (0.64) | 0.66 (0.69) | 0.65 (0.69) | 0.65 (0.71) | 0.61 (0.67) |
| random init | 1 | 0.59 (0.63) | 0.65 (0.70) | 0.65 (0.69) | 0.63 (0.70) | 0.62 (0.68) |

**cam_vel** (the camera's velocity: the background scroll)

| encoder | seed | g1 | g2 | g4 | g8 L1 | g8 L2 |
|---|---|---|---|---|---|---|
| sum ×25 | 0 | 0.80 (0.77) | 0.90 (0.89) | 0.92 (0.92) | 0.91 (0.91) | 0.91 (0.89) |
| sum ×25 | 1 | 0.79 (0.74) | 0.89 (0.88) | 0.92 (0.92) | 0.91 (0.91) | 0.90 (0.89) |
| mean ×1 | 0 | 0.91 (0.93) | 0.94 (0.96) | 0.95 (0.97) | 0.94 (0.95) | 0.95 (0.95) |
| mean ×1 | 1 | 0.91 (0.92) | 0.94 (0.96) | 0.95 (0.97) | 0.94 (0.95) | 0.95 (0.96) |
| random init | 0 | 0.87 (0.85) | 0.94 (0.95) | 0.94 (0.95) | 0.94 (0.95) | 0.94 (0.95) |
| random init | 1 | 0.87 (0.85) | 0.93 (0.94) | 0.94 (0.95) | 0.93 (0.94) | 0.93 (0.94) |

**pos** (world position) stays unreadable for every encoder: dev R² ≤ 0.12 at every grid. mean ×1
overfits the training terrains more at g4: test −0.26 in both seeds, against −0.07…−0.09 under
sum ×25.

**The readouts reverse the deciding metric.**
- mean ×1's embeddings carry *more* linearly readable motion than sum ×25's, for every motion
  target, at every grid, in both seeds.
- mean ×1 matches or beats the random-init floor in every cell, where sum ×25 never reached it.
  The clearest margins:
  - g1: velocity 0.75 against 0.72, cam_vel 0.91 against 0.87;
  - g4: velocity 0.81 against 0.79, vel_rel 0.67–0.69 against 0.65.
- At finer grids the margin shrinks to 0.00–0.02 (cam_vel g8: 0.94 against 0.93–0.94). So "beats
  random" is a small effect, though it holds in both seeds and on test.

## Conclusion
**Refuted, on the deciding metric.** With the guard at mean ×1, near-head k5 relerr on dev at step
8000 is 0.536 / 0.555, against 0.302 / 0.291 under sum ×25. It is worse, not better, in both seeds,
and at every other horizon too.

The guard is still involved in the near-head slip, just not the way the hypothesis said:
- **The slip after far joins is smaller under mean ×1.** On train batches, k5 rises 0.49–0.50 →
  0.54–0.56 from step 2000 to 8000 (+10–12%), against 0.19–0.20 → 0.29–0.32 under sum ×25
  (+53–60%).
- **But mean ×1 starts far worse.** Its near-only phase already ends at k5 0.48–0.50, against
  0.20, and it is already rising before far joins (0.43–0.45 at step 1000).
- So the strong guard makes the near heads much better in the near-only phase, and part of that
  advantage erodes once far joins. It doesn't create the slip from nothing.

**relerr doesn't compare representations.** It measures each arm's predictions against its own
embedding's batch mean. sum ×25's embeddings are 4–5× wider (std 1.4–1.7 against 0.31–0.35),
and every channel is pushed to vary. mean ×1's narrower embeddings may hold variation that is
harder to predict. The readouts are the representation-independent comparison, and they point the
other way:
- mean ×1's embeddings hold more linearly readable motion than sum ×25's: velocity R² (dev, g4)
  0.81 against 0.77, and 0.75 against 0.63 at g1.
- They reach or pass the random-init floor (0.79 at g4, 0.72 at g1), which sum ×25's never did in
  the [first real run](2026-11-17-first-real-run.md). That is consistent with that note's
  suspicion that the guard pushes trained features toward whatever varies most across samples
  (appearance more than motion). It doesn't prove it.

**mean ×1 brings back M4's far-head problem.**
- L3's spread fell to 9–17% of its near-only value within 10 steps of far joining. It recovered
  within about 1000–1400 steps.
- The far log-variance runs to about −3.7 and stays there, and far relerr is worse: k10
  0.73–0.74 against 0.46–0.47.
- That is the behaviour the [guard decision](../docs/decisions/2026-11-17-far-collapse-guard-sum.md)
  was made to stop, and SPEC §7.2 calibrates those variances.

Net: neither guard setting is good at everything.
- sum ×25: good latent prediction and healthy far heads, but poorer linear readouts.
- mean ×1: better readouts, but worse latent prediction and the far collapse symptoms.

The default stays sum ×25. That is the user's decision; nothing here changes it.

## Next (proposals for the user; none run)
1. **Judge the guard by the goal head, not by relerr or probes.**
   - The question is whether mean ×1's more readable embeddings give better world-coordinate
     predictions.
   - Setup: the headline config at mean ×1, 24k steps, 2 seeds (about 2.1 GPU-h). Compare with
     the 24k headline on the k = 20 test ratio to persistence (0.430).
2. **An intermediate guard:** sum ×1, or mean ×25, at the near + far 8000-step budget, 2 seeds
   (about 1 GPU-h for one setting).
   - Is there a strength that keeps the far heads healthy (no L3 dip, log-variance near −1) and
     still reaches the random floor on readouts?
3. **Check the far heads' calibration under mean ×1.**
   - Its log-variance (−3.7) is far below sum ×25's (−0.5). Whether those variances are well
     calibrated in latent space isn't measured.
   - `experiments/latent_relerr.py` could add the z-score coverage of the far heads on dev. That
     is a small script change, with no new run.
