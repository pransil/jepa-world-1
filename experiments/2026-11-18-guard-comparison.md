# Does the variance guard (sum ×25) cause the near-head regression after far joins?

- **Date:** 2026-11-18   **Lane/branch:** worktree-wt-exp-guard   **Commit:** (filled in after the runs)
- **Configs:** sum ×25 arm `configs/exp_first_real_run.yaml`; mean ×1 arm `configs/exp_guard_mean.yaml`.
  Seed 0 is each file, seed 1 is `--set seed=1`.
- **Run dirs:** (filled in after the runs)
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

## Results
(filled in after the runs)

## Conclusion
(filled in after the runs)

## Next
(filled in after the runs)
