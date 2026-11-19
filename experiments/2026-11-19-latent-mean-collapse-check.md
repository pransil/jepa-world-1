# Does averaging the latent losses over channels train near + far without collapse? (latent_reduce: mean)

- **Date:** 2026-11-19   **Lane/branch:** worktree-wt-latent-reduce   **Commit:** (filled in at run time)
- **Config:** `configs/exp_latent_mean.yaml`, which is `configs/exp_first_real_run.yaml` (near + far,
  8000 steps, far joins after 2000, B = 32, train split) with only `loss.latent_reduce: mean`. The
  guard follows that reduction, at sum ×25's ratio (`var_reduce: sum`, w_var 25). Seed 0 is the file,
  seed 1 is `--set seed=1`.
- **Comparison arm, sum ×25:** the first real run's two seeds
  (`20261118T060158Z-worktree-wt-exp-first-real-run-first-real-s0-f3a0d2`, `…-s1-72abd4`), as
  scored in [2026-11-18-guard-comparison.md](2026-11-18-guard-comparison.md). They are reused if a
  2100-step re-run of `configs/exp_first_real_run.yaml` (seed 0) at this commit repeats their
  `metrics.jsonl` bitwise.
- **Asked for:** the orchestrator (recover-lanes-after-restart), for the user on 2026-11-19. The
  user approved the direction, "normalize the latent losses". The design is in the
  [decision](../docs/decisions/2026-11-19-latent-loss-normalization.md), proposed until this passes.

## Hypothesis
Under `mean`, every level's guard-to-latent ratio is sum ×25's, while the far terms' weight against
the near terms drops 2× (k = 10) and 4× (k = 20). So near + far trains without M4's collapse, and
the near heads regress no more than under sum ×25.

**Pass rule** (the orchestrator's, made exact here before any run; all three, in both seeds):
1. **No collapse:** L3's `emb_std` (train batches, every step) never falls below 50% of its
   step-2000 value during steps 2001–8000.
2. **Far log-variance doesn't run away:** for both `logvar/k10` and `logvar/k20`, the mean over steps
   7501–8000 is ≥ −3.0, and no step is below −8 (80% of the −10 bound).
   - For reference, sum ×25 sat at −0.5…+0.1.
   - The collapsed mean ×1 arm sat at −3.4…−3.9, with a minimum of −5.3.
3. **Near heads no worse:** dev `relerr/k5` at step 8000 is at or below the sum ×25 arm's worse
   seed **in each seed**, i.e. inside or below sum ×25's range (0.30203 / 0.29121, mean 0.2966). The
   mean comparison is reported beside it.
   - *Clarified at 16:21Z on 2026-11-19, while the mean runs were still training and before any
     mean result existed:* the first draft wrote the threshold as "0.302", the guard comparison's
     rounding of 0.30203. Taken literally, that would have failed the sum ×25 arm against itself.
     `experiments/latent_mean_check.py` reads the exact value from the sum runs' `latent_dev.json`.
   - Scored with `experiments/latent_relerr.py`: 200 dev batches of 32, seed 0, CPU.
   - These are the same windows as the guard comparison.

**Fails** if any one of these fails in either seed. Then the default stays `sum`.

**Secondary:**
- Dev relerr at every k. Embedding std per level through the run. Far log-variance.
- **Steps 1–2000 (near only):** `mean` there is sum ÷ 64 on one level, which under Adam is the same
  up to ε. So these steps should track sum ×25's closely, though not bitwise. Report the largest
  relative difference in `relerr/k1..5` and `emb_std/L1`.
- **The size of the near-head regression after far joins:** train-batch k5 relerr over steps
  1501–2000 against 7501–8000, next to sum ×25's 0.20 → 0.32 / 0.19 → 0.29.

**Baselines:** this run has no goal head, so persistence and physics are N/A, as in the guard
comparison. The in-latent reference is relerr = 1 (predicting the batch mean), plus the sum ×25 arm.

**Budget:** 2 runs × 8000 steps, about 17–19 min each on MPS, plus the 2100-step re-run (about
5 min). One job at a time under the GPU lock, then the CPU scoring.

## Results
(pending)
