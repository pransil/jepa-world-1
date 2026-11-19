# The M7 reference at the new defaults: the full model at 48k with per-channel latents and a 10% pixel share

- **Date:** 2026-11-19   **Lane/branch:** worktree-wt-exp-headline-48k-new
- **Commit:** the commit that adds this note, before any run (each run's `meta.json` records it)
- **Config:** `configs/exp_headline_48k.yaml`, unchanged (seed 0), plus `--set seed=1`
- **Run dirs:** filled in with the results
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

## Conclusion

## Next
