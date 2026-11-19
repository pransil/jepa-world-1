# Does a larger pixel term hurt the goal head? (M7 §1, rescaled to target shares)

- **Date:** 2026-11-18   **Lane/branch:** worktree-wt-exp-pixel-share   **Commit:** (filled in after the runs)
- **Configs:** `configs/m7_pixel_share01.yaml` (1%) and `configs/m7_pixel_share10.yaml` (10%),
  both `configs/exp_headline.yaml` plus `loss.pix_share`. Seed 0 is each file, seed 1 is `--set seed=1`.
- **≈ 0 arm:** the 24k headline runs, reused
  ([2026-11-17-headline-full-model.md](2026-11-17-headline-full-model.md): `w_pix` 0.05, pixel
  share ≤ 3e-5).
- **Run dirs:** (filled in after the runs)
- **Asked for:** [docs/milestones/M7-plan.md](../docs/milestones/M7-plan.md) §1, with levels revised
  by the user on 2026-11-18
  ([decision](../docs/decisions/2026-11-18-pixel-ablation-target-share.md)). `loss.pix_share`
  was built by m5 (PR #25).

## Hypothesis
From M7-plan §1, with high = 10% and low = 1%: raising the pixel term's share of the loss from
≈ 0 to 10% worsens the goal head's k = 20 position error. The pixel term pulls encoder capacity
toward reconstruction (SPEC §4.7's "blur" concern). Between ≈ 0 and 1% the difference stays within
the seed spread.

**Deciding metric:** goal-head mean position error at **k = 20** as a ratio to persistence, on the
**test** split, at step 24,000. Scored by `uv run python -m jepa_world.eval --run <run> --split
test --split dev`, on the same windows (t ∈ 3..39) and baselines as the headline.
≈ 0 reference: 0.448 (s0) / 0.412 (s1), mean 0.430.

**Supported if** both 10% seeds are above both ≈ 0 seeds (worse beyond the seed spread).
**Refuted if** the 10% mean is at or below the ≈ 0 mean (0.430). In between is inconclusive.

**Secondary:**
- 1% vs ≈ 0 on the same metric (expected within the seed spread);
- every k's ratio to persistence; calibration (50% / 90% coverage) and the visibility `std_ratio`;
- eval's linear probes: ball position relative to the camera (`pos_cam`) at L1 and L3, and pooled;
- embedding std per level;
- the achieved `pix_share` and `w_pix_eff` from `metrics.jsonl`, and `loss/pix` itself (does the
  decoder reconstruct better?);
- near-head relerr over the run, against the headline's drift (k5 0.36 at 24k).

**These are shares of the loss value**, not of the encoder's gradient, which isn't measured (see the
decision). Read 1% and 10% as value shares.

**Baselines:** persistence, constant velocity, and the simulator with mean parameters (the physics
bar), with true parameters as the oracle; all from eval, on the same windows.

**Budget:** 2 levels × 2 seeds × 24,000 steps, about 62 min each on MPS, plus one eval per run.
They run in the order 1% s0, 10% s0, 1% s1, 10% s1, so that if the night runs short, both levels
have one seed.

## Results
(filled in after the runs)

## Conclusion
(filled in after the runs)

## Next
(filled in after the runs)
