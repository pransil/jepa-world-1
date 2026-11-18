# M5 placeholders: goal and pixel start steps, aux head sizes

- **Date:** 2026-11-17   **Decided by:** user (2026-11-18), as proposed by the m5 lane
- **SPEC ref:** §5.5, §4.6, §4.7
- **Status:** accepted

## Question
Wiring the goal head and pixel decoder into training needs values that neither SPEC nor an
existing decision fixes:
- when each stage joins `loss/total` (`train.stage_start.goal`, `.pixel`);
- the width of the goal head (`model.goal_hidden`) and of the pixel decoder (`model.pix_widths`);
- whether the goal head also pools L2 (`model.goal_l2`);
- which horizons it predicts (`model.goal_horizons`);
- how its per-horizon NLLs combine into `loss/goal`.

The [stage-schedule decision](2026-11-17-stage-schedule-within-run.md) set the mechanism and a
far placeholder, and left "M5 adds goal and pixel entries".

## Proposal (in `configs/base.yaml`, all placeholders until a real run tunes them)
- **`stage_start: {near: 0, far: 2000, goal: 4000, pixel: 6000}`.** This follows SPEC §5.5's
  order (near → far → goal → pixel, "pixel decoder last"), and keeps the 2000-step spacing that far
  already uses. Every term is logged from step 1 either way.
  - Alternative: all at 0 (option C of the stage decision). Simpler, but it ignores §5.5.
- **`goal_hidden: 128`.** The goal MLP reads the pooled L3 (256) plus an 8-d k-vector, so 128
  halves the width once. It runs on one vector per (sample, horizon), so its cost is negligible
  at any width.
- **`pix_widths: [64, 32]`.** The decoder stage at L2 (32×32) is 64 wide, and the stage at L1
  (64×64) is 32 wide, the same as the L1 head width the user chose from the MPS benchmark
  ([l1-head-width](2026-11-17-l1-head-width.md)).
  - A symmetric U-Net (128, 64) would roughly double the L1-resolution cost, which is about the
    cost of all five near heads.
  - Not benchmarked yet: the GPU was held by the first real training run.
- **`goal_l2: false`.** SPEC §4.6 reads "pooled L3 (plus optionally pooled L2)", so L3 alone is the
  literal default, and L2 is one override away.
- **`goal_horizons: [1, 2, 3, 4, 5, 10, 20]`**, the predictor's horizons, so that SPEC §7.2's
  per-horizon position error and persistence ratio cover every k.
- **`loss/goal` = the SUM over goal horizons** of the per-horizon NLL. This matches
  Σ_k w_k·L_k for the latent terms. With 7 horizons and w_goal = 0.5, the goal's total weight
  is 3.5 per-horizon NLLs.

## Decision
**Accepted as proposed** by the user on 2026-11-18:
- `stage_start` goal 4000, pixel 6000;
- `goal_hidden: 128` and `pix_widths: [64, 32]`;
- `goal_l2: false`;
- all 7 goal horizons, `[1, 2, 3, 4, 5, 10, 20]`;
- `loss/goal` is the sum over goal horizons.

These are already the values in `configs/base.yaml`, and each is one `--set` away from an
alternative.

## Consequences
- Experiments that train the goal and pixel stages inherit these values unless they override them.
- Revisit once the headline run shows the goal and pixel loss scales against the latent terms.
  The logged `loss/goal`, `loss/pix` and `loss/total` give that scale.
