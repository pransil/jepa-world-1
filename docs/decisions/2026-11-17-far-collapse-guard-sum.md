# Far-head collapse: sum the variance guard over channels, at weight 25

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §9.3 (collapse guard), §9.5 (distance), §5.1
- **Status:** accepted. Amends the guard's reduction and weight in
  [2026-11-17-collapse-guard-on.md](2026-11-17-collapse-guard-on.md). The guard stays on, and its
  form is unchanged.

## Question
When the far heads joined training, their Gaussian NLL collapsed the L2/L3 embeddings. The
NLL rewards shrinking the targets: smaller error lets a smaller variance be predicted, which
makes the NLL more negative. The variance guard, a mean over features at w_var = 1, adds at
most 1 per level, against NLL gains in the hundreds. What should the default loss be?

## Options
Measured in [experiments/2026-11-17-m4-far-collapse.md](../../experiments/2026-11-17-m4-far-collapse.md)
on synthetic data (300 steps) and on M2's real dev episodes (600 steps), one seed each:
- **Raw defaults** (`target_norm: none`, `var_reduce: mean`, `w_var: 1`).
  - Synthetic: L3 std 0.135 → 0.005.
  - Real data: a temporary 3.6× dip that recovers.
  - Far log-variance runs to −4…−8.
  - Best far error at 600 steps (relerr k20 0.44).
- **(a) LayerNorm'd targets.** Behaves like the raw defaults: the L3 dip is 6× on real data, and
  log-variance still runs down. Normalizing the scale doesn't remove the NLL's pull toward
  predictable targets.
- **(b) Guard summed over channels, at w_var 25.** This is the latent losses' scale, at VICReg's
  variance weight.
  - No collapse on either dataset (real data: L3 std 0.39 → 1.08).
  - Log-variance stays near −1.
  - Best near error (relerr k1 0.07).
  - But k = 20 barely beats the batch-mean target at 600 steps (relerr 0.89).
- **(a)+(b).** No collapse, best k = 10 and k = 5 errors, but k = 20 at the batch-mean level
  (relerr 0.99), and the most departures from the M3 decisions.

## Decision
**(b):** `loss.var_reduce: sum`, `loss.w_var: 25.0`, `loss.target_norm: none`. It is the only
setting that never collapsed and kept the far log-variance meaningful on both datasets. That
matters, because SPEC §7.2 calibrates those variances.

## Consequences
- `configs/base.yaml`: `var_reduce: sum`, `w_var: 25.0`, `target_norm: none`, none of them
  marked [OPEN]. The switches remain for ablations (M7).
- The guard now dominates `loss/total` early on: var/L3 starts around 150 at w_var 25. Read the
  per-term logs, not the total.
- **Every config that inherits base.yaml changes behaviour**, including M3's
  `m3_near_synthetic.yaml`: its near-only runs now carry the summed ×25 guard on L1. Runs from
  before this decision reproduce exactly from their own run dir (`train --config
  <run>/config.yaml`; the resolved config records `var_reduce: mean, w_var: 1.0`). The
  comparison configs (`m4_collapse_*.yaml`, `m4_far_dev.yaml`) pin all three switches.
- **Open:** k = 20 under (b) was still near relerr 1.0 after 600 steps. A longer run on the
  train split, with several seeds, should check that the far heads learn under the stronger
  guard before M5's goal head relies on pooled L3. Revisit if k = 20 stays at relerr ≥ 0.9.
