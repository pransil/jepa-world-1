# Latent losses: average over channels instead of summing (`loss.latent_reduce`)

- **Date:** 2026-11-19   **Decided by:** user (the direction, 2026-11-19, relayed by the orchestrator
  session); the design below is the m5 lane's   **SPEC ref:** §5.1, §5.4, §9.3
- **Status:** accepted (user 2026-11-19, design by m5). The collapse check passed in both seeds
  ([note](../../experiments/2026-11-19-latent-mean-collapse-check.md)), so base.yaml's default is `mean` from the PR that added the switch.

## Question
Every latent term is a sum over its level's channels, averaged over batch and positions. That covers
the near squared L2, the far Gaussian NLL, the top-down NLL, and the variance guard
(`var_reduce: sum`). The values run in the hundreds and grow as the embeddings spread: in the headline
runs the rest of the loss went from 431 to 1640. So every O(1) auxiliary term is swamped, and a fixed
weight can't hold it at any particular share. The pixel term showed this
([pixel-ablation-target-share](2026-11-18-pixel-ablation-target-share.md)).

With Adam, a uniform scale on the whole loss barely matters: each parameter's step is divided by
its own gradient's running RMS, so the scale cancels up to Adam's ε. What a reduction change does
in practice is change the **relative weights**:
1. **Between latent levels:** L1 has 64 channels, L2 128 and L3 256. Under `sum`, each level's terms
   carry a factor of its C.
2. **Guard vs latent, per level:** M4's far-collapse fix is a balance between these two.
3. **Goal vs latent.**
4. **Pixel vs latent:** already a share of the rest (`pix_share` 0.10), so it adapts.

So the design has to state what happens to each of these.

## Measured: what `mean` would do to the headline runs
Every logged latent term is a channel sum, so its `mean` value is exactly sum / C. The figures below
are window means from the two 24k headline runs' `metrics.jsonl`
(`runs/20261118T231506Z-…-headline-s0-32ff77`, `runs/20261119T001708Z-…-headline-s1-d2c5c1`),
printed by `uv run python experiments/latent_reduce_scale.py --run <dir> --run <dir>`:

| steps | latent total, sum | latent total, mean | ratio (C_eff) | near/far, sum | near/far, mean | goal term / latent at w_goal 0.5, sum | same under mean |
|---|---|---|---|---|---|---|---|
| 1501–2000 (near only) | 66–68 | 1.04–1.06 | **64.0** | – | – | – | – |
| 2001–2500 (far joins) | 1.9e3 | 12 | 159–161 | 0.32–0.34 | 1.0–1.05 | – | – |
| 4001–4500 (goal joins) | 636 | 4.4–4.7 | 137–144 | 0.22–0.27 | 0.69–0.86 | +0.5% | **+74–76%** |
| 11501–12000 | 737–761 | 6.2 | 120–123 | 0.41–0.44 | 1.27–1.33 | +0.1% | +8–11% |
| 23501–24000 | 829–856 | 7.4–7.6 | 112–113 | 0.56–0.57 | 1.71–1.75 | −0.7…−0.8% | **−83…−93%** |

"Latent total" means the near, far and guard terms (w_var 25). The goal NLL turns negative as the
goal head grows confident.

## Design
1. **`loss.latent_reduce: sum | mean`, one key.** Under `mean`, every per-level term is divided
   by the channel count of the level it is scored on:
   - `loss/k` for k = 1–5 by L1's count;
   - each far head's `loss/k` by its own level's count;
   - `loss/td/k` (top-down) by its target level's count;
   - each level's guard (`var/<L>`) by that level's count (item 2);
   - the logged `sqerr` the same way.

   **`sum` is the existing code path, bitwise.** `test_sum_is_bitwise_the_channel_sum_formulas`
   pins it. `uv run python experiments/latent_reduce_bitwise.py --ref origin/main` trains a tiny
   model for 6 steps through main's `jepa.py` and through this one. It printed BITWISE IDENTICAL:
   every logged term and parameter matched in 5 setups (near only, far joining mid-run, the mean
   ×1 guard, top-down, a config with no key). With `mean` it prints DIFFERENT. A config without
   the key means `sum`, so checkpoints, eval and `latent_relerr.py` reproduce every earlier run.
2. **The guard follows `latent_reduce`.** `var_reduce` keeps its M4 meaning, *relative to the
   latent terms*: `sum` is their channel reduction, `mean` is 1/C of it. So each level's balance
   (guard × w_var against that level's latent terms) moves by 1/C on both sides, and is set by
   `var_reduce` and `w_var` alone under either `latent_reduce`. Sum ×25 stays sum ×25, and the
   mean ×1 ablation stays mean ×1. M4's fix
   ([far-collapse-guard-sum](2026-11-17-far-collapse-guard-sum.md)) holds level by level
   (`test_every_level_keeps_its_guard_to_latent_ratio[sum|mean]`).
   - **Lost: a second key that must agree** (`latent_reduce: mean` with `var_reduce: mean`,
     and the mixed setting refused). Every config would have had to flip both, and the
     invalid mix could be written.
   - **Not the guard comparison's "mean ×1" arm,** which collapsed
     ([experiment](../../experiments/2026-11-18-guard-comparison.md)). Under `mean` the guard
     stays at sum ×25's ratio.
3. **Between levels**, the far heads' weight against the near heads drops 2× (k = 10) and 4×
   (k = 20). This is the intended effect. By encoder stage:
   - **Stage 3** (L3's parameters) sees only L3 terms: the k20 NLL, the L3 guard, and the goal head,
     which reads L3. Their ratios don't change, so in a near + far run its dynamics are the same as
     under `sum`, up to Adam's ε. The pixel decoder reads every level, but it is share-based. The
     top-down k20 branch reads L3 too, but is scored on L2 (item 6).
   - **Stage 2** sees the L2 terms (k10, L2 guard) and the L3 terms. The L2 terms gain 2× over the
     L3 terms.
   - **Stage 1** sees everything. The near and L1 terms gain 2× over L2's and 4× over L3's.

   On the headline runs, near/far moves from 0.2–0.6 to 0.6–1.75 (table). The near heads regress
   once the far terms join the shared encoder: k5 relerr on train batches goes 0.20 → 0.32, and the
   guard comparison ruled out the guard as the cause
   ([experiment](../../experiments/2026-11-18-guard-comparison.md)). If the far terms' weight is
   the cause, this rebalancing should reduce that regression. The collapse check's k5 criterion
   measures it.
4. **Goal: `w_goal` = 0.5 / 256 = 0.001953125**, the same ratio to the L3 latent terms as before.
   - **Kept at 0.5,** the goal term's weight against the latent terms would jump about C_eff×
     (112–161) in value, and 256× at stage 3, where it enters.
   - On the headline it would reach +75% of the latent loss when it joins, and −83…−93% by 24k.
     Then the rest of the loss heads toward zero, and the pixel share's weight
     (s·|L_rest| / L_pix) collapses with it.
   - That is a different experiment (supervising the encoder with ground-truth position), not a
     side effect of a normalization. The collapse check (near + far) can't test it either.
   - **0.5 / 256 makes the goal term move with L3,** the level it reads, so its ratio to the k20
     NLL and the L3 guard is what it was in every run so far. Its value share stays near zero, as
     before (× C_eff / 256, about half).
   - **Alternatives:**
     - 0.5 / C_eff ≈ 0.004, to preserve the value share. C_eff drifts from 112 to 161 within a
       run and depends on which terms are active, so this has no mechanistic anchor.
     - A target share for the goal, like `pix_share`. That is a new mechanism, and a proposal
       below.
   - **Under `sum`, `w_goal` stays 0.5.** This value is tied to `mean`, and the configs that keep
     `sum` pin both.
   - **Not enforced in code.** `latent_reduce: mean` with `w_goal: 0.5`, or with a fixed `w_pix`
     (`pix_share: null`), trains a different objective: goal or pixel term about C_eff× heavier.
     It is a legitimate experiment (the SPEC §5.4 weights on per-channel latents), so it isn't
     refused. base.yaml's comment on `w_goal` states the pairing.
5. **Pixel:** `pix_share` holds the pixel term at 10% of |L_rest| at any scale
   (`test_goal_and_pixel_terms_do_not_depend_on_it`). `w_pix_eff` will be about C_eff× smaller.
   With `w_goal` as above, L_rest stays positive: the latent total under `mean` is 4–8 on the
   headline.
6. **Top-down (`w_td` 0.5, off by default):** each branch's NLL is divided by its target level's
   channel count, like that level's own terms, so its ratio to them is unchanged. A branch reads
   the coarser level, though. So at the stage it enters, it gains 2× against that level's own far
   NLL: k20's branch scores on L2's 128 channels, k20 on L3's 256. M7's top-down ablation should
   compare at one reduction.
7. **`loss/total` is not comparable across the two settings.** It is 64× smaller under `mean`
   before far joins, and about C_eff× (112–161) smaller after. Compare runs on the
   reduction-free logs:
   - `relerr/k`, a ratio of two channel sums, computed before any division. It is bitwise
     identical for the same weights (`test_scale_free_diagnostics_are_bitwise_unchanged`).
   - `emb_std/L`, `logvar/k`, `goalerr/k` and `goalratio/k`.

   To compare a `mean` run's `loss/k`, `sqerr/k`, `var/L` or `loss/var` with a `sum` run's,
   multiply by that level's channel count.

## Collapse check (before any default changes)
`configs/exp_latent_mean.yaml` is `configs/exp_first_real_run.yaml` with only `latent_reduce`
changed. That config is near + far, 8000 steps, far joining after 2000, on the train split, and it
is the sum ×25 arm of the guard comparison. Seeds 0 and 1, under the GPU lock.

The pass rule, with its exact thresholds, lives in one place: [the note](../../experiments/2026-11-19-latent-mean-collapse-check.md), committed
before the runs. In short: no L3 collapse, no runaway far log-variance, and dev k5 relerr no
worse than sum ×25's.

**Result: passed, all three rules, both seeds.**
- **L3 std:** its minimum after far joins was 99–101% of its step-2000 value.
- **Far log-variance:** −0.53…+0.04 over the last 500 steps, minimum −1.74.
- **Dev k5 relerr:** 0.284 / 0.275, against sum ×25's 0.302 / 0.291.
- **The trade-off:** dev k20 relerr is 0.693 / 0.680 against 0.666 / 0.672, +2.6% and worse in
  both seeds. That is the expected price of weighting the far terms less at the shared stages.
  The other horizons overlap across seeds.

## Consequences
- **base.yaml:** `latent_reduce: mean` and `w_goal: 0.001953125` (0.5 / 256). `var_reduce: sum`
  and `w_var: 25` don't change.
- **Configs with recorded `sum` runs pin `sum`**, with `w_goal: 0.5` where there is a goal
  head: `exp_first_real_run` (the comparison arm), `exp_guard_mean`, `m3_*`, `m4_*` and
  `m7_pixel_share*` (`test_configs_with_recorded_sum_runs_keep_sum`).
- **The M7 reference chain inherits `mean`:** `exp_headline`, `exp_headline_48k` and
  `m7_obstacles`, as [m7-ablations-at-48k](2026-11-19-m7-ablations-at-48k.md) requires. To
  reproduce an earlier headline run, use its commit, or `--set loss.pix_share=null --set
  loss.latent_reduce=sum --set loss.w_goal=0.5`.
- **SPEC:** §5.1 points here, and §5.4's `w_goal` line gives 0.5 / 256 under `mean`.
  [far-collapse-guard-sum](2026-11-17-far-collapse-guard-sum.md) has a dated note.
- **Logged scales:** `experiments/latent_relerr.py` records the run's `latent_reduce` in its
  output, because its `loss/k`, `sqerr/k` and `var/L` scale with it.
- **Untested by the check:** the goal and pixel terms under `mean` (it was near + far only). The
  48k M7 reference re-run will be the first full-model run at the new defaults, so read its goal
  metrics against the old 48k runs knowing that the latents, the pixel share and `w_goal` all
  changed.
- **Revisit if** that re-run's k20 metrics (latent relerr, or the goal head's ratio to
  persistence) lose more than the +2.6% latent k20 cost seen here.
- **Proposals, not built:**
  - log each term's encoder-gradient norm, so the relative weights are measured, not inferred
    from values;
  - a goal target share, if the user wants the goal term to shape the encoder.
