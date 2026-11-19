# M7 ablations train for 48k steps, not 24k

- **Date:** 2026-11-19   **Decided by:** user (2026-11-19, relayed by the orchestrator session)   **SPEC ref:** §6, §8 M7
- **Status:** accepted

## Question
How long should each M7 ablation arm train? The ablations were planned at the headline length,
24,000 steps, and the pixel-share ablation ran at 24k
([2026-11-18-pixel-share-ablation.md](../../experiments/2026-11-18-pixel-share-ablation.md)).
Then the 48k run ([2026-11-18-headline-48k.md](../../experiments/2026-11-18-headline-48k.md),
proposal 3) showed that at 24k the goal head is still on the steep part of its learning curve:
- **Training batches:** k20 `goalratio` fell from 0.603 / 0.549 (steps 16,001–20,000, seeds 0 / 1)
  to 0.483 / 0.433 (20,001–24,000). That is about 20% in one 4k window.
- **At the end of the 48k run** it fell from 0.347 / 0.339 to 0.340 / 0.330, which is 2–3% per 4k.
- **Test:** the k = 20 error ratio to persistence went from 0.430 (24k mean) to 0.350 (48k mean),
  and every horizon improved by 15–27%.

So at 24k, an ablation that only changes how fast the model learns moves the deciding metric as
much as one that changes where it ends up. The difference between arms mixes with each arm's
position on the learning curve.

## Options
- **A: 24k steps per arm.** ≈ 0.9–1 h per seed. Arms are compared mid-descent.
- **B: 48k steps per arm.** ≈ 2 h per seed, twice A. The goal head is still improving at 48k, but
  only by 2–3% per 4k on training batches, so arms are compared near the flat part.
- **C: 96k steps per arm.** ≈ 4 h per seed. The 48k note proposes 96k only to find the asymptote;
  as an ablation length it would double B's cost for a 2–3% per 4k tail.

## Decision
**B.** Every remaining M7 ablation arm trains for 48,000 steps, seeds 0 and 1, against a 48k
reference (`configs/exp_headline_48k.yaml`). B moves the comparison off the steep part of the curve
at twice A's cost, and C's extra gain is too small to justify doubling the cost again.

## Consequences
- **Configs:** an M7 ablation config uses `base: exp_headline_48k.yaml` and adds only the keys under
  test. `configs/m7_obstacles.yaml` now does, so it trains 48k steps instead of base.yaml's
  10k placeholder. Its `sim` and `data` blocks are unchanged, so its dataset key is unchanged.
  The skips, camera and top-down configs don't exist yet; their lanes write them this way.
- **The pixel-share ablation keeps its 24k configs.** `configs/m7_pixel_share01.yaml` and
  `configs/m7_pixel_share10.yaml` record what ran in PR #28. A 48k re-run of that ablation is not
  part of this decision; it would need new configs and the user's go-ahead.
- **The 48k reference must be re-run.** The existing 48k runs (commit 5ed1564) used base.yaml's
  defaults at that commit. m5 is changing two of those defaults: `loss.pix_share` to 0.10, and a new
  latent-loss normalization.
  - Once both have landed, re-run `configs/exp_headline_48k.yaml` at the new commit, seeds 0 and 1
    (2 runs, ≈ 4 GPU-h), before comparing any ablation with it. This follows M7-plan's reuse rule.
  - The config file itself needn't change: it inherits base.yaml, so the commit it runs at
    decides which defaults it has.
  - The orchestrator is asking the user about that GPU time. This decision launches no training.
- **Cost:** about 2 GPU-h per seed. Measured, the 48k runs trained in 1.90–1.95 h (0.143–0.146 s/step),
  and the 24k pixel-share runs at 0.130–0.132 s/step; each eval took about 2 min. The remaining
  ablations come to about 25 GPU-h. The per-ablation table is in
  [M7-plan.md](../milestones/M7-plan.md), "Summary".
- **Revisit if** the re-run reference's training `goalratio` has flattened well before 48k (then a
  shorter run would do), or if a 96k run, should the user approve one, shows a large further gain
  between 48k and 96k.
