# Does the far heads' Gaussian NLL collapse L2/L3, and which switch stops it?

- **Date:** 2026-11-17   **Lane/branch:** worktree-wt-m4-far-heads   **Commit:** 2e7d888 (synthetic), c695143 (dev), clean
- **Config:** synthetic: `configs/m4_far_synthetic.yaml`, `m4_collapse_layernorm.yaml`,
  `m4_collapse_guardsum.yaml`, `m4_collapse_both.yaml`. Dev: `configs/m4_far_dev.yaml` with
  `--set` overrides (commands below)
- **Run dirs:** listed in each results table. All MPS, seed 0, B = 32, under the GPU lock.

## Hypothesis
With raw targets, the far heads' Gaussian NLL rewards shrinking the L2/L3 embeddings (smaller
error → smaller variance → more negative NLL) far more than the variance guard (≤ 1 per level,
w_var = 1) penalises it. So L2/L3 collapse when far joins, and the predicted log-variance runs
to its −10 floor. **Deciding metrics** (measured before far joins, and at the end):
- `emb_std/L2`, `emb_std/L3` (collapse);
- `logvar/k10`, `logvar/k20` (runaway);
- `relerr/k*`: error relative to predicting the batch-mean target, scale-free, so it cannot be
  improved by shrinking.

**Refuted if** the baseline keeps `emb_std/L3` within 2× of its pre-far value and `logvar` above
−3.

Variants:
- **(a)** `loss.target_norm: layernorm`: per-position LayerNorm'd targets.
- **(b)** `loss.var_reduce: sum`, `loss.w_var: 25`: guard summed over channels (the latent losses'
  scale), at VICReg's variance weight.
- **(a)+(b)**: both together.

Persistence and physics baselines: N/A. This is about training dynamics, not prediction quality
against baselines. Embedding persistence is an M6 metric.

## Results: synthetic moving disc (300 steps, far joins after step 100)

| setting | run dir | L3 std: 100 → 110 → 300 | logvar k10 / k20 @300 | relerr k10 / k20 @300 | relerr k1 / k5 @300 |
|---|---|---|---|---|---|
| baseline | `20261118T052010Z-…-cmp-m4_far_synthetic-89917d` | 0.135 → **0.005** → 0.027 | −7.3 / −7.9 | 0.16 / 0.48 | 0.32 / 0.52 |
| (a) layernorm | `20261118T052109Z-…-cmp-m4_collapse_layernorm-b2ba01` | 0.139 → **0.014** → 0.035 | −6.3 / −6.9 | 0.13 / 0.70 | 0.40 / 0.51 |
| (b) guard sum ×25 | `20261118T052229Z-…-cmp-m4_collapse_guardsum-aa39f2` | 0.43 → 0.68 → 0.93 | −0.65 / −0.39 | 0.81 / 1.01 | 0.21 / 0.65 |
| (a)+(b) | `20261118T052350Z-…-cmp-m4_collapse_both-619f0b` | 0.43 → 0.67 → 0.97 | −1.0 / −0.56 | 1.10 / 1.07 | 0.89 / 1.02 |

All values were finite in every run.

## Results: real episodes, M2 dev split (600 steps, far joins after step 200) — dev, debugging only

Commands (config `configs/m4_far_dev.yaml`, commit c695143):
```
uv run python -m jepa_world.train --config configs/m4_far_dev.yaml --name dev-baseline
  ... --set loss.target_norm=layernorm --name dev-layernorm
  ... --set loss.var_reduce=sum --set loss.w_var=25.0 --name dev-guardsum
  ... --set loss.target_norm=layernorm --set loss.var_reduce=sum --set loss.w_var=25.0 --name dev-both
```
Runs 2–4 record `dirty: true`, because a YAML comment in base.yaml and some docs were edited
while they ran. `git diff c695143 -- '*.py'` is empty, so the code is identical.

| setting | run dir | L3 std: 200 → 210 → 600 | logvar k10 / k20 @600 | relerr k10 / k20 @600 | relerr k1 / k5 @600 |
|---|---|---|---|---|---|
| baseline | `20261118T052743Z-…-dev-baseline-a710ab` | 0.35 → **0.097** → 0.30 | −4.2 / −5.1 | 0.59 / 0.44 | 0.13 / 0.44 |
| (a) layernorm | `20261118T052943Z-…-dev-layernorm-3c5d77` | 0.35 → **0.055** → 0.25 | −2.7 / −4.1 | 0.51 / 0.43 | 0.12 / 0.25 |
| (b) guard sum ×25 | `20261118T053203Z-…-dev-guardsum-df9b23` | 0.39 → 0.70 → 1.08 | −1.2 / −0.4 | 0.66 / 0.89 | 0.07 / 0.26 |
| (a)+(b) | `20261118T053413Z-…-dev-both-ecc99d` | 0.40 → 0.71 → 1.05 | −0.9 / −0.6 | 0.46 / 0.99 | 0.12 / 0.21 |

All values were finite. The dev split (40 episodes) is for debugging (SPEC §3.5). These are
training-set numbers on dev, not generalization.

Training sanity: every run trained without NaN. Before far joins, the far terms are logged and
flat (no gradient), which confirms the schedule.

## Conclusion
**Supported on synthetic data, partly on real data.** With raw targets, the far NLL collapses
L3 (synthetic: 0.135 → 0.005, ×27; dev: a temporary ×3.6 dip) and drives log-variance toward
the floor (−4 to −8). (a) does not fix it. (b) prevents both on both datasets. Its open cost is
the k = 20 error (relerr 0.89 at 600 steps, vs 0.44 for the baseline). The user chose (b) as the
M4 default: [docs/decisions/2026-11-17-far-collapse-guard-sum.md](../docs/decisions/2026-11-17-far-collapse-guard-sum.md).

## Next
A longer run (≥ 5k steps) on the train split, with several seeds: does k = 20 learn under (b)?
Compare relerr k20 against the baseline, and against embedding persistence once M6 provides it.
