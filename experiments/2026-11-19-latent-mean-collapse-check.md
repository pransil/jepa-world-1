# Does averaging the latent losses over channels train near + far without collapse? (latent_reduce: mean)

- **Date:** 2026-11-19   **Lane/branch:** worktree-wt-latent-reduce   **Commits** (all clean, per each
  run's `meta.json`): the mean runs 00a4e78; the re-run bc2b819. The two differ only in `experiments/`,
  with the same model, training and config code.
  Later commits on the branch changed only the aux heads' weights (goal and pixel), which a near +
  far run doesn't have. `uv run python experiments/latent_reduce_bitwise.py --ref 00a4e78 --reduce mean
  --match "near+far"` trains 6 `mean` near + far steps through 00a4e78's `jepa.py`/`world.py` and
  through the final ones. It printed BITWISE IDENTICAL: every logged term and parameter in all 4
  near + far setups, including the world path with no aux head, which is what these runs trained.
- **Run dirs:** mean `20261119T162336Z-worktree-wt-latent-reduce-latent-mean-s0-e82d5b` (s0) and `20261119T163737Z-worktree-wt-latent-reduce-latent-mean-s1-de9b34` (s1). Each holds `latent_dev.json`. The sum ×25
  re-run is `20261119T162037Z-worktree-wt-latent-reduce-latent-sum-repro-s0-3a0441`.
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
Scored by `uv run python experiments/latent_mean_check.py --mean <s0> --mean <s1> --sum <s0> --sum
<s1>` (rule 3's threshold is read from the sum runs). Dev scores come from
`experiments/latent_relerr.py --run <dir> --split dev` (200 batches of 32, seed 0, CPU), the guard
comparison's windows.

### The sum ×25 arm reproduces at this commit
The 2100-step re-run of `configs/exp_first_real_run.yaml` (seed 0, bc2b819) matches the first real
run's seed 0 `metrics.jsonl` exactly, including the 100 steps after far joins:
`experiments/compare_metrics.py --run <re-run> --ref <first-real-s0>` printed "2100 rows x 27 keys
compared; differing values: 0; max abs diff 0.0". So the first real run's two seeds are the sum ×25 arm, unchanged.

### Pass rule: **PASS**, all three rules, both seeds

| seed | L3 std: min after far joins / step 2000 | logvar k10: last 500 / min | logvar k20: last 500 / min | **dev k5 relerr** (≤ 0.30203) |
|---|---|---|---|---|
| mean s0 | 101% (0.379 at step 2000) | −0.53 / −1.74 | +0.03 / −1.26 | **0.284** |
| mean s1 | 99% (0.386) | −0.51 / −1.70 | +0.04 / −1.28 | **0.275** |
| *sum ×25 s0* | *101%* | *−0.46 / −1.83* | *−0.03 / −1.21* | *0.302* |
| *sum ×25 s1* | *99%* | *−0.42 / −1.68* | *+0.03 / −1.25* | *0.291* |

### Dev relerr at every horizon, step 8000 (development split; SPEC §7.3)

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| mean s0 / s1 | 0.078 / 0.070 | 0.140 / 0.126 | 0.197 / 0.184 | 0.247 / 0.233 | **0.284 / 0.275** | 0.473 / 0.440 | **0.693 / 0.680** |
| sum ×25 s0 / s1 | 0.075 / 0.067 | 0.142 / 0.133 | 0.206 / 0.196 | 0.258 / 0.245 | 0.302 / 0.291 | 0.467 / 0.462 | 0.666 / 0.672 |
| mean of seeds, mean vs sum | 0.074 vs 0.071 | 0.133 vs 0.138 | 0.191 vs 0.201 | 0.240 vs 0.252 | **0.279 vs 0.297 (−6%)** | 0.457 vs 0.465 | **0.687 vs 0.669 (+2.6%)** |

- **k5:** both mean seeds are below both sum seeds.
- **k20:** both mean seeds are above both sum seeds. The far heads lose a little, as the design
  expected: under `mean` their terms weigh 2–4× less at the shared encoder stages.
- **Other horizons:** the seeds overlap at k = 1–4 and k = 10. The per-batch std is 0.05 for k5 and
  0.10 for k20, over the same 200 windows for every run.

### Training (train batches, `metrics.jsonl`)
- **The near heads regress less once far joins.** k5 relerr, steps 1501–2000 → 7501–8000:
  - mean: 0.197 → 0.287 (s0), 0.194 → 0.280 (s1);
  - sum ×25: 0.199 → 0.318, 0.195 → 0.291.
- **L1's dip when far joins is smaller:** to 83% / 86% of its step-2000 std, against 76% / 71%.
  It then grows to 1.61–1.66, as under sum (1.65).
- **L2 and L3 never dip.** Their minimum after step 2000 is at step 2001, at 99–102%. They end at
  1.37–1.48, against sum's 1.40–1.52.
- **Every logged value is finite** in both runs, 8000 rows each.
- **`loss/total` is about 64× smaller in the near-only phase and about 135× smaller at the end**
  (1.03–1.05 vs 66–68 over steps 1501–2000; 5.2 vs 704 over the last 500). These are the
  predicted 64× and C_eff.
- **Steps 1–2000 (near only, mean = sum ÷ 64): close, not bitwise.** These numbers are printed by
  `latent_mean_check.py` ("steps 1-2000, mean vs sum").
  - The windowed means agree: k5 relerr 0.197 vs 0.199, and 0.194 vs 0.195, over steps 1501–2000.
  - Per step, relerr differs by a median 1.3–2.9%. The largest differences are early, up to 76%
    while relerr is still large and moving fast.
  - `emb_std/L1` differs by a median 0.1%, at most 1–3.6%.
  - In code, the only difference in this phase is the uniform ÷ 64, and Adam cancels a uniform
    scale except through ε (1e-8). So ε is the likely source of the drift. That is inferred from
    the code, not measured; a run with ε × 64 would test it. Either way the windowed statistics
    match, which supports "a uniform scale barely matters".

## Conclusion
**Passed.** Averaging the latent losses over channels, with the guard following them (sum ×25's
ratio at every level), trains near + far without collapse:
- L3 never dips;
- the far log-variance stays at −0.5…+0.04;
- the near heads regress less once far joins: dev k5 relerr is **0.284 / 0.275**, against sum ×25's
  0.302 / 0.291.

The cost is a small loss at k = 20: dev relerr 0.693 / 0.680 against 0.666 / 0.672 (+2.6%).

As pre-registered, base.yaml's default becomes `latent_reduce: mean` in the same PR
([decision](../docs/decisions/2026-11-19-latent-loss-normalization.md), now accepted). The goal
and fixed pixel weights keep SPEC's values, read relative to L3's latent terms.

## Next
- The 48k M7 reference re-run (the user's call on GPU time) is the first full-model run at the new
  defaults. Watch k20, both the latent relerr and the goal head's ratio to persistence, against
  the old 48k runs.
- Proposed, not built: log each term's encoder-gradient norm, so the weights between levels and
  against the aux terms are measured, not inferred from values.
