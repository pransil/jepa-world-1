# Headline: does the full model's goal head beat persistence and the baselines on held-out terrains?

- **Date:** 2026-11-17   **Lane/branch:** worktree-wt-exp-headline   **Commit:** RUN_SHA (training)
- **Config:** `configs/exp_headline.yaml` (seed 0), plus `--set seed=1`
- **Run dirs:** RUN_DIRS

## Hypothesis
The full model (near, far, goal and pixel; base.yaml defaults, train split, 24000 steps) gets a
goal-head mean position error below persistence's at every horizon on the held-out **test**
terrains, and below constant velocity's at k ≥ 5. It does not reach the simulator bar.

**Deciding metric:** the goal-head mean position error as a ratio to persistence, per k
(1, 2, 3, 4, 5, 10, 20), on the test split, for both seeds.
- Windows: t ∈ 3..39, the baselines' set.
- Scored by `python -m jepa_world.eval --run <run> --split test --split dev`.
- Dev numbers are reported separately, never as the headline (SPEC §7.3).

**Refuted if** the ratio is ≥ 1 at any horizon for either seed.

**Baselines, on the same test windows** (ratio to persistence, at k = 1 / 5 / 20), from run
`20261118T065246Z-worktree-wt-sim-baseline-baselines-test-590f46`:
- constant velocity: 0.17 / 0.57 / 1.21;
- simulator with mean ball parameters, **the headline physics bar** (accepted by the user on
  2026-11-18;
  [docs/decisions/2026-11-17-simulator-baseline-params.md](../docs/decisions/2026-11-17-simulator-baseline-params.md),
  whose status m6's docs PR flips): 0.014 / 0.055 / 0.080;
- simulator with true parameters, **the oracle ceiling** (same decision): 0.000.

**Also reported:**
- calibration: the fraction of truths inside the 50% and 90% intervals;
- the visibility `std_ratio` (does uncertainty rise when the ball is out of view?);
- `loss/pix`'s share of `loss/total` (SPEC §8 M5: the pixel term must not dominate);
- the aux heads' MPS step cost (asked for by m5);
- whether the near heads regress when goal and pixel join, as they did when far joined in
  [the first real run](2026-11-17-first-real-run.md) (k5 relerr 0.17 → 0.30).

**Budget:** 2 seeds × 24000 steps, B = 32, MPS under the GPU lock, sized to about 75–90 min per
seed at the measured step time (below).

**Settings this run uses, accepted by the user on 2026-11-18** (relayed by the orchestrator; m6's
docs PR flips both decision files from proposed):
- `stage_start` goal 4000 and pixel 6000, `goal_hidden` 128, `pix_widths` [64, 32], and
  `loss/goal` = the sum over 7 horizons
  ([docs/decisions/2026-11-17-m5-stage-starts-and-aux-sizes.md](../docs/decisions/2026-11-17-m5-stage-starts-and-aux-sizes.md));
- the physics bar: the simulator with mean parameters, with true parameters as the oracle (above).

## Step cost (MPS, B = 32, synthetic batches, `python -m jepa_world.bench`, load average 70–85)

| stages | ms per train step | MPS driver MB | run dir |
|---|---|---|---|
| near + far | 150.2 | 3261 | `20261118T072549Z-worktree-wt-exp-headline-bench-l1-heads-7d682d` |
| near + far + goal + pixel | 182.0 | 3266 | `20261118T072600Z-worktree-wt-exp-headline-bench-l1-heads-0a229e` |

The goal head and the pixel decoder add **31.8 ms per step (+21%)**. Memory is unchanged, and no op
falls back to CPU.

## Results

RESULTS

## Conclusion

CONCLUSION

## Next

NEXT
