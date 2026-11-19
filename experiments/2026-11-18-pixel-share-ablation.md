# Does a larger pixel term hurt the goal head? (M7 §1, rescaled to target shares)

- **Date:** 2026-11-18/19   **Lane/branch:** worktree-wt-exp-pixel-share
- **Commit:** f1a7c03 (clean, on main 27284ff with `loss.pix_share` from PR #25), for training
  and eval of all four new runs
- **Configs:** `configs/m7_pixel_share01.yaml` (1%) and `configs/m7_pixel_share10.yaml` (10%),
  both `configs/exp_headline.yaml` plus `loss.pix_share`. Seed 0 is each file, seed 1 is `--set seed=1`.
- **≈ 0 arm:** the 24k headline runs, reused
  ([2026-11-17-headline-full-model.md](2026-11-17-headline-full-model.md): `w_pix` 0.05, pixel
  share ≤ 3e-5).
- **Run dirs** (each holds `eval/<time>-test-dev-<id>.json` from
  `uv run python -m jepa_world.eval --run <dir> --split test --split dev`, at step 24,000):
  - 1%: `20261119T105929Z-worktree-wt-exp-pixel-share-pixshare01-s0-114d5d` (eval `20261119T115351Z-test-dev-a786f1.json`)
    and `20261119T124737Z-worktree-wt-exp-pixel-share-pixshare01-s1-a68b15` (eval `20261119T134112Z-test-dev-9ce125.json`);
  - 10%: `20261119T115353Z-worktree-wt-exp-pixel-share-pixshare10-s0-aad2ee` (eval `20261119T124735Z-test-dev-923920.json`)
    and `20261119T134114Z-worktree-wt-exp-pixel-share-pixshare10-s1-880ec0` (eval `20261119T143443Z-test-dev-9450dc.json`);
  - ≈ 0: `20261118T231506Z-…-headline-s0-32ff77` and `20261119T001708Z-…-headline-s1-d2c5c1`
    (commit ba81066), with their evals as listed in the headline note.
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

### The share does what it says
- **Steps 1–6000 of every new run are bitwise identical** to the ≈ 0 run with the same seed. The
  first logged difference is at step 6001, when the pixel stage starts.
- **After that, the achieved `pix_share` is exactly the target at every step:** 0.0100 or 0.1000
  (minimum over all steps > 6000 equals the target).
- **`w_pix_eff` ramps up as the decoder learns, then keeps growing as `loss/pix` falls:**
  - 1%: 440 over steps 6001–6250, 5000–5200 over the last 1000;
  - 10%: 4600, then 59,500–60,800.
- **No extra cost:** 53 min per run. (The ≈ 0 runs took about 62 min, but under a different machine load.) The share only rescales a scalar weight.

### Goal head on TEST (headline): 200 episodes, 7400 windows, t ∈ 3..39, step 24,000
Model error ÷ persistence's, s0 / s1, then the mean.

| k | ≈ 0 s0 / s1 | ≈ 0 mean | 1% s0 / s1 | 1% mean | 10% s0 / s1 | **10% mean** | 10% vs ≈ 0 |
|---:|---|---:|---|---:|---|---:|---:|
| 1 | 0.422 / 0.400 | 0.411 | 0.412 / 0.383 | 0.397 | 0.364 / 0.357 | **0.360** | −12% |
| 2 | 0.376 / 0.348 | 0.362 | 0.362 / 0.337 | 0.350 | 0.308 / 0.303 | **0.306** | −15% |
| 3 | 0.350 / 0.319 | 0.334 | 0.340 / 0.314 | 0.327 | 0.293 / 0.284 | **0.288** | −14% |
| 4 | 0.339 / 0.313 | 0.326 | 0.331 / 0.309 | 0.320 | 0.292 / 0.281 | **0.286** | −12% |
| 5 | 0.334 / 0.312 | 0.323 | 0.329 / 0.308 | 0.319 | 0.294 / 0.289 | **0.292** | −10% |
| 10 | 0.343 / 0.313 | 0.328 | 0.338 / 0.313 | 0.326 | 0.320 / 0.309 | **0.314** | −4% |
| 20 | **0.448 / 0.412** | 0.430 | 0.432 / 0.408 | 0.420 | **0.427 / 0.404** | **0.416** | −3% |

- **k = 20 (the deciding metric):** 10% is not worse. Its mean is 0.416, against 0.430, and its
  seeds (0.427 / 0.404) overlap ≈ 0's (0.448 / 0.412).
- **k = 1–5: 10% is better beyond the seed spread.** Both 10% seeds are below both ≈ 0 seeds at
  every k from 1 to 5, by 10–15% of the ratio. At k = 10 and 20 the seeds overlap.
- **1% is within the seed spread of ≈ 0 at every k**, slightly better on average (0–3%).
- **Against constant velocity (mean of seeds):**
  - k = 1: ≈ 0 2.36, 1% 2.28, 10% 2.07;
  - k = 2: 1.20, 1.16, 1.01 (10% reaches parity);
  - k = 20: 0.36, 0.35, 0.34.
- **Against the mean-parameter simulator**, the error stays 4–6× the simulator's at k = 10–20 for
  every arm. At k = 1 it is 26–27× for 10%, against 29–31× for ≈ 0.

### DEV (development only; SPEC §7.3): ratio to persistence, mean of seeds

| k | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| ≈ 0 | 0.420 | 0.370 | 0.339 | 0.323 | 0.315 | 0.303 | 0.384 |
| 1% | 0.403 | 0.359 | 0.331 | 0.315 | 0.308 | 0.301 | 0.383 |
| 10% | 0.355 | 0.304 | 0.282 | 0.274 | 0.275 | 0.287 | 0.381 |

Dev shows the same pattern: a large gain at short horizons, none at k = 20.

### Calibration and visibility (test)
The fraction of truths inside the predicted interval, s0 / s1:

| k | interval | ≈ 0 | 1% | 10% |
|---|---|---|---|---|
| 1 | 50% | 0.61 / 0.65 | 0.65 / 0.67 | 0.55 / 0.68 |
| 1 | 90% | 0.90 / 0.92 | 0.93 / 0.92 | 0.90 / 0.94 |
| 5 | 50% | 0.51 / 0.55 | 0.60 / 0.57 | 0.55 / 0.59 |
| 5 | 90% | 0.87 / 0.88 | 0.91 / 0.89 | 0.89 / 0.91 |
| 20 | 50% | 0.54 / 0.48 | 0.56 / 0.50 | 0.58 / 0.46 |
| 20 | 90% | 0.89 / 0.85 | 0.91 / 0.85 | 0.90 / 0.85 |

- **Calibration is about the same in every arm**, near nominal at 90%. The pixel term doesn't make
  the goal head over- or under-confident.
- **The visibility response is unchanged.** Predicted std is 2.5–5.1× higher when the ball is
  hidden in every arm. The 90% interval holds 72–92% of truths when the ball is hidden, in every
  arm.

### Probes: ball position relative to the camera (`pos_cam`), test R²

| level | ≈ 0 s0 / s1 | 1% s0 / s1 | 10% s0 / s1 |
|---|---|---|---|
| L1 | 0.697 / 0.674 | 0.673 / 0.678 | 0.652 / 0.663 |
| L2 | 0.673 / 0.719 | 0.682 / 0.731 | 0.746 / 0.765 |
| L3 | 0.848 / 0.869 | 0.845 / 0.862 | 0.873 / 0.890 |
| pooled | 0.884 / 0.903 | 0.877 / 0.903 | 0.896 / 0.916 |

- **At 10%, L2 and L3 hold more readable position, and L1 a little less.** L2 rises by 0.05–0.07
  and L3 by 0.02–0.03, while L1 falls by 0.01–0.05.
- **`radius` (pooled)** rises too: 0.885 / 0.890 at ≈ 0, 0.888 / 0.902 at 1%, 0.907 / 0.918 at 10%.
- **World `pos` stays unreadable** (R² ≤ 0.06) in every arm.

### Training (train split, last 1000 steps unless noted)

| | ≈ 0 s0 / s1 | 1% s0 / s1 | 10% s0 / s1 |
|---|---|---|---|
| `loss/pix` (mean abs error, frames in [0, 1]) | 0.00175 / 0.00173 | 0.00164 / 0.00164 | 0.00136 / 0.00134 |
| near relerr k1 | 0.111 / 0.107 | 0.108 / 0.106 | 0.106 / 0.100 |
| near relerr k5 | 0.362 / 0.362 | 0.358 / 0.361 | 0.353 / 0.351 |
| far relerr k20 | 0.583 / 0.564 | 0.583 / 0.557 | 0.585 / 0.560 |
| emb std L1 / L3 | 1.84 / 1.88, 1.82 / 1.84 | 1.84 / 1.87, 1.82 / 1.82 | 1.78 / 1.87, 1.78 / 1.84 |
| train `goalratio` k20, steps 20,001–24,000 | 0.483 / 0.433 | 0.467 / 0.430 | 0.464 / 0.416 |

- **The decoder reconstructs better at 10%**: `loss/pix` is 22% lower than at ≈ 0.
- **The near heads drift a little less at 10%**, with k5 at 0.35 against 0.36. The far heads are
  unchanged.
- Every term is finite in all four runs over all 24,000 steps. There is no collapse: the
  embedding std is 1.8–1.9 at every level by the end.

## Conclusion
**Refuted.** Holding the pixel term at 10% of the loss does not worsen the goal head's k = 20
error on the test terrains. Its ratio to persistence is 0.427 / 0.404 (mean **0.416**), against
0.448 / 0.412 (mean 0.430) at ≈ 0. The 1% arm is within the seed spread of ≈ 0 at every horizon,
as the hypothesis expected.

The pixel term at 10% helps, mostly at short horizons:
- The k = 1–5 ratios fall by 10–15%, with both seeds better than both ≈ 0 seeds.
- k = 2 reaches parity with constant velocity.
- Position becomes more linearly readable at L2/L3.
- The near heads drift slightly less.
- Calibration and the visibility response don't change.

SPEC §4.7's "blur" concern, that reconstruction pulls capacity away from prediction, doesn't
show up at a 10% value share and 24k steps.

These are shares of the loss **value**. How much of the encoder's gradient the pixel term
supplies isn't measured (see the [decision](../docs/decisions/2026-11-18-pixel-ablation-target-share.md)).

## Next (proposals for the user; none run)
1. **Find where the pixel term starts to hurt:** a 30% share, same setup, 2 seeds (about 2 GPU-h).
   - The k = 1 gain over ≈ 0 grew from about 3% (1% share) to 12% (10% share).
   - Does it keep growing, saturate, or turn into the blur cost at k = 20?
2. **Measure the gradient share.** Log the pixel term's and the latent terms' gradient norms at
   the encoder, a small `/builder` change.
   - That would turn "10% of the value" into something comparable across settings and runs.
3. **Should 10% become the default?**
   - The current default is a fixed `w_pix` of 0.05 (SPEC §5.4), which this and the headline
     show is effectively zero.
   - Switching the base config to `pix_share: 0.10` would change every later run. That is the
     user's decision, and it would need a `docs/decisions/` entry.
   - If it is adopted, a 48k run at 10% would show whether the short-horizon gain adds to the
     longer-training gain ([2026-11-18-headline-48k.md](2026-11-18-headline-48k.md): k = 1 0.306 at
     48k, against 0.360 here at 10% / 24k).
