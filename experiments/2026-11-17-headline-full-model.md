# Headline: does the full model's goal head beat persistence and the baselines on held-out terrains?

- **Date:** planned 2026-11-17, run 2026-11-18   **Lane/branch:** worktree-wt-exp-headline
  (taken over from the m3 session after the Mac restarted)
- **Commit:** ba81066 (clean), for training and eval of both seeds
- **Config:** `configs/exp_headline.yaml` (seed 0), plus `--set seed=1`
- **Run dirs** (each holds `eval/<time>-test-dev-<id>.json` from
  `uv run python -m jepa_world.eval --run <dir> --split test --split dev`):
  - `20261118T231506Z-worktree-wt-exp-headline-headline-s0-32ff77` (seed 0; eval `20261119T012311Z-test-dev-6c1fea.json`)
  - `20261119T001708Z-worktree-wt-exp-headline-headline-s1-d2c5c1` (seed 1; eval `20261119T012436Z-test-dev-3e244c.json`)
  - `20261118T224604Z-…-headline-s0-e4b6c7` is an aborted first launch. Claude Code's
    low-memory guard killed it at step ~5750, while three readout processes from another lane ran
    alongside. Ignore it.

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

### Goal head on TEST (headline): 200 episodes, 7400 windows, t ∈ 3..39, checkpoint at step 24000

The baselines are scored on the same windows in both evals, so their ratios are the same for both
seeds. "/persistence" below 1 beats persistence.

| k | error m s0 / s1 | model / persistence s0 / s1 | **mean** | const-vel / pers. | sim-mean / pers. | model / const-vel (mean) | model / sim-mean (mean) | coverage 50% s0 / s1 | coverage 90% s0 / s1 |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | 0.056 / 0.053 | 0.422 / 0.400 | **0.411** | 0.174 | 0.0135 | 2.36 | 30.3 | 0.61 / 0.65 | 0.90 / 0.92 |
| 2 | 0.097 / 0.089 | 0.376 / 0.348 | **0.362** | 0.302 | 0.0252 | 1.20 | 14.3 | 0.56 / 0.60 | 0.88 / 0.91 |
| 3 | 0.130 / 0.119 | 0.350 / 0.319 | **0.334** | 0.407 | 0.0362 | 0.82 | 9.2 | 0.55 / 0.58 | 0.88 / 0.90 |
| 4 | 0.163 / 0.151 | 0.339 / 0.313 | **0.326** | 0.494 | 0.0459 | 0.66 | 7.1 | 0.52 / 0.55 | 0.87 / 0.89 |
| 5 | 0.194 / 0.182 | 0.334 / 0.312 | **0.323** | 0.568 | 0.0546 | 0.57 | 5.9 | 0.51 / 0.55 | 0.87 / 0.88 |
| 10 | 0.349 / 0.319 | 0.343 / 0.313 | **0.328** | 0.834 | 0.0751 | 0.39 | 4.4 | 0.51 / 0.55 | 0.87 / 0.88 |
| 20 | 0.768 / 0.706 | 0.448 / 0.412 | **0.430** | 1.211 | 0.0800 | 0.36 | 5.4 | 0.54 / 0.48 | 0.89 / 0.85 |

- **The two seeds agree to within 0.04** at every k.
- **Constant velocity from the context positions only** (`const_velocity_fd`, the fair comparison:
  the model also sees only frames t−3…t), model / baseline:
  - k = 1: 1.44 (s0) and 1.37 (s1);
  - k = 2: 0.95 and 0.88;
  - beaten from k = 2 on.

  True-v const-vel uses the simulator's state at t, which the model never sees.
- **The simulator with true parameters** has about zero error. Its ratio column (10⁴–10⁸) carries
  no information, so it is left out.

### Goal head on DEV (development only, never a headline; SPEC §7.3): 40 episodes, 1480 windows

| k | model / persistence s0 / s1 | mean | model / const-vel (mean) | model / sim-mean (mean) | coverage 50% s0 / s1 | coverage 90% s0 / s1 |
|---:|---|---:|---:|---:|---|---|
| 1 | 0.420 / 0.420 | 0.420 | 2.68 | 35.7 | 0.63 / 0.63 | 0.93 / 0.92 |
| 5 | 0.318 / 0.312 | 0.315 | 0.60 | 6.7 | 0.58 / 0.59 | 0.93 / 0.90 |
| 10 | 0.306 / 0.300 | 0.303 | 0.41 | 4.8 | 0.64 / 0.63 | 0.94 / 0.92 |
| 20 | 0.390 / 0.379 | 0.384 | 0.36 | 5.7 | 0.62 / 0.55 | 0.93 / 0.89 |

(k = 2–4 behave the same way. The full table is in the eval JSONs.)

### Visibility: does uncertainty rise when the ball is out of view? (test)

156 of 7400 windows have the ball out of view at t. No window is `occluded`, because the base config
has no obstacles.

| k | `std_ratio` (hidden ÷ visible) s0 / s1 | 90% coverage, hidden s0 / s1 | 90% coverage, visible s0 / s1 |
|---:|---|---|---|
| 1 | 3.58 / 4.45 | 0.72 / 0.87 | 0.91 / 0.92 |
| 5 | 4.50 / 4.90 | 0.74 / 0.82 | 0.88 / 0.88 |
| 10 | 3.81 / 4.35 | 0.79 / 0.88 | 0.87 / 0.88 |
| 20 | 2.49 / 3.11 | 0.88 / 0.90 | 0.89 / 0.85 |

**Events** (k = 1, the predicted std 5 frames before vs after; test, 41 exits, 38 re-entries):
- when the ball leaves the frame, the std rises: 0.113 → 0.127 (s0) and 0.123 → 0.164 (s1);
- when it comes back into view, the std falls: 0.134 → 0.094 and 0.165 → 0.108.

The same rise-then-fall holds at every k up to 10 in both seeds. At k = 20, seed 0's exit
std dips slightly (1.51 → 1.43).

### Training (train split, MPS, B = 32; about 62 min per seed at about 150 ms/step, load average about 5–8)

Per-window means from `metrics.jsonl`. `relerr` is scale-free: 1.0 means no better than the batch
mean target.

| window (steps) | relerr k1 s0 / s1 | relerr k5 s0 / s1 | relerr k20 s0 / s1 | emb std L1 / L3 (s0) | loss/var s0 / s1 |
|---|---|---|---|---|---|
| 1500–2000, near only | 0.047 / 0.047 | 0.199 / 0.195 | (inactive) | 1.28 / 0.38 | 0.18 / 0.18 |
| 3500–4000, +far | 0.065 / 0.048 | 0.263 / 0.222 | 0.685 / 0.746 | 1.47 / 1.25 | 0.81 / 1.01 |
| 5500–6000, +goal | 0.075 / 0.063 | 0.305 / 0.288 | 0.656 / 0.699 | 1.55 / 1.33 | 0.42 / 0.41 |
| 7500–8000, +pixel | 0.068 / 0.070 | 0.306 / 0.293 | 0.625 / 0.669 | 1.65 / 1.40 | 0.17 / 0.21 |
| last 1000 (23k–24k) | 0.111 / 0.107 | 0.362 / 0.362 | 0.583 / 0.564 | 1.84 / 1.88 | 0.09 / 0.10 |

**The goal head was slow to learn.** On training batches, `goalratio` (error ÷ persistence's) stayed
near 1.0 until about step 12,000. For seed 0 it then averaged 0.50–0.60 over steps 16k–20k
(k = 1, 5, 20), and 0.39–0.48 over 20k–24k. It
was still falling at step 24,000, so the run is not converged for the goal head.

**Sanity:**
- Every term is finite in both seeds, over all 24,000 steps.
- No collapse: embedding std rises to 1.8–1.9 at every level, and the guard is nearly idle at the
  end (`loss/var` about 0.1).
- After step 6000, `loss/total` ranges over 430–1640 and drifts up with the embedding spread: seed
  0's mean goes from 684 (steps 6000–6500) to 848 (the last 1000). The raw latent losses scale with
  the spread, so `relerr` is the comparable measure.

**`loss/pix`'s share of `loss/total`:** after the pixel decoder joins at step 6000,
`w_pix · loss/pix / |loss/total|` peaks at 3.2e-5 (s0) and 2.1e-5 (s1), with a mean of about 2e-7.
`loss/pix` itself falls from 0.035–0.039 to 0.0017–0.0018 (mean absolute error, frames in [0, 1]).
The pixel term doesn't dominate (SPEC §8 M5). At w_pix = 0.05 it is effectively absent from
the encoder's gradient.

**Near heads when goal and pixel join:**
- When **goal** joins at step 4000, k5 relerr goes from 0.263 / 0.222 to 0.305 / 0.288, and k1 from
  0.065 / 0.048 to 0.075 / 0.063.
- When **pixel** joins at step 6000, nothing more changes: k5 0.306 / 0.293.
- After that, both drift steadily worse until the end: k5 0.362 and k1 about 0.11. Meanwhile k20
  keeps improving, to 0.58 / 0.56.

So yes, the near heads regress again. Goal adds a step, as far did in
[the first real run](2026-11-17-first-real-run.md). It isn't a one-off at a stage join: it keeps
going while the embedding spread grows.

### Probes (eval's linear probes, 4000 train windows, pooled L1–L3 global mean; test R²)

| target | s0 | s1 | first real run, s1 (near + far, 8k steps) |
|---|---|---|---|
| `pos_cam` (ball position relative to the camera) | 0.884 | 0.903 | 0.733 |
| `radius` | 0.885 | 0.890 | 0.85 |
| world `pos` | 0.052 | 0.054 | 0.006 |
| friction / restitution | −0.01 / 0.03 | −0.04 / 0.02 | ≈ 0 |

## Conclusion

**Supported**, on all three parts of the hypothesis, for both seeds, on the held-out test terrains.
1. **Beats persistence at every horizon.** The goal-head error ratio is 0.31–0.45 for every k and both
   seeds; the worst is 0.448, at k = 20 for seed 0. The mean over seeds is 0.411, 0.362, 0.334,
   0.326, 0.323, 0.328 and 0.430 for k = 1, 2, 3, 4, 5, 10, 20. Nothing is near the refutation
   line of 1.
2. **Beats constant velocity at k ≥ 5:** 0.57 at k = 5, 0.39 at k = 10, 0.36 at k = 20. In fact it
   wins from k = 3 (0.82).
   - At k = 1–2 it loses to the true-velocity baseline (2.36, 1.20), which reads the simulator's
     state.
   - Against constant velocity from context positions only, it loses only at k = 1 (1.37–1.44).
3. **Does not reach the simulator bar.** Its error is 4.4–30× the mean-parameter simulator's,
   which knows the terrain.

The uncertainty is useful:
- Overall 90% coverage is 0.85–0.92 against a nominal 0.90.
- The predicted std is 2.5–4.9× higher when the ball is out of view, rises after it leaves and falls
  after it reappears.

But it is **overconfident when the ball is hidden**: 90% coverage there is 0.72–0.79 for seed 0 at
k ≤ 10 (0.88 at k = 20), and 0.82–0.90 for seed 1, over 156 windows. Dev numbers agree with test (k = 5 ratio 0.315 dev vs 0.323
test).

## Next

These are proposals for the user, not run. Each needs a go-ahead.
1. **Train longer:** 48k steps, 2 seeds, about 2 h each at 150 ms/step.
   - Why: the goal head only started beating persistence on training batches after step
     12,000, and was still improving at 24,000.
   - Deciding metric: the k = 20 test ratio to persistence (now 0.43). Watch the near heads too.
2. **Guard ablation:** sum ×25 vs mean ×1. This is the same proposal as in the first real run's
   addendum. The near-head relerr keeps rising as the embedding spread grows under the guard.
3. **Pixel weight:** at w_pix = 0.05, the pixel term is about 1e-5 of the loss, so the M7 pixel
   ablation's "low" setting is in effect 0. Even 0.5 would be about 1e-4. The ablation's levels
   (docs/milestones/M7-plan.md §1) probably need rescaling, for example normalising each term or
   choosing w_pix by the term's share. That's the user's call.
4. **Calibration when the ball is out of view:** seed 0 under-covers there, 0.72 at k = 1. That's
   worth a look in M6/M7 (the camera-mode ablation, M7-plan §3).
