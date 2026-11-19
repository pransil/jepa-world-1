# Does near+far JEPA training make ball position and velocity linearly decodable from each level?

- **Date:** 2026-11-17   **Lane/branch:** worktree-wt-exp-first-real-run
- **Commit:** e95576c (training, clean); 9b1fcb1 (probe script `experiments/probe_embeddings.py`, clean).
  After merging main, `window_frames` takes the `EpisodeSet` (PR #13), so the script was updated
  to match. A 6-episode smoke run gives an identical report before and after the change.
- **Config:** `configs/exp_first_real_run.yaml` (seed 0), plus `--set seed=1`
- **Run dirs** (each also holds `probes.json` and `probes_random_init.json`):
  - `20261118T060158Z-worktree-wt-exp-first-real-run-first-real-s0-f3a0d2` (seed 0)
  - `20261118T060158Z-worktree-wt-exp-first-real-run-first-real-s1-72abd4` (seed 1)
  - `20261118T060138Z-…-first-real-s0-61efab` is an aborted launch: config and meta only, no
    steps. I stopped it before it trained, because its output pipe could have killed it
    mid-run. Ignore it.

## Hypothesis
After 8000 steps of near+far training on the train split (far joining after step 2000, M4's
default loss), a linear probe on the frozen, globally pooled context embeddings decodes the
ball's velocity and position at time t better than the same probe on a randomly initialised
encoder (same architecture, same seed), at every level (L1, L2, L3, pooled).

**Deciding metric:** probe R² for `vel` and `pos` (world x, z at t), per level.
- Probes are fit on the **train** split and scored on the **dev** split (held-out terrains).
- The **test** split is scored once, as the headline. Nothing is fit or selected on it (SPEC §7.3).
- Windows: t ∈ [C−1, T−1−20] = 3..39, the baselines' window set (`jepa_world/data/DESIGN.md`).

**Refuted if:** for both seeds, the trained encoder's dev R² is at or below the random-init floor
at every level, for both targets.

**Baselines:**
- The random-init encoder is the floor.
- Persistence and physics (SPEC §7.1) score *predictions*, not probes. M6's eval doesn't run a
  model yet, so they are N/A here. Training-time `relerr/k` (error relative to the batch-mean
  target) is reported as the in-latent sanity check.

**Budget:** 2 seeds × 8000 steps, B = 32, MPS under the GPU lock, about 1 GPU-hour in total.

**Known limit:** the camera tracks the ball (`track_lagged`), so the ball sits near the frame
centre, and its world (x, z) must be read from scene content (terrain shape, camera offset), not
from where it appears in the frame. Global pooling also discards layout. Expect low position R²
for any encoder. Velocity, visible as motion across the 4 context frames, is the fairer test.

## Results

### Training (train split, MPS, B = 32, 8000 steps, far joins after 2000; about 19 min per seed)

`relerr/k` is the error relative to predicting the batch-mean target (1.0 = no better). It is
scale-free, so it is the right comparison under M4's guard, which doubles the embedding spread.
The raw `loss/k` grows with that spread.

| seed | step | relerr k1 / k5 | relerr k10 / k20 | far logvar k10 / k20 | emb std L1 / L2 / L3 |
|---|---|---|---|---|---|
| 0 | 1000 (near only) | 0.030 / 0.133 | (inactive) | ≈0 / ≈0 | 1.21 / 0.38 / 0.37 |
| 0 | 2000 | 0.036 / 0.179 | (inactive) | ≈0 / ≈0 | 1.28 / 0.38 / 0.38 |
| 0 | 2010 (far joined) | 0.32 / 0.46 | 1.12 / 1.21 | −0.42 / −0.77 | 1.12 / 0.63 / 0.69 |
| 0 | last 500 mean | 0.079 / 0.318 | 0.486 / 0.637 | −0.46 / −0.03 | 1.62 / 1.45 / 1.40 |
| 1 | 2000 | 0.040 / 0.173 | (inactive) | ≈0 / ≈0 | 1.30 / 0.41 / 0.39 |
| 1 | last 500 mean | 0.069 / 0.291 | 0.479 / 0.664 | −0.42 / +0.03 | 1.66 / 1.48 / 1.38 |

Training sanity, both seeds:
- Every value is finite.
- No collapse: embedding std rises to 1.4–1.7 at every level, and the guard is nearly idle
  (`loss/var` sampled at steps 4000, 6000 and 8000 is 0.03–0.49, against about 150 when far
  joins).
- The far heads learn on the train split (k20 relerr 0.64–0.66, vs 0.89 in the 600-step dev
  run of the M4 comparison).
- Log-variance stays near 0 to −0.5, nowhere near the ±10 bound.
- **The near heads got worse after far joined, in both seeds:** k5 0.17–0.18 → 0.29–0.32, and
  k1 0.04 → 0.07–0.08.
- The near-only phase matches M2's dev reference (relerr 1.82 → 0.15, ordered k1 < … < k5;
  `runs/20261118T055916Z-worktree-wt-gt-batches-m3-m3-near-dev-790df2`).

### Probes: R² on dev (on test in parentheses)

Ridge probes are fit on train (74,000 windows), and scored on dev (1,480) and test (7,400). Features
are each level's global spatial mean. "pooled" concatenates L1–L3.

| target | encoder | seed | L1 | L2 | L3 | pooled |
|---|---|---|---|---|---|---|
| **vel** | trained | 0 | 0.512 (0.508) | 0.426 (0.401) | 0.431 (0.409) | 0.631 (0.630) |
| | trained | 1 | 0.518 (0.475) | 0.434 (0.428) | 0.388 (0.349) | 0.629 (0.607) |
| | **random init** | 0 | **0.606** (0.611) | **0.628** (0.623) | **0.498** (0.481) | **0.718** (0.728) |
| | **random init** | 1 | **0.608** (0.624) | **0.649** (0.644) | **0.453** (0.436) | **0.725** (0.731) |
| pos | trained | 0 | 0.089 (0.047) | 0.085 (0.036) | 0.090 (0.053) | 0.114 (0.054) |
| | trained | 1 | 0.105 (0.023) | 0.045 (0.036) | 0.048 (0.053) | 0.079 (0.049) |
| | random init | 0 | 0.110 (0.062) | 0.104 (0.059) | 0.053 (0.063) | 0.098 (0.056) |
| | random init | 1 | 0.117 (0.072) | 0.092 (0.068) | 0.075 (0.049) | 0.086 (0.061) |

Persistence and physics baselines: N/A (they score predictions, not probes; see the hypothesis).

## Conclusion
**Refuted, for globally pooled linear probes.** On both seeds, dev and test, the randomly
initialised encoder decodes velocity better than the trained one at every level. Dev,
mean of two seeds:

| level | trained | random init |
|---|---|---|
| L1 | 0.52 | 0.61 |
| L2 | 0.43 | 0.64 |
| L3 | 0.41 | 0.48 |
| pooled | 0.63 | 0.72 |

Position is near zero for every encoder (R² ≤ 0.12), as expected with a ball-tracking camera
and global pooling. It doesn't separate the encoders.

What this does and doesn't show:
- 8000 steps of near+far JEPA training on the shipped loss did *not* make velocity more linearly
  available in globally pooled features. They have less of it than random conv features have.
- It does not show the embeddings lack velocity. The heads do predict future embeddings far
  better than the batch mean (relerr 0.07–0.66), so the information may be held locally, near
  the ball. A 4–6 px ball is about 1% of a 64×64 map, and the global mean dilutes it.
- Random features may also benefit from a global cue. The tracking camera scrolls the whole
  background at roughly the ball's speed, and random filters on the 4-frame stack pick that up.
  Meanwhile, the variance guard pushes every trained feature to vary across samples, which
  appearance (terrain, sky line) does more than motion. Both are hypotheses, not tested here.

## Next
1. **Probe without global pooling:**
   - a coarse spatial grid (e.g. 4×4 average pool per level);
   - or features read at the ball's pixel location (needs `ball_px` from M1's episodes, which
     the M2 loader doesn't carry yet).
   This separates "not encoded" from "diluted by pooling".
2. **Probe camera-relative targets:** ball pixel velocity, and camera motion. These say what the
   embeddings track, independent of world coordinates.
3. **Guard ablation at the same budget:** guard sum ×25 vs mean ×1. Does the guard cost the near
   heads (the k5 slip after far joins) or the probes?
4. **Longer training** (≥ 30k steps) before judging the hierarchy. 8k steps is early.

## Addendum (2026-11-18): readout probes (spatial grids, camera-relative targets)

This follows up Next 1 and 2 above. The lane was taken over after the Mac restarted, so the seed 1
readout and both random-init floors were run on 2026-11-18.
- **Commit:** ee51d13 (`experiments/probe_readout.py`, clean), for every readout below.
- **Script:** `uv run python experiments/probe_readout.py --run <run dir> [--random-init] [--device cpu]`.
- **Files:** `readout.json` and `readout_random_init.json` in each seed's run dir.
- **Device:** seed 0's trained readout embedded on MPS (2026-11-17). The others embedded on CPU, so
  they didn't contend for the GPU lock with the headline training. fp32 both ways.
- **Same protocol as the probes above:** ridge fit on train (every 2nd window, 38,000 rows), scored
  on dev (1,480) and test (7,400). Nothing is fit on test.

What changes:
- Each level is average-pooled to a g×g grid (g = 1 is the global mean above), and the pooled
  vector is probed. "pooled" concatenates L1–L3.
- Targets at t: world `pos` and `vel`; `cam_vel`, the camera's own velocity (the background
  scroll between the last two context frames); and `vel_rel` = vel − cam_vel, the ball's
  velocity relative to the camera. On the orthographic camera, that is its pixel motion.

### R², trained vs random init

Each table gives R² on **dev**, with **test** in parentheses: the pooled probe for g1, g2 and g4, and the single level for g8. The trained encoder is the step-8000 checkpoint. Random init is the same architecture with the run's seed.

**vel**: R² on dev (test)

| encoder | seed | g1 pooled | g2 pooled | g4 pooled | g8_L1 L1 | g8_L2 L2 |
|---|---|---|---|---|---|---|
| trained | 0 | 0.63 (0.63) | 0.73 (0.75) | 0.77 (0.79) | 0.76 (0.78) | 0.72 (0.72) |
| trained | 1 | 0.63 (0.61) | 0.73 (0.74) | 0.77 (0.79) | 0.77 (0.78) | 0.73 (0.72) |
| random init | 0 | 0.72 (0.72) | 0.79 (0.81) | 0.79 (0.81) | 0.79 (0.81) | 0.76 (0.79) |
| random init | 1 | 0.72 (0.73) | 0.78 (0.80) | 0.79 (0.80) | 0.78 (0.80) | 0.77 (0.79) |

**vel_rel**: R² on dev (test)

| encoder | seed | g1 pooled | g2 pooled | g4 pooled | g8_L1 L1 | g8_L2 L2 |
|---|---|---|---|---|---|---|
| trained | 0 | 0.48 (0.50) | 0.60 (0.62) | 0.64 (0.68) | 0.61 (0.67) | 0.56 (0.60) |
| trained | 1 | 0.49 (0.50) | 0.58 (0.62) | 0.63 (0.67) | 0.61 (0.65) | 0.55 (0.58) |
| random init | 0 | 0.60 (0.64) | 0.66 (0.69) | 0.65 (0.69) | 0.65 (0.71) | 0.61 (0.67) |
| random init | 1 | 0.59 (0.63) | 0.65 (0.70) | 0.65 (0.69) | 0.63 (0.70) | 0.62 (0.68) |

**cam_vel**: R² on dev (test)

| encoder | seed | g1 pooled | g2 pooled | g4 pooled | g8_L1 L1 | g8_L2 L2 |
|---|---|---|---|---|---|---|
| trained | 0 | 0.80 (0.77) | 0.90 (0.89) | 0.92 (0.92) | 0.91 (0.91) | 0.91 (0.89) |
| trained | 1 | 0.79 (0.74) | 0.89 (0.88) | 0.92 (0.92) | 0.91 (0.91) | 0.90 (0.89) |
| random init | 0 | 0.87 (0.85) | 0.94 (0.95) | 0.94 (0.95) | 0.94 (0.95) | 0.94 (0.95) |
| random init | 1 | 0.87 (0.85) | 0.93 (0.94) | 0.94 (0.95) | 0.93 (0.94) | 0.93 (0.94) |

**pos**: R² on dev (test)

| encoder | seed | g1 pooled | g2 pooled | g4 pooled | g8_L1 L1 | g8_L2 L2 |
|---|---|---|---|---|---|---|
| trained | 0 | 0.11 (0.06) | 0.09 (0.00) | -0.00 (-0.09) | 0.00 (-0.04) | -0.01 (-0.04) |
| trained | 1 | 0.08 (0.05) | 0.07 (0.03) | 0.02 (-0.07) | 0.02 (-0.02) | 0.00 (-0.06) |
| random init | 0 | 0.10 (0.05) | 0.03 (0.01) | 0.04 (-0.05) | 0.02 (0.00) | 0.05 (-0.01) |
| random init | 1 | 0.09 (0.06) | 0.05 (-0.00) | -0.01 (-0.03) | 0.00 (-0.01) | 0.05 (-0.01) |

### m6's eval check (seed 1)
`runs/…-first-real-s1-72abd4/eval/20261118T071819Z-test-dev-9ff2cb.json` (sha 492d8c9, clean). Its
probes are fit on 4,000 train windows, pooled over L1–L3, global mean. **Test** R²:
- `pos_cam` (the ball's position relative to the camera centre) 0.73;
- world `pos` 0.006;
- `radius` 0.85.

Friction and restitution are about 0. The encoder knows where the ball is *in the frame*, and its
size. It doesn't know where it is in the world, which a ball-tracking camera hides.

### The near heads regress after far joins
This is stated plainly, from the training table above. In both seeds, once the far heads join at
step 2000:
- k5 relerr goes from 0.17–0.18 to 0.29–0.32 (last 500 steps);
- k1 goes from 0.04 to 0.07–0.08.

The far heads improve over the same steps. The headline run
([2026-11-17-headline-full-model.md](2026-11-17-headline-full-model.md)) shows the same drift
continuing through 24,000 steps.

### Verdict
**Pooling explains most of the gap, but not the ordering.** The original "refuted" holds, now
with spatial readouts too. Both seeds give the same table to within 0.02.
- **Grids help trained features more than random ones.** Velocity R² on dev, mean of two seeds:

  | | global mean | 4×4 grid |
  |---|---|---|
  | trained | 0.63 | 0.77 |
  | random init | 0.72 | 0.79 |

  The trained–random gap shrinks from 0.09 to 0.02. Part of the original refutation was the
  global mean diluting information held near the ball, as the first conclusion suspected.
- **The trained encoder still never beats the random-init floor**, at any grid size, for any
  motion target: vel, vel_rel (the ball relative to the camera, which is its motion in the image)
  and cam_vel (the scroll).
  - The gap is largest for vel_rel at global pooling: 0.48 vs 0.60.
  - At 8×8 on L1 it is 0.61 vs 0.64.
- **After 8000 steps of near + far JEPA**, the context embeddings hold no *more* linearly
  readable motion than random conv features of the same architecture.
- **World position stays unreadable** for every encoder (R² ≤ 0.11). Finer grids make the test
  score negative (−0.09 at g4, trained), which is ridge overfitting to the training terrains.
  m6's `pos_cam` result above shows the position information is there in camera coordinates. With
  a tracking camera, world position simply isn't in the frame.

One cross-reference, not a controlled comparison:
- The headline full-model run ([2026-11-17-headline-full-model.md](2026-11-17-headline-full-model.md):
  near + far + goal + pixel, 24,000 steps) reaches pooled test `pos_cam` R² 0.88–0.90, against 0.73
  here. That's with eval's 4000-window probes in both cases.
- Longer training or the goal head, or both, makes the embeddings more position-aware. The two
  aren't separated.

### Proposals for the user (not run; each needs a go-ahead)
1. **Guard ablation, at the same budget.** Variance guard summed ×25 (the current default) vs mean
   ×1 (M3's original), near + far, 8000 steps, 2 seeds.
   - Question: does the guard cause the near-head regression after far joins, or the probe gap?
   - Deciding metric: k5 relerr (dev), last 500 steps; secondary, g4 pooled vel R² (dev).
   - Cost: 4 runs × about 19 min, about 1.3 GPU-h.
2. **Longer training: ≥ 30,000 steps** before judging the hierarchy.
   - Question: do the probes and the far heads keep improving past 8000 steps?
   - The headline's goal head only started beating persistence on train batches after
     step 12,000.
   - Deciding metric: g4 pooled vel R² and k20 relerr (dev).
   - Cost: about 75–95 min per seed for near + far at 150 ms/step, 2 seeds.
