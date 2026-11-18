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
