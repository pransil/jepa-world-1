# Does near+far JEPA training make ball position and velocity linearly decodable from each level?

- **Date:** 2026-11-17   **Lane/branch:** worktree-wt-exp-first-real-run   **Commit:** RUN_SHA (training), PROBE_SHA (probes)
- **Config:** `configs/exp_first_real_run.yaml` (seed 0), plus `--set seed=1`
- **Run dirs:** RUN_DIRS

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

RESULTS

## Conclusion

CONCLUSION

## Next

NEXT
