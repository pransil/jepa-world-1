# M7 §5: when an obstacle hides the ball, does the model's uncertainty rise, fall back, and stay calibrated?

- **Date:** 2026-11-20   **Lane/branch:** worktree-wt-exp-m7-obstacles
- **Commit:** the one this lane has after merging origin/main right before its first job. It is
  clean, and filled in under Results.
- **Config:** `configs/m7_obstacles.yaml` (seed 0), plus `--set seed=1`.
  - It is the M7 reference `configs/exp_headline_48k.yaml` plus the `sim.obstacles` block: 3–6 posts
    per terrain, 0.8–1.6 m wide, with a 1 m gap.
  - The camera stays `track_lagged` (SPEC's `track_lagged_obstacles`).
  - The block keys its own datasets, which are built before training. They aren't built yet, so
    `train.check_config` refuses the config until then.
- **Reference (no obstacles):** the 48k runs at the new defaults (commit 6fd7059;
  [note](2026-11-19-headline-48k-new-defaults.md)):
  `20261119T224253Z-…-headline48k-new-s0-0f2411` and `20261120T003253Z-…-headline48k-new-s1-ada237`.
  They stand only if [the reuse check](2026-11-20-m7-reference-reuse-check.md) passes.
- **Plan:** [M7-plan.md](../docs/milestones/M7-plan.md) §5. Obstacles are off when the key is
  absent ([decision](../docs/decisions/2026-11-17-obstacles-dataset-keying.md)); the design is in
  `jepa_world/sim/DESIGN.md`, "Obstacles and occlusion".
  - The occluders don't collide, so physics and ground truth are unchanged.
  - Tuning gave 7.7% of frames occluded over terrain seeds 0–199.
- **Asked for:** the user, 2026-11-19 ("the remaining M7 ablations", relayed by the orchestrator
  session).

## Hypothesis and rules (pre-registered; the thresholds are M7-plan §5's, unchanged)
**The obstacle-trained model's goal-head std rises on occluded windows and falls back after the
ball reappears, and its intervals stay calibrated there.**

On the **test** split, at **k = 1**, per run, from its eval (`experiments/m7_occlusion_check.py`):
1. **Rise (deciding metric):** predicted std on windows whose last context frame is `occluded`, ÷
   std on the other windows (`visibility["1"]["occluded"]["std_ratio"]`), is **≥ 1.2**.
2. **Calibration:** the joint 90% interval holds **≥ 0.8** of the truths on occluded windows.
3. **Recovery:** the mean std 3 frames after the ball reappears (eval's `occluded` offset curve at
   +3), ÷ the std on non-occluded windows, is **≤ 1.2**.

**Verdict:**
- **supported** if all three hold in both seeds;
- **refuted** if any rule fails in both seeds;
- **mixed** otherwise.

**Disclosed before the runs: the same rules on the reference's out-of-view windows.** That is the
ball leaving the frame under `track_lagged`, not occlusion. I applied the scorer to them as a code
check; seed 0 gave:
- rise **8.41** (holds);
- 90% coverage **0.78** (fails, just under 0.8);
- recovery **2.02** at +3 (fails). The std three frames after re-entry was still twice the
  visible level.

So rules 2 and 3 are demanding. If occlusion behaves like leaving the frame, expect them to fail.
The thresholds stay as M7-plan set them on 2026-11-17, before any data.

**Reported, no prediction:**
- every horizon's occluded and out-of-view numbers: n, std hidden and visible, ratio, 50% and 90%
  coverage, error;
- the full k = 1 onset and offset curves (±5 frames);
- **the accuracy cost of occlusion:** k = 5 position error on occluded windows, against the
  reference model on the **same physical windows**, whose frames have no occluder;
  - scored by `experiments/m7_matched_windows.py` (M7 §3's lane): the reference run, its own
    frames, `--flags-config configs/m7_obstacles.yaml --flag occluded`;
  - the script refuses unless the obstacle test set's seeds, `ball_pos` and `ball_vel` equal the
    base test set's exactly. That is M7-plan's GT check before sharing the baseline table;
- **the k = 20 headline ratio,** on the obstacle test frames against the same persistence (same
  physics);
- goal head, dev, the baselines, probes and training tables, from
  `experiments/m7_ablation_check.py --deciding goal_k20`. Its "DECIDING" line is the k = 20
  comparison, not this note's;
- a comparison with §3's out-of-view behaviour, for M7's write-up.

**Training-sanity gates:** the reference's A1–A5 thresholds, applied by that scorer. A run that
fails a gate is reported, and not used for a verdict.

## Procedure
One GPU job at a time, each only with no other project python job (`tools/gpu-status.sh`) and
≥ 10 GB free. The worktree stays clean throughout.
1. **Build the obstacle datasets:**
   `uv run python -m jepa_world.data.build --config configs/m7_obstacles.yaml --split all`.
   It takes the GPU lock, and should take about 5 min. Report the fraction of occluded frames per
   split.
2. **Train s0:** `uv run python -m jepa_world.train --config configs/m7_obstacles.yaml --name m7-obstacles-s0`.
   Then eval it (`--split test --split dev`).
3. **Train s1** (`--set seed=1 --name m7-obstacles-s1`), then eval it.
4. **Matched windows**, once §3's PR has merged the script: both reference runs, with
   `--flags-config configs/m7_obstacles.yaml --flag occluded`.
5. **Score:** `uv run python experiments/m7_occlusion_check.py --run <s0> --run <s1>`, plus
   `experiments/m7_ablation_check.py` and the matched JSONs.

**Budget:** one dataset build (≈ 5 min), 2 × 48,000 steps at about 1.8 h, 2 evals and 2 matched
runs of a few minutes each. About 4 GPU-h.

## Results
(Filled in after the runs.)

## Conclusion
(Filled in after the runs.)

## Next
(Filled in after the runs.)
