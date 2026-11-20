# M7 §3: does the model's uncertainty track visibility, or trajectory difficulty?

- **Date:** 2026-11-20   **Lane/branch:** worktree-wt-exp-m7-camera
- **Commit:** the one this lane has after merging origin/main right before its first job. It is
  clean, and filled in under Results.
- **Config:** `configs/m7_camera_full.yaml` (seed 0), plus `--set seed=1`. It is the M7 reference
  `configs/exp_headline_48k.yaml` with only `sim.camera_mode: track_full`. That was checked by
  diffing every leaf of the two resolved configs.
  - It keys its own datasets, which are built before training.
  - `train.check_config` refuses it until they exist; that is the only check it fails now.
- **Reference (track_lagged):** the 48k runs at the new defaults (commit 6fd7059;
  [note](2026-11-19-headline-48k-new-defaults.md)):
  `20261119T224253Z-…-headline48k-new-s0-0f2411` and `20261120T003253Z-…-headline48k-new-s1-ada237`.
  They stand only if [the reuse check](2026-11-20-m7-reference-reuse-check.md) passes.
- **Plan:** [M7-plan.md](../docs/milestones/M7-plan.md) §3, with the
  [decision](../docs/decisions/2026-11-17-camera-mode-track-lagged.md).
- **Asked for:** the user, 2026-11-19 ("the remaining M7 ablations", relayed by the orchestrator
  session).

## Why this ablation separates the two
Under `track_lagged`, the ball leaves the frame only on deep drops (M1.md), and deep drops are
also hard to predict. So "uncertainty rises when the ball is out of view" can't tell visibility
from difficulty. `track_full` keeps the ball in frame on exactly the same trajectories: camera mode
doesn't change physics. So scoring both models on the **same (episode, t) windows** separates them.
Those are the windows where the ball is out of view under `track_lagged`, and each model is scored
on its own frames.

## The matched-window scorer (new: `experiments/m7_matched_windows.py`)
The script scores a run's goal head on one dataset's frames, with the hidden windows flagged by
another dataset's ground truth.
- **Same windows and metrics as eval:** `valid_starts`, `model_outputs`, `goal_positions`,
  `metrics.visibility`, `metrics.event_behaviour`.
- **Refuses** unless the two datasets have the same seeds and bitwise-identical `ball_pos` and
  `ball_vel`. That is also the GT check M7-plan requires before the baseline table is shared.
- **Writes to `<run>/matched/`,** never into `eval/`.
- **Tests** (`tests/test_m7_matched_windows.py`, 5, fast tier), all seen failing against a stub:
  - on its own flags it equals eval's `visibility` and `events` exactly;
  - flags come only from the flags set, and frames only from the frames set;
  - it refuses sets that differ in seeds, `ball_pos` or `ball_vel`;
  - it accepts only a known flag;
  - the file and metadata it writes.
- **Three deliberate breaks were each caught:** scoring the flags set's frames, taking the flag
  from the frames set, and dropping the same-episodes check.
- **Real-data check, before this note:** the reference s0, on its own dev split and flag, run on
  CPU. It reproduced its MPS eval's dev `visibility` and `events` for out-of-view:
  - 630 values compared;
  - integer counts identical;
  - largest relative float difference 1.6e-6.

## Hypothesis and rules (pre-registered)
**The lagged model's goal-head std rises when the ball is out of view, and the full model's
doesn't on the same windows: the rise comes from visibility, not difficulty.**

- **Deciding metric (SPEC §7.2, event behaviour):** the goal-head predicted std at **k = 1** on
  out-of-view windows ÷ the same on in-view windows (`std_ratio`), on the **test** split.
  - The out-of-view windows are those whose last context frame t is `out_of_view` in the
    **track_lagged** test set.
  - Lagged: the reference runs, from their eval (`visibility["1"]["out_of_view"]["std_ratio"]`).
    That is **8.4 / 7.7**, on 156 hidden windows.
  - Full: the track_full runs, scored by the matched-window script with
    `--flags-config configs/exp_headline_48k.yaml --flag out_of_view`.
- **Verdict:**
  - **supported** if the lagged ratio is ≥ 1.2 in both seeds, and both lagged seeds exceed both
    full seeds;
  - **refuted** if the lagged ratio is < 1.2, or the two overlap (M7-plan §3).
  - The lagged side already meets the 1.2 bar, so the verdict turns on the full model's ratio.
- **Reported, no prediction:**
  - **coverage** of the 50% and 90% intervals on out-of-view windows, lagged. Calibrated means
    within ±0.1 of nominal; the reference's eval has 90% at 0.78 / 0.81;
  - the **same for the full model** on the same windows;
  - **std around exit and re-entry** (the matched `events`: onset and offset curves, ±5 frames),
    for both models: the "rise then fall" of §7.2;
  - **k = 5 position error on out-of-view windows,** lagged ÷ full: the accuracy cost of the lagged
    camera;
  - **the k = 20 headline ratio** for both. The full model is scored on its own track_full test
    frames, against the same persistence (same physics);
  - goal head k = 1–20, dev, the baselines, probes and training tables, from
    `experiments/m7_ablation_check.py --deciding goal_k20`. Its "DECIDING" line is the k = 20
    comparison, not this note's deciding metric.
- **Training-sanity gates:** the reference's A1–A5 thresholds, applied by that scorer. A run that
  fails a gate is reported, and not used for a verdict.
- **Small numbers:** only 156 of the 7,400 test windows are out of view. The both-seeds rule is the
  decision; the counts are reported next to every ratio.

## Procedure
One GPU job at a time, each only with no other project python job (`tools/gpu-status.sh`) and
≥ 10 GB free. The worktree stays clean throughout.
1. **Build the track_full datasets:**
   `uv run python -m jepa_world.data.build --config configs/m7_camera_full.yaml --split all`.
   It takes the GPU lock; M2 built a full set in under 5 min. Then report the fraction of
   `out_of_view` frames in the track_full test set, which should be zero or near it.
2. **Train s0:** `uv run python -m jepa_world.train --config configs/m7_camera_full.yaml --name m7-camera-full-s0`.
   Then eval it (`--split test --split dev`).
3. **Matched windows s0:** `uv run python experiments/m7_matched_windows.py --run <s0> --flags-config configs/exp_headline_48k.yaml --flag out_of_view`.
   This refuses if the track_full and lagged test sets differ in physics.
4. **Train s1** (`--set seed=1 --name m7-camera-full-s1`), then eval and matched windows, as for s0.
5. **Symmetry check:** the matched-window script on both reference runs with their own flags
   (`--flags-config configs/exp_headline_48k.yaml`). It must reproduce their test eval's
   out-of-view numbers; on MPS, as they were scored, the match should be exact.
6. **Score** with `experiments/m7_ablation_check.py`, and from the matched JSONs.

**Budget:** one dataset build (≈ 5 min), 2 × 48,000 steps at about 1.8 h, 2 evals and 4 matched
runs of a minute or two each. About 4 GPU-h.

## Results
(Filled in after the runs.)

## Conclusion
(Filled in after the runs.)

## Next
(Filled in after the runs.)
