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
**Commit:** **5b8c9dd**, clean (`meta.json`, both evals and all four matched JSONs: dirty false). That
is this lane merged with main 4b92689. Its training code is identical to 2fa0446's, where
[the reuse check](2026-11-20-m7-reference-reuse-check.md) passed: `git diff 2fa0446 5b8c9dd --
jepa_world/models jepa_world/train.py jepa_world/data configs/base.yaml` is empty. So the reference
stands.

**Datasets (track_full),** built 07:45:35–07:47:13 local (1 min 38 s):
`episodes-train-c9118f94ec96`, `episodes-dev-91a4e8663a21`, `episodes-test-f1adb5814534`.
- **Out-of-view frames:** none in any split (train 0 of 120,000; dev 0 of 2,400; test 0 of
  12,000). The lagged sets have 2.82%, 3.13% and 2.08% (3,380, 75 and 250 frames).
- **Occluded frames:** none in either camera's sets.
- **Same physics:** the matched script checked that the track_full and lagged test sets have the
  same terrain and ball seeds, and bitwise-identical `ball_pos` and `ball_vel`, before it scored
  each full run. It would have refused otherwise.

| seed | run | trained (local, 2026-11-20) | eval | matched (lagged flags) |
|---|---|---|---|---|
| 0 | `20261120T154715Z-worktree-wt-exp-m7-camera-m7-camera-full-s0-b4cf3e` | 07:47–09:38 (0.14 s/step) | `eval/20261120T173926Z-test-dev-76c517.json` | `matched/20261120T173930Z-test-out_of_view-0f7d91.json` |
| 1 | `20261120T173932Z-worktree-wt-exp-m7-camera-m7-camera-full-s1-0c4b84` | 09:39–11:30 | `eval/20261120T193206Z-test-dev-f449ac.json` | `matched/20261120T193209Z-test-out_of_view-2a227a.json` |

The reference's evals are `eval/20261120T003237Z-test-dev-f36d50.json` and
`eval/20261120T022242Z-test-dev-8c61ea.json` (6fd7059).

**Symmetry check: PASS, exact.** The matched script on each reference run, with its own lagged flags
(`matched/20261120T193212Z-test-out_of_view-6469e6.json` and `…193215Z-…-887e2f.json`, on MPS via
`eval.device: auto`),
reproduces its test eval's out-of-view `visibility` and `events` at every k: 630 values compared per
run, 0 differ.

**Scorers:**
- `uv run python experiments/m7_camera_check.py --lagged-eval <0f2411>/eval/<f36d50> --lagged-eval <ada237>/eval/<8c61ea> --full <b4cf3e> --full <0c4b84>`,
  new in this PR. It reads JSON only, and does the symmetry comparison above.
- `uv run python experiments/m7_ablation_check.py --arm <b4cf3e> --arm <0c4b84> --ref <0f2411> --ref <ada237> --deciding goal_k20`
  ("arm" = track_full).

### Deciding metric
**k = 1 goal-head std ratio, out of view ÷ in view, TEST, on the windows that are out of view under
track_lagged (156 of 7,400 in every run):**

| | s0 | s1 |
|---|---:|---:|
| lagged (the reference) | 8.41 | 7.72 |
| full (track_full) | 3.96 | 4.21 |

The lagged ratio is ≥ 1.2 in both seeds, and both lagged seeds exceed both full seeds: **supported,
by the pre-registered rule.** But the full model's ratio is about 4, not about 1; see the Conclusion.

### Std ratio at every k (hidden ÷ visible), TEST (s0 / s1)
| k | n hidden | lagged | full | lagged ÷ full (seed means) |
|---:|---|---|---|---:|
| 1 | 156 / 156 | 8.41 / 7.72 | 3.96 / 4.21 | 1.97 |
| 2 | 156 / 156 | 8.43 / 8.26 | 4.61 / 5.03 | 1.73 |
| 3 | 156 / 156 | 7.52 / 7.91 | 4.66 / 5.26 | 1.55 |
| 4 | 156 / 156 | 6.89 / 7.23 | 4.50 / 5.17 | 1.46 |
| 5 | 156 / 156 | 6.21 / 6.75 | 4.29 / 4.96 | 1.40 |
| 10 | 156 / 156 | 4.65 / 5.15 | 3.33 / 3.85 | 1.37 |
| 20 | 156 / 156 | 3.35 / 3.06 | 2.43 / 2.47 | 1.31 |

### The predicted std itself, m (s0 / s1)
| k | lagged, hidden | full, hidden | lagged, visible | full, visible |
|---:|---|---|---|---|
| 1 | 0.157 / 0.168 | 0.077 / 0.087 | 0.019 / 0.022 | 0.019 / 0.021 |
| 2 | 0.284 / 0.306 | 0.157 / 0.174 | 0.034 / 0.037 | 0.034 / 0.035 |
| 3 | 0.377 / 0.433 | 0.232 / 0.265 | 0.050 / 0.055 | 0.050 / 0.050 |
| 4 | 0.460 / 0.526 | 0.296 / 0.344 | 0.067 / 0.073 | 0.066 / 0.067 |
| 5 | 0.526 / 0.620 | 0.356 / 0.417 | 0.085 / 0.092 | 0.083 / 0.084 |
| 10 | 0.815 / 0.972 | 0.586 / 0.675 | 0.175 / 0.189 | 0.176 / 0.175 |
| 20 | 1.298 / 1.404 | 1.009 / 1.042 | 0.387 / 0.459 | 0.416 / 0.421 |

The in-view std is close in the two models: within 0.003 m at k = 1, and within 0.014 m up to
k = 10. So the difference in the ratio comes from the hidden windows, where the lagged model's std
is about twice the full model's at k = 1.

### Position error on the same windows, m (s0 / s1)
| k | lagged, hidden | full, hidden | lagged ÷ full, hidden (seed means) | lagged, visible | full, visible |
|---:|---|---|---:|---|---|
| 1 | 0.251 / 0.256 | 0.111 / 0.102 | 2.38 | 0.023 / 0.026 | 0.029 / 0.024 |
| 2 | 0.437 / 0.450 | 0.224 / 0.213 | 2.03 | 0.045 / 0.048 | 0.045 / 0.042 |
| 3 | 0.573 / 0.598 | 0.345 / 0.315 | 1.77 | 0.067 / 0.071 | 0.066 / 0.063 |
| 4 | 0.681 / 0.710 | 0.453 / 0.413 | 1.61 | 0.089 / 0.094 | 0.089 / 0.086 |
| 5 | 0.783 / 0.816 | 0.554 / 0.521 | 1.49 | 0.113 / 0.118 | 0.112 / 0.110 |
| 10 | 1.153 / 1.224 | 0.900 / 0.917 | 1.31 | 0.232 / 0.234 | 0.233 / 0.236 |
| 20 | 1.596 / 1.782 | 1.304 / 1.383 | 1.26 | 0.541 / 0.541 | 0.541 / 0.562 |

- **k = 5, lagged ÷ full on the hidden windows: 1.49.** This is the accuracy cost of the lagged
  camera that M7-plan asks for. At k = 1 it is 2.38.
- **On the in-view windows, the seed means differ by under 0.005 m up to k = 10** (0.011 m at
  k = 20). At k = 2–5 the full model is about 4% lower there.

### Coverage on the hidden windows, joint (s0 / s1)
Calibrated means within ±0.1 of nominal.

| k | 50%, lagged | 50%, full | 90%, lagged | 90%, full |
|---:|---|---|---|---|
| 1 | 0.28 / 0.39 | 0.46 / 0.50 | 0.78 / 0.81 | 0.80 / 0.83 |
| 2 | 0.34 / 0.43 | 0.42 / 0.53 | 0.83 / 0.79 | 0.81 / 0.88 |
| 3 | 0.37 / 0.48 | 0.38 / 0.56 | 0.79 / 0.80 | 0.81 / 0.89 |
| 4 | 0.38 / 0.46 | 0.40 / 0.50 | 0.81 / 0.81 | 0.79 / 0.88 |
| 5 | 0.37 / 0.50 | 0.40 / 0.49 | 0.79 / 0.85 | 0.76 / 0.88 |
| 10 | 0.42 / 0.44 | 0.44 / 0.46 | 0.81 / 0.88 | 0.79 / 0.87 |
| 20 | 0.47 / 0.44 | 0.50 / 0.44 | 0.87 / 0.90 | 0.90 / 0.88 |

- **The lagged model is overconfident when the ball is out of view at short horizons.** At k = 1 its
  50% coverage is 0.28 / 0.39, below the 0.40 bar in both seeds, and its 90% coverage is 0.78 / 0.81,
  below the 0.80 bar in s0.
- **The full model on the same windows is within the bar at k = 1** (0.46 / 0.50 and 0.80 / 0.83).
- **By k = 20 both are within the bar.**

### Std around exit and re-entry, k = 1 (SPEC §7.2's "rise then fall")
Mean predicted std over events, by the frame offset of the window's last context frame t from the
event. Offset 0 is the first frame with the ball out of view (exit) or back in view (re-entry), under
track_lagged. The same events and windows are used in all four runs.

**Exit:**
| run | -5 | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lagged s0 | 0.035 | 0.039 | 0.050 | 0.085 | 0.132 | 0.124 | 0.127 | 0.128 | 0.118 | 0.106 |
| lagged s1 | 0.042 | 0.045 | 0.058 | 0.091 | 0.147 | 0.162 | 0.157 | 0.144 | 0.125 | 0.104 |
| full s0 | 0.031 | 0.034 | 0.054 | 0.084 | 0.089 | 0.097 | 0.086 | 0.074 | 0.056 | 0.049 |
| full s1 | 0.032 | 0.039 | 0.059 | 0.087 | 0.108 | 0.116 | 0.095 | 0.082 | 0.065 | 0.053 |
| events per offset | 34 | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 31 | 29 |

**Re-entry:**
| run | -5 | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lagged s0 | 0.106 | 0.125 | 0.134 | 0.138 | 0.131 | 0.079 | 0.052 | 0.050 | 0.038 | 0.035 |
| lagged s1 | 0.119 | 0.141 | 0.152 | 0.148 | 0.135 | 0.094 | 0.067 | 0.055 | 0.040 | 0.038 |
| full s0 | 0.070 | 0.074 | 0.072 | 0.067 | 0.054 | 0.042 | 0.035 | 0.032 | 0.028 | 0.026 |
| full s1 | 0.081 | 0.084 | 0.087 | 0.077 | 0.056 | 0.045 | 0.038 | 0.034 | 0.030 | 0.028 |
| events per offset | 30 | 30 | 30 | 29 | 27 | 26 | 25 | 25 | 24 | 23 |

- **Before exit, the two models are the same.** From 5 frames before exit to 2 before, while the
  ball is in view in both cameras, both models' std rises from about 0.03 to 0.085. That is the
  trajectory signal.
- **From 1 frame before exit, they part.** The lagged model's std rises further, to 0.13 / 0.16,
  and stays at 0.12–0.16 through 3 frames after exit. The full model's peaks at the exit frame
  (0.097 / 0.116), then falls back to 0.056 / 0.065 by 3 frames after.
- **At re-entry, the lagged model's std drops at once.** It goes from 0.131 / 0.135 one frame
  before to 0.079 / 0.094 at the re-entry frame, and halves by one frame after (0.052 / 0.067). By
  3 frames after, it is close to the full model's (0.038 / 0.040 against 0.028 / 0.030).
- **Per-event deltas (after − before):** lagged +0.054 / +0.064 at exit and −0.070 / −0.072 at
  re-entry; full +0.015 / +0.019 and −0.036 / −0.043. They use the 32 of 41 exits and 26 of 38
  re-entries with scored windows on both sides.

### The same at k = 5
**Exit:**
| run | -5 | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lagged s0 | 0.287 | 0.345 | 0.417 | 0.517 | 0.567 | 0.466 | 0.436 | 0.431 | 0.386 | 0.346 |
| lagged s1 | 0.282 | 0.334 | 0.427 | 0.563 | 0.644 | 0.586 | 0.550 | 0.514 | 0.447 | 0.382 |
| full s0 | 0.279 | 0.360 | 0.449 | 0.505 | 0.502 | 0.481 | 0.379 | 0.330 | 0.236 | 0.203 |
| full s1 | 0.251 | 0.337 | 0.466 | 0.577 | 0.589 | 0.573 | 0.450 | 0.374 | 0.293 | 0.228 |
| events per offset | 34 | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 31 | 29 |

**Re-entry:**
| run | -5 | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lagged s0 | 0.478 | 0.491 | 0.474 | 0.449 | 0.411 | 0.278 | 0.198 | 0.200 | 0.167 | 0.161 |
| lagged s1 | 0.545 | 0.586 | 0.584 | 0.553 | 0.495 | 0.343 | 0.256 | 0.223 | 0.171 | 0.166 |
| full s0 | 0.413 | 0.392 | 0.359 | 0.296 | 0.222 | 0.171 | 0.143 | 0.123 | 0.107 | 0.108 |
| full s1 | 0.484 | 0.461 | 0.416 | 0.356 | 0.244 | 0.189 | 0.156 | 0.137 | 0.116 | 0.111 |
| events per offset | 30 | 30 | 30 | 29 | 27 | 26 | 25 | 25 | 24 | 23 |

At k = 5 both models' std peaks before exit, so the exit delta isn't a clean rise in either. After
exit, the full model's std falls faster.

### Goal head, TEST, error ÷ persistence (s0 / s1)
The full model is scored on its own track_full test frames, against the same persistence (the
physics is identical).

| k | reference | arm | ref mean | arm mean | change | verdict |
|---:|---|---|---:|---:|---:|---|
| 1 | 0.2111 / 0.2310 | 0.2280 / 0.1899 | 0.221 | 0.209 | -5.5% | within |
| 2 | 0.2055 / 0.2184 | 0.1910 / 0.1772 | 0.212 | 0.184 | -13.1% | better |
| 3 | 0.2087 / 0.2208 | 0.1932 / 0.1843 | 0.215 | 0.189 | -12.1% | better |
| 4 | 0.2122 / 0.2234 | 0.2009 / 0.1927 | 0.218 | 0.197 | -9.6% | better |
| 5 | 0.2184 / 0.2277 | 0.2093 / 0.2036 | 0.223 | 0.206 | -7.5% | better |
| 10 | 0.2471 / 0.2506 | 0.2425 / 0.2458 | 0.249 | 0.244 | -1.9% | better |
| 20 | 0.3288 / 0.3312 | 0.3252 / 0.3381 | 0.330 | 0.332 | +0.5% | within |

- **k = 20, the headline: within** (0.3252 / 0.3381 against 0.3288 / 0.3312).
- **k = 2–10: the full model is better in both seeds,** by 2–13%.
- **About half of the k = 5 gain is on the 2% of windows where the lagged camera loses the ball.**
  Weighting the hidden and in-view errors above by their counts (156 and 7,244), the mean k = 5
  error is 0.130 m lagged and 0.120 m full. The hidden windows account for 0.0055 m of that
  0.0099 m difference.

### Against the other baselines, TEST
| k | / const-vel | / sim-mean | error m arm s0 / s1 |
|---:|---|---|---|
| 1 | 1.20 (1.27) | 16.8 / 14.0 (16.3) | 0.030 / 0.025 |
| 2 | 0.61 (0.70) | 7.6 / 7.0 (8.4) | 0.049 / 0.046 |
| 3 | 0.46 (0.53) | 5.3 / 5.1 (5.9) | 0.072 / 0.069 |
| 4 | 0.40 (0.44) | 4.4 / 4.2 (4.7) | 0.097 / 0.093 |
| 5 | 0.36 (0.39) | 3.8 / 3.7 (4.1) | 0.122 / 0.118 |
| 10 | 0.29 (0.30) | 3.2 / 3.3 (3.3) | 0.247 / 0.250 |
| 20 | 0.27 (0.27) | 4.1 / 4.2 (4.1) | 0.557 / 0.579 |

### DEV (development only), error ÷ persistence (s0 / s1)
| run | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| reference | 0.216 / 0.231 | 0.212 / 0.217 | 0.215 / 0.218 | 0.216 / 0.221 | 0.220 / 0.224 | 0.246 / 0.243 | 0.330 / 0.319 |
| arm | 0.233 / 0.188 | 0.192 / 0.180 | 0.187 / 0.183 | 0.189 / 0.187 | 0.197 / 0.194 | 0.231 / 0.229 | 0.322 / 0.321 |

### Calibration, all TEST windows: joint fraction of truths inside the interval (s0 / s1)
| k | 50% ref | 50% arm | 90% ref | 90% arm |
|---:|---|---|---|---|
| 1 | 0.58 / 0.61 | 0.30 / 0.61 | 0.90 / 0.91 | 0.78 / 0.92 |
| 2 | 0.49 / 0.54 | 0.45 / 0.54 | 0.86 / 0.90 | 0.85 / 0.89 |
| 3 | 0.47 / 0.53 | 0.50 / 0.49 | 0.83 / 0.88 | 0.86 / 0.85 |
| 4 | 0.46 / 0.52 | 0.50 / 0.46 | 0.82 / 0.88 | 0.87 / 0.83 |
| 5 | 0.47 / 0.52 | 0.50 / 0.43 | 0.83 / 0.87 | 0.87 / 0.81 |
| 10 | 0.47 / 0.53 | 0.49 / 0.41 | 0.85 / 0.87 | 0.86 / 0.78 |
| 20 | 0.44 / 0.56 | 0.50 / 0.45 | 0.83 / 0.88 | 0.87 / 0.84 |

**Full s0's k = 1 interval is too narrow overall:** its 50% coverage is 0.30 and its 90% coverage
0.78, against 0.61 and 0.92 in s1. Most of that is in x (50% coverage in x on the in-view windows is
0.20). This is one seed, so it is reported, not explained. It doesn't inflate s0's deciding ratio:
s0's in-view k = 1 std is 0.019 m in both models.

### Probes, TEST: R², and Euclidean RMSE for `pos_cam` (s0 / s1)
| target | level | R^2 ref | R^2 arm | RMSE ref | RMSE arm | RMSE verdict |
|---|---|---|---|---|---|---|
| pos_cam | L1 | 0.681 / 0.649 | 0.699 / 0.661 | 0.3623 / 0.3940 | 0.3325 / 0.3506 | better |
| pos_cam | L2 | 0.771 / 0.759 | 0.759 / 0.776 | 0.3087 / 0.3127 | 0.2886 / 0.2809 | better |
| pos_cam | L3 | 0.925 / 0.907 | 0.930 / 0.937 | 0.1733 / 0.1921 | 0.1550 / 0.1491 | better |
| pos_cam | pooled | 0.936 / 0.923 | 0.946 / 0.951 | 0.1580 / 0.1736 | 0.1370 / 0.1298 | better |
| radius | pooled | 0.935 / 0.931 | 0.976 / 0.991 | 0.0150 / 0.0155 | 0.0091 / 0.0055 | better |
| pos | pooled | 0.049 / 0.044 | 0.064 / 0.035 | 5.7390 / 5.7453 | 5.6823 / 5.7669 | within |
| friction | pooled | -0.029 / -0.032 | -0.041 / -0.033 | 0.2696 / 0.2700 | 0.2713 / 0.2702 | worse |
| restitution | pooled | 0.047 / 0.055 | 0.076 / 0.117 | 0.1686 / 0.1679 | 0.1661 / 0.1623 | better |

**`pos_cam` is not like-for-like here.** It is `ball_pos − cam_pos`, and the camera moves
differently in the two modes, so the two arms probe different targets. `radius`, `pos`, `friction`
and `restitution` are the same targets in both.
- `radius` R² is higher with track_full (0.976 / 0.991 against 0.935 / 0.931).
- `friction` and `restitution` R² is below 0.12 in every run, so neither model reads them.

### Training batches, steps 47,001–48,000 (s0 / s1)
| key | reference | arm |
|---|---|---|
| relerr/k1 | 0.1376 / 0.1375 | 0.1451 / 0.1535 |
| relerr/k2 | 0.2136 / 0.2114 | 0.2253 / 0.2313 |
| relerr/k3 | 0.2896 / 0.2837 | 0.2993 / 0.3041 |
| relerr/k4 | 0.3509 / 0.3455 | 0.3623 / 0.3654 |
| relerr/k5 | 0.402 / 0.3966 | 0.4109 / 0.4136 |
| relerr/k10 | 0.5666 / 0.549 | 0.5873 / 0.5738 |
| relerr/k20 | 0.5532 / 0.5512 | 0.576 / 0.5516 |
| emb_std/L1 | 1.873 / 1.853 | 1.845 / 1.849 |
| emb_std/L2 | 1.967 / 1.983 | 1.975 / 1.95 |
| emb_std/L3 | 1.897 / 1.895 | 1.892 / 1.898 |
| logvar/k10 | -0.1951 / -0.2894 | -0.1994 / -0.2315 |
| logvar/k20 | -0.04424 / -0.04341 | 0.00418 / -0.04554 |
| loss/pix | 0.0009216 / 0.0009722 | 0.001088 / 0.001111 |
| w_pix_eff | 952.8 / 862.3 | 816.5 / 800.7 |
| pix_share | 0.1 / 0.1 | 0.1 / 0.1 |
| w_goal_eff | 0.001953 / 0.001953 | 0.001953 / 0.001953 |

`relerr` and `loss/pix` are each measured on the run's own frames and EMA targets, so they don't
compare across cameras.

### `goalratio` on training batches, mean over each 4k window (s0 / s1)
| steps | k1 ref | k1 arm | k5 ref | k5 arm | k20 ref | k20 arm |
|---|---|---|---|---|---|---|
| 4,001-8,000 | 1.051 / 1.053 | 1.057 / 1.056 | 0.988 / 0.989 | 0.989 / 0.989 | 0.988 / 0.989 | 0.989 / 0.989 |
| 8,001-12,000 | 0.999 / 1.003 | 1.001 / 1.000 | 0.976 / 0.976 | 0.974 / 0.975 | 0.977 / 0.978 | 0.978 / 0.978 |
| 12,001-16,000 | 0.927 / 0.925 | 0.941 / 0.945 | 0.918 / 0.889 | 0.925 / 0.938 | 0.944 / 0.908 | 0.952 / 0.961 |
| 16,001-20,000 | 0.537 / 0.587 | 0.549 / 0.596 | 0.444 / 0.498 | 0.457 / 0.493 | 0.599 / 0.572 | 0.595 / 0.619 |
| 20,001-24,000 | 0.400 / 0.460 | 0.378 / 0.407 | 0.332 / 0.372 | 0.316 / 0.312 | 0.464 / 0.468 | 0.446 / 0.424 |
| 24,001-28,000 | 0.332 / 0.400 | 0.307 / 0.338 | 0.291 / 0.321 | 0.272 / 0.269 | 0.399 / 0.414 | 0.389 / 0.381 |
| 28,001-32,000 | 0.289 / 0.360 | 0.264 / 0.295 | 0.266 / 0.296 | 0.247 / 0.248 | 0.373 / 0.384 | 0.365 / 0.358 |
| 32,001-36,000 | 0.262 / 0.327 | 0.236 / 0.264 | 0.251 / 0.280 | 0.232 / 0.233 | 0.357 / 0.366 | 0.350 / 0.343 |
| 36,001-40,000 | 0.245 / 0.298 | 0.218 / 0.239 | 0.239 / 0.266 | 0.220 / 0.220 | 0.347 / 0.354 | 0.338 / 0.331 |
| 40,001-44,000 | 0.235 / 0.279 | 0.203 / 0.225 | 0.231 / 0.256 | 0.211 / 0.213 | 0.338 / 0.348 | 0.329 / 0.326 |
| 44,001-48,000 | 0.227 / 0.263 | 0.195 / 0.210 | 0.227 / 0.247 | 0.206 / 0.204 | 0.332 / 0.340 | 0.323 / 0.317 |

**Training-sanity gates: PASS in both runs.**
- every logged value finite (no inf, no NaN) through step 48,000;
- `emb_std` lowest after step 2,000, ÷ its step-2,000 value: 0.83–1.02, against the 0.5 bar;
- far `logvar` last-500 mean −0.25 to 0.00, and lowest −1.67;
- `pix_share` within 1.6e-8 of 0.10;
- `w_goal_eff` 0.5/256 at every step after 4,000.

## Conclusion
**Supported, by the pre-registered rule; but only half the hypothesis holds.**

**The rule:** the lagged model's k = 1 std ratio on the out-of-view windows is 8.41 / 7.72, which is
≥ 1.2, and both seeds exceed both of the full model's (3.96 / 4.21).

**What holds:**
- **The lagged model's std rises when the ball is out of view, beyond what the trajectory explains.**
- **Its std is about twice the full model's on the same windows** (0.157 / 0.168 m against
  0.077 / 0.087 at k = 1). The in-view std is nearly the same in both models.
- **The rise and fall line up with the events.** The two models agree until 1 frame before exit.
  The lagged model's std then stays high while the ball is hidden, and halves by one frame after
  re-entry.

**What doesn't:** the hypothesis also said the full model's std "doesn't" rise on those windows.
**It does, to 4.0 / 4.2 times its in-view std,** with the ball in view the whole time. Those windows
are deep drops, and the full model is uncertain there anyway.

**So the model's uncertainty tracks both difficulty and visibility.** On these windows, difficulty
alone raises the std about fourfold, and losing sight of the ball roughly doubles it again.

**The extra std is not enough to cover the extra error.** With the ball hidden, the lagged model's
k = 1 error is 2.4 times the full model's (0.25 against 0.11 m), and its intervals undercover: 50%
coverage is 0.28 / 0.39 and 90% coverage 0.78 / 0.81. The full model is within the calibration bar
on the same windows.

**The headline is unchanged.** k = 20 is within the seed spread. The full model is 2–13% better at
k = 2–10, and at k = 5 about half of that gain comes from the 2% of windows where the lagged camera
loses the ball.

**Not done:** M7-plan suggests a bootstrap over episodes for these small counts. This note
pre-registered the both-seeds rule instead, and the gap (lowest lagged 7.72 against highest full
4.21) is wide. A bootstrap needs per-window stds, which the matched JSONs don't keep.

## Next
- **No change to the defaults.** `sim.camera_mode` stays `track_lagged`
  ([decision](../docs/decisions/2026-11-17-camera-mode-track-lagged.md)). It exists to make
  out-of-view events, and this shows the model reacts to them. The track_full gains at k = 2–10 are
  partly those same events.
- **For M7's write-up:**
  - the model's uncertainty rises with difficulty, and rises further when the ball is out of view;
  - when hidden, the k = 1–5 intervals are too narrow, so its 50% coverage is 0.28–0.50.
- **§5 (obstacles) asks the same question with occluders.** There, hiding isn't tied to deep drops.
  Its matched-window step uses this PR's `m7_matched_windows.py` on the references with
  `--flag occluded`.
- **Optional, not proposed:** an episode bootstrap of the deciding ratio. It needs the matched script
  to save per-window stds, plus four short MPS passes. It wouldn't change this verdict.
