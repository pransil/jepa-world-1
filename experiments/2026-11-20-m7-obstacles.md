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
**Commit:** **74d1925**, clean (`meta.json`, both evals and both matched files: dirty false). That
is this lane merged with main 6351464. Its training code is identical to 2fa0446's, where
[the reuse check](2026-11-20-m7-reference-reuse-check.md) passed: `git diff 2fa0446 74d1925 --
jepa_world/models jepa_world/train.py jepa_world/data configs/base.yaml` is empty. So the reference
stands.

**Obstacle datasets** (built 17:04:04–17:05:45 local, 1 min 41 s): `episodes-train-a40f2d2a2106`,
`episodes-dev-0319737ff9e9`, `episodes-test-ebd6e53c63a1`.

| split | episodes | `occluded` frames | `out_of_view` frames |
|---|---:|---:|---:|
| train | 2,000 | 11,051 (9.21%) | 3,380 (2.82%) |
| dev | 40 | 221 (9.21%) | 75 (3.13%) |
| test | 200 | 1,112 (9.27%) | 250 (2.08%) |

The `out_of_view` counts equal the reference datasets' exactly, as they should: the posts don't
move the ball or the camera. Tuning had given 7.7% occluded on terrain seeds 0–199.

| seed | run | trained (local, 2026-11-20) | eval |
|---|---|---|---|
| 0 | `20261121T010547Z-worktree-wt-exp-m7-obstacles-m7-obstacles-s0-37a557` | 17:05–18:56 | `eval/20261121T025758Z-test-dev-feca05.json` |
| 1 | `20261121T025800Z-worktree-wt-exp-m7-obstacles-m7-obstacles-s1-37d120` | 18:57–20:47 | `eval/20261121T044924Z-test-dev-2efee7.json` |

The reference's evals are `eval/20261120T003237Z-test-dev-f36d50.json` and
`eval/20261120T022242Z-test-dev-8c61ea.json` (6fd7059). Its matched-window scores on the obstacle
set's `occluded` windows, made at 74d1925, are `matched/20261121T044927Z-test-occluded-e59741.json`
(0f2411) and `matched/20261121T044930Z-test-occluded-4f5b6e.json` (ada237). The script checks that
the obstacle test set's seeds, `ball_pos` and `ball_vel` equal the base test set's before writing,
and it wrote both, so that check passed.

**Scorers:**
- `uv run python experiments/m7_occlusion_check.py --run <37a557> --run <37d120>`: the three rules.
- `uv run python experiments/m7_ablation_check.py --arm <37a557> --arm <37d120> --ref <0f2411> --ref <ada237> --deciding goal_k20`
  ("arm" = the obstacle model). Its DECIDING line is the k = 20 comparison, reported below; it is
  not this note's verdict.
- The matched-window tables join the two matched JSONs with the two evals' `occluded` blocks (606
  windows in each, the same 606). The join is a scratch script, not committed: every number is
  one field of one of those four files (`error_m_hidden`, `error_m_visible`, `std_ratio`, and the
  k = 1 `events` curves ÷ `std_visible`).

### Deciding metric: the three rules, TEST, k = 1
| rule | bar | s0 | s1 |
|---|---|---:|---:|
| 1. rise: std on occluded windows ÷ std on the others | ≥ 1.2 | **1.90** holds | **2.09** holds |
| 2. calibration: joint 90% coverage on occluded windows | ≥ 0.8 | **0.78** fails | **0.92** holds |
| 3. recovery: std 3 frames after the ball reappears ÷ std on the others | ≤ 1.2 | **1.26** fails | **1.24** fails |

Recovery fails in both seeds: **refuted**, by the pre-registered rule. Rise holds in both seeds, and
calibration splits.

**Training-sanity gates: PASS in both runs** (below), so the verdict stands on usable runs.

### What the matched windows add to the recovery rule (description, not a test)
The reference model has never seen a post. On the same 606 physical windows, rendered without
posts, its k = 1 std is **0.83 / 0.84** of its std on the other windows. But around the moments the
ball would vanish and reappear, it is higher than its average, with no occluder at all:

**k = 1 std around occlusion onset, as a multiple of each run's std on the other windows**
(offset 0 = the first fully hidden frame, as the last context frame):

| run | -5 | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference, no posts s0 | 1.40 | 1.36 | 1.34 | 1.29 | 1.18 | 1.11 | 1.12 | 1.10 | 1.11 | 1.15 |
| reference, no posts s1 | 1.39 | 1.37 | 1.31 | 1.27 | 1.15 | 1.09 | 1.17 | 1.17 | 1.16 | 1.17 |
| obstacle model s0 | 1.37 | 1.41 | 1.52 | 1.70 | 2.08 | 2.18 | 2.16 | 2.13 | 1.97 | 1.90 |
| obstacle model s1 | 1.57 | 1.60 | 1.66 | 1.93 | 2.31 | 2.48 | 2.50 | 2.36 | 2.23 | 1.99 |

**Around the offset** (offset 0 = the first frame the ball is visible again):

| run | -5 | -4 | -3 | -2 | -1 | +0 | +1 | +2 | +3 | +4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference, no posts s0 | 1.06 | 0.97 | 0.97 | 1.03 | 1.10 | 1.11 | 1.12 | 1.20 | 1.30 | 1.22 |
| reference, no posts s1 | 1.04 | 0.99 | 1.00 | 1.06 | 1.12 | 1.16 | 1.16 | 1.31 | 1.30 | 1.23 |
| obstacle model s0 | 1.82 | 1.93 | 2.17 | 2.15 | 2.17 | 1.95 | 1.46 | 1.37 | 1.26 | 1.27 |
| obstacle model s1 | 1.99 | 2.20 | 2.34 | 2.39 | 2.46 | 2.24 | 1.48 | 1.37 | 1.24 | 1.24 |

- **At +3 after reappearance, the reference is at 1.30 / 1.30,** above the obstacle model's 1.26 /
  1.24. So by +3 the obstacle model's std is back to what these moments cost a model with no
  occluders. Its excess is in frames 0 to +2.
- **The rule compares with the model's own other windows,** so it fails, and the verdict stays
  refuted. Reading recovery against the matched baseline instead is a post-hoc choice for the user,
  and would need new data to count as a test.
- **The std starts rising before the ball is fully hidden** (from offset −3, 1.52 / 1.66), while
  the reference's falls on the same frames. `occluded` means no pixel visible, so in those frames
  the ball is likely already partly behind the post. The posts themselves are always in view.

### Calibration on occluded windows: s0's failure isn't specific to occlusion
s0's k = 1 intervals are too narrow on **every** window: joint 90% coverage **0.73** on all test
windows, against 0.78 on occluded ones (calibration table below; the reference gets 0.90 / 0.91).
§3's track_full s0 had a similar k = 1 oddity (0.30 at 50%). s1's k = 1 coverage is 0.65 at 50% and
0.94 at 90% on all windows, and 0.71 / 0.92 on occluded ones.

### Every horizon, TEST, occluded windows (s0 / s1)
| k | n hidden | std hidden | std visible | std ratio | 50% hidden | 90% hidden | error m hidden | error m visible |
|---:|---|---|---|---|---|---|---|---|
| 1 | 606 / 606 | 0.0546 / 0.0666 | 0.0288 / 0.0318 | 1.90 / 2.09 | 0.36 / 0.71 | 0.78 / 0.92 | 0.070 / 0.069 | 0.044 / 0.035 |
| 2 | 606 / 606 | 0.1035 / 0.1257 | 0.0503 / 0.0568 | 2.06 / 2.21 | 0.46 / 0.71 | 0.83 / 0.92 | 0.123 / 0.129 | 0.070 / 0.066 |
| 3 | 606 / 606 | 0.1429 / 0.1725 | 0.0732 / 0.0827 | 1.95 / 2.09 | 0.47 / 0.69 | 0.83 / 0.92 | 0.172 / 0.179 | 0.099 / 0.097 |
| 4 | 606 / 606 | 0.1731 / 0.2073 | 0.0963 / 0.1088 | 1.80 / 1.90 | 0.48 / 0.66 | 0.83 / 0.91 | 0.217 / 0.223 | 0.127 / 0.127 |
| 5 | 606 / 606 | 0.1969 / 0.2401 | 0.1189 / 0.1359 | 1.66 / 1.77 | 0.46 / 0.66 | 0.82 / 0.92 | 0.253 / 0.257 | 0.157 / 0.157 |
| 10 | 606 / 606 | 0.3016 / 0.3866 | 0.2198 / 0.2615 | 1.37 / 1.48 | 0.44 / 0.64 | 0.79 / 0.90 | 0.439 / 0.443 | 0.300 / 0.292 |
| 20 | 606 / 606 | 0.5139 / 0.6394 | 0.4212 / 0.5351 | 1.22 / 1.19 | 0.35 / 0.58 | 0.78 / 0.87 | 0.845 / 0.810 | 0.642 / 0.633 |

The rise is largest at k = 1–3 (about 2×) and fades with horizon, to 1.2 at k = 20, where the far
std is large anyway.

### The accuracy cost of occlusion: the same physical windows, TEST (s0 / s1)
**Goal-head position error on the 606 occluded windows:** the reference on frames without posts,
against the obstacle model on frames with them.

| k | n | reference, no posts | obstacle model | ratio |
|---:|---:|---|---|---|
| 1 | 606 | 0.023 / 0.025 | 0.070 / 0.069 | 3.07 / 2.72 |
| 2 | 606 | 0.043 / 0.047 | 0.123 / 0.129 | 2.85 / 2.72 |
| 3 | 606 | 0.064 / 0.069 | 0.172 / 0.179 | 2.70 / 2.61 |
| 4 | 606 | 0.082 / 0.091 | 0.217 / 0.223 | 2.63 / 2.45 |
| 5 | 606 | 0.101 / 0.113 | 0.253 / 0.257 | 2.50 / 2.28 |
| 10 | 606 | 0.240 / 0.236 | 0.439 / 0.443 | 1.83 / 1.88 |
| 20 | 606 | 0.550 / 0.546 | 0.845 / 0.810 | 1.54 / 1.48 |

**At k = 5: 0.253 / 0.257 m, against 0.101 / 0.113 m, about 2.4×.** It is about 3× at k = 1, and the
gap narrows with horizon.

**The same comparison on the other 6,794 windows,** where no post hides the ball (they include the
156 out-of-view windows, the same in both):

| k | n | reference | obstacle model | ratio |
|---:|---:|---|---|---|
| 1 | 6794 | 0.029 / 0.031 | 0.044 / 0.035 | 1.55 / 1.13 |
| 2 | 6794 | 0.054 / 0.057 | 0.070 / 0.066 | 1.31 / 1.16 |
| 3 | 6794 | 0.079 / 0.084 | 0.099 / 0.097 | 1.25 / 1.16 |
| 4 | 6794 | 0.104 / 0.109 | 0.127 / 0.127 | 1.23 / 1.17 |
| 5 | 6794 | 0.129 / 0.134 | 0.157 / 0.157 | 1.21 / 1.17 |
| 10 | 6794 | 0.252 / 0.257 | 0.300 / 0.292 | 1.19 / 1.14 |
| 20 | 6794 | 0.565 / 0.569 | 0.642 / 0.633 | 1.14 / 1.11 |

The obstacle model is worse there too, by 11–55%. So a world with posts costs accuracy everywhere,
not only while the ball is hidden.

**Std on occluded windows ÷ std on the others (the same windows):**

| k | reference, no posts | obstacle model |
|---:|---|---|
| 1 | 0.83 / 0.84 | 1.90 / 2.09 |
| 2 | 0.82 / 0.83 | 2.06 / 2.21 |
| 3 | 0.83 / 0.83 | 1.95 / 2.09 |
| 4 | 0.84 / 0.84 | 1.80 / 1.90 |
| 5 | 0.85 / 0.85 | 1.66 / 1.77 |
| 10 | 0.89 / 0.89 | 1.37 / 1.48 |
| 20 | 0.89 / 0.91 | 1.22 / 1.19 |

### The k = 20 headline ratio, and the goal head at every horizon, TEST
The obstacle model on obstacle test frames; the reference on its own (no posts). Persistence is the
same in both, because the physics is.

`DECIDING (goal_k20, lower is better): reference 0.3288 / 0.3312, arm 0.3842 / 0.3778 -> worse`

| k | reference | arm | ref mean | arm mean | change | verdict |
|---:|---|---|---:|---:|---:|---|
| 1 | 0.2111 / 0.2310 | 0.3476 / 0.2867 | 0.221 | 0.317 | +43.4% | worse |
| 2 | 0.2055 / 0.2184 | 0.2902 / 0.2771 | 0.212 | 0.284 | +33.8% | worse |
| 3 | 0.2087 / 0.2208 | 0.2805 / 0.2786 | 0.215 | 0.280 | +30.2% | worse |
| 4 | 0.2122 / 0.2234 | 0.2801 / 0.2803 | 0.218 | 0.280 | +28.6% | worse |
| 5 | 0.2184 / 0.2277 | 0.2836 / 0.2839 | 0.223 | 0.284 | +27.2% | worse |
| 10 | 0.2471 / 0.2506 | 0.3061 / 0.2992 | 0.249 | 0.303 | +21.6% | worse |
| 20 | 0.3288 / 0.3312 | 0.3842 / 0.3778 | 0.330 | 0.381 | +15.5% | worse |

**Worse at every horizon, in both seeds:** +43% at k = 1, +27% at k = 5, +15.5% at k = 20.

### Against the other baselines, TEST
| k | / const-vel | / sim-mean | error m arm s0 / s1 |
|---:|---|---|---|
| 1 | 1.82 (1.27) | 25.7 / 21.2 (16.3) | 0.046 / 0.038 |
| 2 | 0.94 (0.70) | 11.5 / 11.0 (8.4) | 0.075 / 0.071 |
| 3 | 0.69 (0.53) | 7.7 / 7.7 (5.9) | 0.105 / 0.104 |
| 4 | 0.57 (0.44) | 6.1 / 6.1 (4.7) | 0.135 / 0.135 |
| 5 | 0.50 (0.39) | 5.2 / 5.2 (4.1) | 0.165 / 0.165 |
| 10 | 0.36 (0.30) | 4.1 / 4.0 (3.3) | 0.311 / 0.304 |
| 20 | 0.31 (0.27) | 4.8 / 4.7 (4.1) | 0.658 / 0.647 |

### DEV (development only), error ÷ persistence (s0 / s1)
| run | 1 | 2 | 3 | 4 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|---|
| reference | 0.216 / 0.231 | 0.212 / 0.217 | 0.215 / 0.218 | 0.216 / 0.221 | 0.220 / 0.224 | 0.246 / 0.243 | 0.330 / 0.319 |
| arm | 0.367 / 0.294 | 0.308 / 0.293 | 0.298 / 0.294 | 0.299 / 0.293 | 0.303 / 0.294 | 0.321 / 0.300 | 0.387 / 0.367 |

Dev agrees: worse at every horizon, in both seeds.

### Calibration, TEST: joint fraction of truths inside the interval, all windows (s0 / s1)
| k | 50% ref | 50% arm | 90% ref | 90% arm |
|---:|---|---|---|---|
| 1 | 0.58 / 0.61 | 0.26 / 0.65 | 0.90 / 0.91 | 0.73 / 0.94 |
| 2 | 0.49 / 0.54 | 0.39 / 0.56 | 0.86 / 0.90 | 0.79 / 0.91 |
| 3 | 0.47 / 0.53 | 0.44 / 0.53 | 0.83 / 0.88 | 0.82 / 0.89 |
| 4 | 0.46 / 0.52 | 0.46 / 0.52 | 0.82 / 0.88 | 0.84 / 0.89 |
| 5 | 0.47 / 0.52 | 0.47 / 0.52 | 0.83 / 0.87 | 0.84 / 0.89 |
| 10 | 0.47 / 0.53 | 0.47 / 0.55 | 0.85 / 0.87 | 0.83 / 0.89 |
| 20 | 0.44 / 0.56 | 0.43 / 0.52 | 0.83 / 0.88 | 0.80 / 0.88 |

### Ball out of view, TEST: compared with the reference and with §3
| k | n ref | n arm | std ratio ref | std ratio arm | 90% hidden ref | 90% hidden arm |
|---:|---|---|---|---|---|---|
| 1 | 156 / 156 | 156 / 156 | 8.4 / 7.7 | 4.9 / 5.3 | 0.78 / 0.81 | 0.78 / 0.83 |
| 2 | 156 / 156 | 156 / 156 | 8.4 / 8.3 | 5.1 / 5.2 | 0.83 / 0.79 | 0.79 / 0.82 |
| 3 | 156 / 156 | 156 / 156 | 7.5 / 7.9 | 4.8 / 5.1 | 0.79 / 0.80 | 0.76 / 0.78 |
| 4 | 156 / 156 | 156 / 156 | 6.9 / 7.2 | 4.5 / 5.0 | 0.81 / 0.81 | 0.75 / 0.79 |
| 5 | 156 / 156 | 156 / 156 | 6.2 / 6.7 | 4.3 / 4.9 | 0.79 / 0.85 | 0.73 / 0.82 |
| 10 | 156 / 156 | 156 / 156 | 4.7 / 5.2 | 3.4 / 4.1 | 0.81 / 0.88 | 0.69 / 0.81 |
| 20 | 156 / 156 | 156 / 156 | 3.4 / 3.1 | 2.4 / 3.0 | 0.87 / 0.90 | 0.76 / 0.90 |

- **Occlusion is a smaller event than leaving the frame,** in the same model: the k = 1 std rises
  1.90 / 2.09× when occluded and 4.88 / 5.27× when out of view. The error is 0.070 / 0.069 m against
  0.221 / 0.247 m.
- **The out-of-view ratio is lower than the reference's (8.4 / 7.7) because the visible std rose,**
  not because the hidden std fell. Visible: 0.0285 / 0.0318, against 0.0187 / 0.0218. Hidden:
  0.139 / 0.168, against 0.157 / 0.168.
- **The same three rules on out-of-view windows:** rise holds (4.88 / 5.27), 90% coverage 0.78 /
  0.83, and recovery fails (2.01 / 2.61). The reference's were 0.78 / 0.81 and 2.02 / 1.85. So
  in the obstacle model, leaving the frame takes longer to recover from (2.01 / 2.61 at +3) than
  occlusion does (1.26 / 1.24).
- **§3:** on the same physical windows filmed with track_full, where the ball stays in view, the
  track_full model's k = 1 std ratio was still 3.96 / 4.21
  ([note](2026-11-20-m7-camera-full.md)). Those are hard moments in any view, as the matched
  occlusion windows are here.

### Probes, TEST: R², and Euclidean RMSE for pos_cam (s0 / s1)
| target | level | R^2 ref | R^2 arm | RMSE ref | RMSE arm | RMSE verdict |
|---|---|---|---|---|---|---|
| pos_cam | L1 | 0.681 / 0.649 | 0.630 / 0.662 | 0.3623 / 0.3940 | 0.4339 / 0.4145 | worse |
| pos_cam | L2 | 0.771 / 0.759 | 0.769 / 0.802 | 0.3087 / 0.3127 | 0.3589 / 0.3320 | worse |
| pos_cam | L3 | 0.925 / 0.907 | 0.894 / 0.887 | 0.1733 / 0.1921 | 0.2324 / 0.2356 | worse |
| pos_cam | pooled | 0.936 / 0.923 | 0.905 / 0.903 | 0.1580 / 0.1736 | 0.2231 / 0.2218 | worse |
| radius | pooled | 0.935 / 0.931 | 0.717 / 0.698 | 0.0150 / 0.0155 | 0.0314 / 0.0325 | worse |
| pos | pooled | 0.049 / 0.044 | 0.070 / 0.057 | 5.7390 / 5.7453 | 5.6359 / 5.6955 | better |
| friction | pooled | -0.029 / -0.032 | -0.026 / -0.022 | 0.2696 / 0.2700 | 0.2693 / 0.2687 | better |
| restitution | pooled | 0.047 / 0.055 | 0.079 / 0.061 | 0.1686 / 0.1679 | 0.1658 / 0.1674 | better |

- **`pos_cam` RMSE is worse at every level,** in both seeds (L3 RMSE 0.232 / 0.236, against 0.173 /
  0.192).
- **The radius readout drops sharply:** R² 0.717 / 0.698, against 0.935 / 0.931. Ball size is the
  radius cue, and a partly hidden ball shows less of its disc. That is one candidate cause, not
  tested here.
- Friction, restitution and world `pos` stay near zero R² in every run, as before.

### Training batches, steps 47,001–48,000 (s0 / s1)
These are the obstacle runs on obstacle frames, and the reference on frames without posts, so the
batches differ.

| key | reference | arm |
|---|---|---|
| relerr/k1 | 0.1376 / 0.1375 | 0.144 / 0.1502 |
| relerr/k2 | 0.2136 / 0.2114 | 0.2479 / 0.2562 |
| relerr/k3 | 0.2896 / 0.2837 | 0.3376 / 0.3479 |
| relerr/k4 | 0.3509 / 0.3455 | 0.4119 / 0.4231 |
| relerr/k5 | 0.402 / 0.3966 | 0.4733 / 0.4816 |
| relerr/k10 | 0.5666 / 0.549 | 0.6847 / 0.678 |
| relerr/k20 | 0.5532 / 0.5512 | 0.6257 / 0.6229 |
| emb_std/L1 | 1.873 / 1.853 | 1.846 / 1.814 |
| emb_std/L2 | 1.967 / 1.983 | 2.025 / 2.056 |
| emb_std/L3 | 1.897 / 1.895 | 1.891 / 1.909 |
| logvar/k10 | -0.1951 / -0.2894 | 0.4141 / 0.403 |
| logvar/k20 | -0.04424 / -0.04341 | 0.3478 / 0.3134 |
| loss/pix | 0.0009216 / 0.0009722 | 0.002328 / 0.002352 |
| w_pix_eff | 952.8 / 862.3 | 413.6 / 406.7 |
| pix_share | 0.1 / 0.1 | 0.1 / 0.1 |
| w_goal_eff | 0.001953 / 0.001953 | 0.001953 / 0.001953 |

- **Every latent horizon's `relerr` is higher** (k = 5: 0.473 / 0.482, against 0.402 / 0.397).
- **The far heads are less confident:** k = 20 `logvar` +0.35 / +0.31, against −0.04 / −0.04.
- **`loss/pix` is about 2.5× higher,** so `w_pix_eff` fell by about half to keep `pix_share` at 0.10. The posts
  add pixel content to reconstruct.

### `goalratio` on training batches, mean over each 4k window (s0 / s1)
| steps | k1 ref | k1 arm | k5 ref | k5 arm | k20 ref | k20 arm |
|---|---|---|---|---|---|---|
| 4,001-8,000 | 1.051 / 1.053 | 1.075 / 1.073 | 0.988 / 0.989 | 1.009 / 1.009 | 0.988 / 0.989 | 1.000 / 1.000 |
| 8,001-12,000 | 0.999 / 1.003 | 1.002 / 1.007 | 0.976 / 0.976 | 0.983 / 0.985 | 0.977 / 0.978 | 0.985 / 0.984 |
| 12,001-16,000 | 0.927 / 0.925 | 0.706 / 0.900 | 0.918 / 0.889 | 0.649 / 0.870 | 0.944 / 0.908 | 0.759 / 0.927 |
| 16,001-20,000 | 0.537 / 0.587 | 0.514 / 0.537 | 0.444 / 0.498 | 0.432 / 0.451 | 0.599 / 0.572 | 0.537 / 0.578 |
| 20,001-24,000 | 0.400 / 0.460 | 0.445 / 0.470 | 0.332 / 0.372 | 0.376 / 0.387 | 0.464 / 0.468 | 0.461 / 0.469 |
| 24,001-28,000 | 0.332 / 0.400 | 0.400 / 0.424 | 0.291 / 0.321 | 0.344 / 0.350 | 0.399 / 0.414 | 0.424 / 0.428 |
| 28,001-32,000 | 0.289 / 0.360 | 0.368 / 0.394 | 0.266 / 0.296 | 0.321 / 0.328 | 0.373 / 0.384 | 0.400 / 0.406 |
| 32,001-36,000 | 0.262 / 0.327 | 0.345 / 0.359 | 0.251 / 0.280 | 0.307 / 0.311 | 0.357 / 0.366 | 0.385 / 0.387 |
| 36,001-40,000 | 0.245 / 0.298 | 0.326 / 0.338 | 0.239 / 0.266 | 0.295 / 0.298 | 0.347 / 0.354 | 0.375 / 0.376 |
| 40,001-44,000 | 0.235 / 0.279 | 0.311 / 0.328 | 0.231 / 0.256 | 0.284 / 0.293 | 0.338 / 0.348 | 0.367 / 0.369 |
| 44,001-48,000 | 0.227 / 0.263 | 0.300 / 0.312 | 0.227 / 0.247 | 0.278 / 0.282 | 0.332 / 0.340 | 0.361 / 0.358 |

The obstacle runs leave persistence at about the same time (s0 sooner), then plateau higher: k = 1
0.300 / 0.312 over the last 4k, against 0.227 / 0.263.

**Training-sanity gates: PASS in both runs.**
- every logged value finite (no inf, no NaN) through step 48,000;
- `emb_std` lowest after step 2,000, ÷ its step-2,000 value: 0.90–1.03, against the 0.5 bar;
- far `logvar` last-500 mean +0.31 to +0.42, and lowest −1.19;
- `pix_share` within 1.6e-8 of 0.10;
- `w_goal_eff` 0.5/256 at every step after 4,000.

### Step time
Steady state, from checkpoint times over steps 8,001–48,000: **0.141 / 0.140 s/step**, the same as
§3's runs (0.141 / 0.141), which have the same model and diagnostics. The posts are rendered into
the dataset, so they cost nothing at training time.

## Conclusion
**Refuted, on recovery.** The obstacle-trained model's k = 1 std rises when a post hides the ball
(1.90 / 2.09×, rule 1 holds). Three frames after the ball reappears, it is still 1.26 / 1.24× its
level on other windows, above the 1.2 bar in both seeds. Calibration on occluded windows splits
(0.78 / 0.92). s0's failure there comes from its k = 1 intervals being too narrow everywhere.

**What the matched windows add (description, not a test):**
- The reference model, with no posts, is at 1.30 / 1.30× on the same moments at +3. So the
  obstacle model's std is back to what those moments cost anyway. Its excess is in frames 0 to +2
  after reappearance.
- Occlusion costs accuracy: on the same physical windows, the k = 5 error is 0.253 / 0.257 m
  against 0.101 / 0.113 m.
- A world with posts costs accuracy everywhere: k = 20 test error ÷ persistence 0.3842 / 0.3778,
  against 0.3288 / 0.3312, worse at every horizon. The radius probe drops from R² 0.93 to 0.70.
- Occlusion is a smaller event than leaving the frame, in the same model: a 2× std rise, not 5×,
  and a third of the error.

## Next
- **No change to the defaults.** `sim.obstacles` stays absent (off).
- **For M7's write-up:**
  - the model's uncertainty rises on occlusion, falls within three frames to the level the moment
    costs without occluders, and doesn't reach the pre-registered bar;
  - occlusion triples the near-horizon error on the hidden windows, and posts in the world cost
    11–55% on the others;
  - s0's k = 1 under-dispersion appeared in two of the four M7 ablations' s0 runs (§3 track_full
    and here), never in an s1, and not in the reference s0.
- **The user's call, not proposed as a test:** whether M7's write-up should read recovery against
  the matched no-occluder baseline rather than the model's own other windows. That reading is
  post-hoc, so this note's verdict stays refuted either way.
