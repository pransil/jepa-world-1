# M7 plan: ablations

- **Status:** plan only, with no runs yet. Written 2026-11-17, before M2, M4, M5 and M6 merged.
- **SPEC ref:** §8 M7: "Skip connections on and off. Top-down prediction (see section 9). Camera
  modes and obstacles. Pixel weight at 0, low, and high."
- **Decisions made for this plan (user, 2026-11-17):**
  - [top-down targets](../decisions/2026-11-17-top-down-targets.md) (SPEC §9.2);
  - [obstacle keys: absent means off](../decisions/2026-11-17-obstacles-dataset-keying.md);
  - [skip connections = decoder skips](../decisions/2026-11-17-skip-connections-meaning.md).

Each section below is written so it can become a single `/experiment` run
(`.claude/skills/experiment/SKILL.md`). Its block maps onto Step 1 (hypothesis, deciding metric,
refuted-if, baselines, budget) and Step 2 (the config). When a section runs, copy its block into
the experiment note, then fill in the shas and run dirs.

## What blocks M7

| Needs | From | Why | State on 2026-11-17 |
|---|---|---|---|
| Episode datasets, loader, persistence and const-velocity baselines | M2 (lane m2) | Every run trains on episodes, and every result reports the baselines | branch `worktree-wt-m2-data`, not merged |
| Far heads (k = 10, 20) with Gaussian NLL | M4 | k = 20 is the headline horizon, and top-down extends these heads | branch `worktree-wt-m4-far-heads`, not merged |
| Goal head (position mean and variance per k), pixel decoder, decoder path | M5 (lane m5) | Position error and calibration come from the goal head. The pixel and skip ablations need the decoder | in progress |
| `metrics.py`, `probes.py`, `eval --run <dir> --split test`, the simulator-aware baseline | M6 (lane m6) | Every deciding metric below is an M6 metric | in progress. The simulator-aware baseline needs a `sim/` from-state entry point |

## Shared protocol

- **Reference run:** the base config after M5 (`configs/base.yaml`, `w_pix` 0.05, skips on,
  top-down off, `track_lagged`, no obstacles), with **seeds 0 and 1**. Every ablation is a diff
  against it. Reuse the reference only if
  `git diff <ref sha> <ablation sha> -- jepa_world/models jepa_world/train.py jepa_world/data`
  is empty. Otherwise re-run it at the ablation's sha.
- **Seeds:** 2 per setting, which is the minimum `/experiment` allows for claiming a difference.
  Add a 3rd seed when the difference between settings is under 2× the spread between seeds.
- **Split:** the headline numbers are on the **test** terrain seeds only, and dev numbers are
  labelled dev (SPEC §7.3).
- **Baselines (SPEC §7.1), next to every result:**
  - persistence;
  - const_velocity (the true v(t));
  - const_velocity_fd (the context positions only);
  - from M6, the simulator-aware baseline on known terrain.

  The baselines read only ground-truth state. Camera mode and occluders don't change physics: the
  camera is a pure function of the ball state, and the occluders don't collide. So one baseline
  table per test split serves every ablation. **Check this before sharing the table:** the GT
  `pos` and `vel` arrays of each rebuilt test split must equal the base split's exactly.
- **Configs:** `configs/m7_<slug>.yaml` with `base: base.yaml`, holding only the keys under test.
  The tree is clean when the run starts (`/experiment` Step 2).
- **Training sanity before any conclusion** (`/experiment` Step 4):
  - did it converge, judged from the whole of `metrics.jsonl`?
  - embedding std per level, to rule out collapse;
  - every loss term sane;
  - `w_pix·L_pix` as a share of `loss/total`.
- **GPU:** every run takes `locks.gpu_lock` through `train.main`. Queue one job per lane at a time.

## Compute basis

Measured (MPS, fp32, B = 32, synthetic frames, so timing depends on the shapes and not on content):

| What | Step time | Source |
|---|---|---|
| Near heads only (M3) | 75–93 ms | [l1-head-width](../decisions/2026-11-17-l1-head-width.md); runs `20261118T042046Z-…-a92e58`, `20261118T021043Z-…-a5add4` |
| Near heads only (M4 code) | 104 ms | `20261118T050651Z-worktree-wt-m4-far-heads-bench-l1-heads-9542d0` (sha 67ab30d, clean) |
| Near + far, far heads 32 wide | 144 ms | `20261118T050707Z-worktree-wt-m4-far-heads-bench-l1-heads-2e1860` |
| **Near + far, far heads at the level's width (M4 default)** | **158 ms**, 3.3M trainable params, 3257 MB driver memory | `20261118T050637Z-worktree-wt-m4-far-heads-bench-l1-heads-337448` |
| One L1 head, fwd+bwd | ≈ 8–10 ms (40–49 ms for 5) | l1-head-width |

The step time varied by up to 25% between identical runs (l1-head-width), so read every figure as ±25%.

**Per-run estimate (an extrapolation, not measured):**
- 158 ms × 10k steps (the `train.steps` placeholder) ≈ 26 min.
- M5's goal head and pixel decoder add an estimated +20–40%, giving **≈ 32–37 min per run**.
- That leaves out the real data loader and M6 eval time, neither yet measured.
- Re-check against M5's bench, and scale linearly if real-data runs need more than 10k steps.

**Dataset builds (CPU, no GPU lock):**
- One full set of splits is dev 10×4 + train 500×4 + test 50×4 = 2240 episodes (base.yaml on the M2 branch).
- Rendering takes 49–97 ms per episode, a rough figure taken unlocked on a shared machine
  (docs/milestones/M1.md, run `20261118T023301Z-…-6c2ce8`). So a set takes **≈ 2–4 min** and ≈ 74 MB
  (33 KB per episode).
- Only §3 (track_full) and §5 (obstacles) need new sets.

## 1. Pixel weight: 0, low, high

- **Question:** does the pixel term help or hurt the latent world model, and from what weight?
- **Switch:** `loss.w_pix`. The key **exists** in `configs/base.yaml` (0.05). It is consumed by
  `models/pixel.py` and the training loss (M5, lane m5).
- **Settings:**
  - **0**;
  - **0.05** (low, the SPEC §5.4 default, i.e. the reference);
  - **0.5** (high, 10× the base).

  Revisit "high" once M5's logs show how big `L_pix` is next to the latent terms.
- **Hypothesis:** raising `w_pix` from 0.05 to 0.5 worsens the k = 20 goal-head position error.
  The pixel term pulls encoder capacity toward reconstruction, which is SPEC §4.7's "blur"
  concern. Between 0 and 0.05 the difference stays within the seed spread.
- **Deciding metric:** the goal-head position error at **k = 20**, as a ratio to persistence, on
  the **test** split. Compare 0.5 with 0.05.
- **Refuted if:** at 0.5 the ratio is no worse than at 0.05, beyond the seed spread.
- **Secondary:**
  - 0 vs 0.05 on the same metric;
  - linear-probe ball-position error at L1 and L3 (SPEC §7.2);
  - embedding std per level;
  - `w_pix·L_pix / loss/total`. SPEC §8 M5 tests that the pixel term "does not dominate" at 0.05, and
    at 0.5 it may.
- **Baselines:** the shared table.
- **Runs:** 0 and 0.5, × seeds 0 and 1 = **4 new runs**, plus the reference. ≈ 2.1–2.5 GPU-h.
- **Depends on:** M2, M4, M5, M6. No new code.
- **Config:** `configs/m7_pixel_w0.yaml` and `configs/m7_pixel_w05.yaml`, each setting `loss.w_pix`.

## 2. Skip connections on/off (U-Net decoder skips)

- **Question:** when the pixel decoder can't take a shortcut through L1/L2, does L3 come to hold
  more of the scene, and is that good for prediction?
- **Switch:** `model.unet_skips` (the proposed name, default `true`). It **does not exist yet**. It
  belongs to the decoder path in `models/unet.py` and `models/pixel.py`. m5 has been asked to
  expose it; otherwise a later M7 lane adds it through `/builder`. With `false`, the decoder
  upsamples from L3 alone ([decision](../decisions/2026-11-17-skip-connections-meaning.md)).
- **Settings:** skips on or off, at `w_pix` 0.05 and 0.5. At `w_pix = 0` the switch is a no-op,
  because it only acts through the pixel loss.
- **Hypothesis:** with the skips off, the pixel loss has to be satisfied through L3, so L3 holds
  more ball-position information.
- **Deciding metric:** the **L3 linear-probe ball-position error** (M6 `probes.py`) on the **test**
  split, skips off vs on, at `w_pix = 0.5`.
- **Refuted if:** the difference at 0.5 is within the seed spread, or the off setting is worse.
- **Secondary:**
  - the same comparison at 0.05;
  - the k = 20 goal-head error ÷ persistence, to see whether the richer L3 helps or costs far
    prediction;
  - `loss/pix`, which should be worse with the skips off. That is a sanity check that the switch
    did something.
- **Runs:** off × {0.05, 0.5} × seeds 0, 1 = **4 new runs**. Skips on reuses the reference (0.05)
  and §1's 0.5 runs, if the shas allow it under the reuse rule. ≈ 2.1–2.5 GPU-h.
- **Depends on:** M5 (the decoder), the switch, M6. Run it after §1, so the 0.5 runs can be shared.
- **Config:** `configs/m7_skips_off_w005.yaml` and `configs/m7_skips_off_w05.yaml`.

## 3. Camera mode: track_lagged vs track_full

- **Question:** does the model's uncertainty track *visibility*, or just trajectory difficulty?
  Out-of-view events under `track_lagged` come only from deep drops (M1.md), and deep drops are
  also dynamically hard. `track_full` keeps the ball in frame on exactly the same trajectories,
  which separates the two.
- **Switch:** `sim.camera_mode`. It **exists** in `jepa_world/sim/camera.py`, merged in M1
  ([decision](../decisions/2026-11-17-camera-mode-track-lagged.md)). No new code. Setting it to
  `track_full` changes the `sim` hash, so `build_dataset_once` builds a separate track_full dataset
  (≈ 2–4 min of CPU). Every existing dataset stays as it is.
- **Evaluation windows:** the "OOV windows" are test windows whose last context frame t is
  `out_of_view` under `track_lagged`, flagged from the lagged dataset's GT. The same (episode, t)
  pairs are scored in both models, each on its own frames.
  - M1 measured 2.4% of frames out of view, in 17% of episodes, on seeds 0–199. That suggests
    roughly 290 out-of-view frames in the 200 test episodes, so the CIs will be wide. Bootstrap
    over episodes.
- **Hypothesis:** the lagged model's predicted goal-head std rises when the ball is out of view,
  and the full model's doesn't on the same windows. The rise comes from visibility, not difficulty.
- **Deciding metric** (SPEC §7.2, event behavior): the goal-head predicted std at **k = 1** on OOV
  windows ÷ the same on in-view windows. Compute it for each model, and compare lagged with full
  on the **test** split. If M6 defines event behavior differently, use M6's definition and say so
  in the note.
- **Refuted if:** the lagged ratio is < 1.2, or it doesn't exceed the full model's ratio by more
  than the seed spread.
- **Secondary:**
  - coverage of the 50% and 90% intervals on OOV windows (lagged), which must be within ±0.1 of
    nominal to count as calibrated;
  - std against frames since exit and since re-entry, the "rise then fall" of §7.2;
  - k = 5 position error on OOV windows, lagged ÷ full, which is the accuracy cost of the lagged
    camera;
  - the k = 20 headline ratio for both.
- **Runs:** track_full × seeds 0, 1 = **2 new runs**, plus a dataset build. Lagged = the reference.
  ≈ 1.1–1.2 GPU-h.
- **Depends on:** M2 (the build pipeline), M5 (the goal head's variance), M6 (calibration, event
  metric).
- **Config:** `configs/m7_camera_full.yaml` (`sim.camera_mode: track_full`).

## 4. Top-down prediction

- **Question (SPEC §9.2):** if the far heads must also predict the next-finer future target, do
  the coarse levels become better world models at long horizons?
- **Switch:** `model.top_down` (default `false`) and `loss.w_td` (0.5), the proposed names. They
  **do not exist yet**. The code goes in `models/heads.py`, `models/jepa.py` and `train.py`
  (logging), in a later M7 code lane through `/builder`, after M4 and M5 merge. The targets are
  set by the [decision](../decisions/2026-11-17-top-down-targets.md):
  - k = 20 on L3 also predicts the target **L2(t+20)**, through up×4;
  - k = 10 on L2 also predicts the target **L1(t+10)**, through up×2;
  - both use Gaussian NLL at `w_td = 0.5`, with the gradient flowing into context L3/L2.
- **Hypothesis:** the top-down terms make context L3 keep finer information about the ball, which
  lowers the k = 20 goal-head position error.
- **Deciding metric:** the goal-head position error at **k = 20**, ÷ persistence, on the **test**
  split, top-down on vs the reference.
- **Refuted if:** the on setting isn't lower by more than the seed spread.
- **Guards (a win only counts if these hold):**
  - the near-head losses (k = 1..5) and the far heads' own `sqerr/10`, `sqerr/20` are no worse
    than the reference, beyond the seed spread;
  - the variance on every level stays above the floor.
- **Secondary:** the L3 and L2 linear-probe ball-position error, and probes for the hidden
  parameters at L3.
- **Runs:** top-down on × seeds 0, 1 = **2 new runs**. ≈ 1.1–1.4 GPU-h, with the step about 10%
  slower (the decision estimates +10–20 ms).
- **Depends on:** M4, M5, M6, plus the code lane above.
- **Config:** `configs/m7_top_down.yaml` (`model.top_down: true`).

## 5. Obstacles and occlusion

- **Question:** when the ball is hidden behind a foreground object, while the camera still points
  at it, does the model's uncertainty rise and then fall again, and is it calibrated?
- **Switch:** `sim.obstacles`, a block of keys **that does not exist yet**. The code goes in
  `sim/scene.py` (occluder geoms), `sim/render.py`, and `sim/episode.py` (the `occluded` flag),
  in a later sim lane through `/builder`.
- **How the keys arrive (decided: absent means off):**
  - the keys are never added to base.yaml's parsed config, only as a commented block;
  - `sim/` treats a missing `obstacles` as off;
  - the experiment yaml carries the block, because `--set` can't create keys.

  So every existing dataset hash stays valid, and there is no `generator_version` bump
  ([decision](../decisions/2026-11-17-obstacles-dataset-keying.md)).
- **Design sketch (not binding; the sim lane decides and records its choices):**
  - the occluders are boxes at y between the camera (y = −10) and the ball's plane (y = 0), with
    `contype = conaffinity = 0`, so physics and GT are unchanged;
  - their x positions and widths come from a seed stream derived from the terrain seed, which
    keeps the split by terrain seed intact;
  - `occluded[t]` is defined geometrically (a pure function, tested in the fast tier), and it is
    cross-checked against segmentation in the render tier, following M1's pattern;
  - tune for roughly 5–10% occluded frames, using `sim.preview` stats as M1 did for the
    out-of-view rates.
- **Hypothesis:** the obstacle-trained model's goal-head std rises on occluded windows and falls
  back after reappearance, and its intervals stay calibrated there.
- **Deciding metric** (SPEC §7.2, event behavior): the goal-head predicted std at **k = 1** on
  windows whose last context frame is `occluded` ÷ visible windows, on the **test** split. Use
  M6's event metric if it differs.
- **Refuted if:**
  - the ratio is < 1.2;
  - or the 90% coverage on occluded windows is < 0.8;
  - or the std hasn't come back to within 1.2× the visible level by 3 frames after reappearance.
- **Secondary:**
  - k = 5 position error on occluded windows, compared with the reference model on the same
    physical windows (its frames have no occluder), which is the accuracy cost of occlusion;
  - the k = 20 headline ratio;
  - for M7's write-up, a comparison with §3's out-of-view behavior.
- **Runs:** obstacles × seeds 0, 1 = **2 new runs**, plus a dataset build. ≈ 1.1–1.2 GPU-h.
- **Depends on:** M2, M5, M6, the sim lane, and **the timing below**.
- **Config:** `configs/m7_obstacles.yaml` (`sim.obstacles: {…}`, with the camera left at
  `track_lagged`, which is SPEC's `track_lagged_obstacles`).

### Flag: the obstacle work changes `sim/` and touches dataset hashes

- M2's dataset key hashes the **whole** `sim` config (`data/build.py:split_key`, on the M2 branch).
  A new `sim.*` key in base.yaml, even one that is off, would re-key and orphan every dataset. The
  decision above avoids that: the key is absent, which means off.
- **The sim lane must not start** until both of these hold:
  - M2's datasets are built and merged;
  - the `sim/` from-state entry point (for M6's simulator-aware baseline) has merged.

  Otherwise two lanes are in `sim/episode.py` at once.
- If the obstacle change alters simulator output when the key is absent, it is a bug. The
  alternative would be a `generator_version` bump that re-keys everything, and that decision goes
  back to the user.

## Summary

| § | Ablation | Switch | Exists now? | Code owner | New runs | Est. GPU-h |
|---|---|---|---|---|---|---|
| — | Reference | base.yaml | yes, after M5 | — | 2 | 1.1–1.2 |
| 1 | Pixel weight 0 / 0.05 / 0.5 | `loss.w_pix` | key yes; consumer M5 | m5 | 4 | 2.1–2.5 |
| 2 | Decoder skips on/off | `model.unet_skips` | no | m5, or a later M7 lane | 4 | 2.1–2.5 |
| 3 | track_lagged vs track_full | `sim.camera_mode` | **yes** (M1) | — (dataset build) | 2 | 1.1–1.2 |
| 4 | Top-down (A) | `model.top_down`, `loss.w_td` | no | a later M7 code lane | 2 | 1.1–1.4 |
| 5 | Obstacles | `sim.obstacles` | no | a later sim lane | 2 | 1.1–1.2 |
| | **Total** | | | | **16** | **≈ 8.5–10** (training only) |

**Order:** §1, then §2 (it shares §1's runs), then §3, then §4, then §5. §1 and §3 need no new code
once M5 and M6 land. Run them back to back, keeping one queued job per lane.
