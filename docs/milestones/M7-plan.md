# M7 plan: ablations

- **Status:** plan. Written 2026-11-17, before M2, M4, M5 and M6 merged. §1 has run, at 24k steps (PR #28).
- **SPEC ref:** §8 M7: "Skip connections on and off. Top-down prediction (see section 9). Camera
  modes and obstacles. Pixel weight at 0, low, and high."
- **Decisions made for this plan (user, 2026-11-17):**
  - [top-down targets](../decisions/2026-11-17-top-down-targets.md) (SPEC §9.2);
  - [obstacle keys: absent means off](../decisions/2026-11-17-obstacles-dataset-keying.md);
  - [skip connections = decoder skips](../decisions/2026-11-17-skip-connections-meaning.md).
- **Ablation length (user, 2026-11-19):** every remaining arm trains for **48,000 steps**, seeds 0
  and 1, against the 48k reference ([m7-ablations-at-48k](../decisions/2026-11-19-m7-ablations-at-48k.md)).
- **Defaults the reference and arms inherit (user, 2026-11-19):**
  - a 10% pixel share ([pixel-share-default](../decisions/2026-11-19-pixel-share-default.md),
    PR #30);
  - per-channel latent losses ([latent-loss-normalization](../decisions/2026-11-19-latent-loss-normalization.md),
    PR #31).

  So §1 and §2 are stated relative to the 10% share, and runs compare on scale-free metrics
  ("Shared protocol").

Each section below is written so it can become a single `/experiment` run
(`.claude/skills/experiment/SKILL.md`). Its block maps onto Step 1 (hypothesis, deciding metric,
refuted-if, baselines, budget) and Step 2 (the config). When a section runs, copy its block into
the experiment note, then fill in the shas and run dirs.

## What blocks M7

| Needs | From | Why | State on 2026-11-17 |
|---|---|---|---|
| Episode datasets, loader, persistence and const-velocity baselines | M2 | Every run trains on episodes, and every result reports the baselines | **merged** (PR #7). The base datasets are built ([M2.md](M2.md)) |
| Far heads (k = 10, 20) with Gaussian NLL | M4 | k = 20 is the headline horizon, and top-down extends these heads | branch `worktree-wt-m4-far-heads`, not merged |
| Goal head (position mean and variance per k), pixel decoder, decoder path | M5 (lane m5) | Position error and calibration come from the goal head. The pixel and skip ablations need the decoder | in progress |
| `metrics.py`, `probes.py`, `eval --run <dir> --split test`, the simulator-aware baseline | M6 (lane m6) | Every deciding metric below is an M6 metric | in progress. The simulator-aware baseline needs a `sim/` from-state entry point |

## Shared protocol

- **Reference run:** `configs/exp_headline_48k.yaml` **at the 2026-11-19 defaults**, with **seeds
  0 and 1**. That is base.yaml as of PRs #30 and #31, trained for 48,000 steps:
  - **pixel term at a 10% share** (`loss.pix_share: 0.10`,
    [pixel-share-default](../decisions/2026-11-19-pixel-share-default.md));
  - **per-channel latent losses** (`loss.latent_reduce: mean`,
    [latent-loss-normalization](../decisions/2026-11-19-latent-loss-normalization.md)). Every
    per-level latent term, the guard included, is divided by its level's channel count. `w_goal`
    stays 0.5 and is divided by C(L3); a fixed `w_pix` would be divided by C(L1);
  - skips on, top-down off, `track_lagged`, no obstacles, guard sum ×25.

  Every ablation is a diff against it. Reuse the reference only if
  `git diff <ref sha> <ablation sha> -- jepa_world/models jepa_world/train.py jepa_world/data configs/base.yaml`
  is empty. Otherwise re-run it at the ablation's sha.
  - **The re-run at these defaults is assigned to m5** (2026-11-19): 2 runs, ≈ 4 GPU-h. It is the
    first full-model run at the new defaults, because the latent-normalization collapse check was
    near + far only.
  - **The existing 48k runs are not the reference** (commit 5ed1564,
    [headline-48k](../../experiments/2026-11-18-headline-48k.md)). They trained at a fixed `w_pix`
    0.05 with channel-summed latents. They reproduce at their commit, or with
    `--set loss.pix_share=null --set loss.latent_reduce=sum`.
- **Compare runs only on scale-free metrics.** Under `latent_reduce: mean`, `loss/total` is about
  64× smaller than under `sum` before far joins, and about 112–161× smaller after
  ([latent-loss-normalization](../decisions/2026-11-19-latent-loss-normalization.md), item 7).
  - **Use:** `relerr/k`, `goalratio/k`, `goalerr/k`, `emb_std/L`, `logvar/k`, and M6's eval
    metrics. These are unchanged by the reduction.
  - **Never `loss/total`**, and not `loss/k`, `sqerr/k`, `var/L` or `loss/var` across reductions.
    That covers any comparison with the old headline, 48k or pixel-share runs, which are all `sum`.
    To put a `mean` run's per-level term on the `sum` scale, multiply by that level's channel count.
  - Within the M7 arms, all at `mean`, the per-level terms are comparable. The scale-free ones
    are still preferred.
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

  The base test table already exists: run `20261118T051751Z-worktree-wt-m2-data-baselines-test-73e338`,
  with 200 episodes and 7400 windows, t ∈ [3, 39] ([M2.md](M2.md)).
  - Persistence at k = 1 / 5 / 20 is 0.133 / 0.581 / 1.713 m.
  - Constant velocity (true v) is 0.174 / 0.568 / **1.211** × persistence. It loses to persistence
    at k = 20.

  So "beats persistence at k = 20" is a real bar. M6 must score models on these same windows (M2.md).
- **Configs:** `configs/m7_<slug>.yaml` with `base: exp_headline_48k.yaml`, holding only the keys
  under test, so each arm trains 48,000 steps with all four stages. Each arm inherits the 10% share
  and `latent_reduce: mean`. Don't pin either unless it is the key under test. The tree is clean
  when the run starts (`/experiment` Step 2).
- **Training sanity before any conclusion** (`/experiment` Step 4):
  - did it converge, judged from the whole of `metrics.jsonl`?
  - embedding std per level, to rule out collapse;
  - every loss term finite, with its trend sane. Judge the size of each term only against the
    reference, at the same reduction;
  - the logged `pix_share`, which should sit at the target (0.10 unless the arm sets another)
    once the pixel stage starts, and `w_pix_eff` / `w_goal_eff`.
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

**Per-run cost, measured on the full model (real data, all four stages, MPS, B = 32):**

| Runs | Steps | Training wall-clock | Step time |
|---|---|---|---|
| headline 48k, seeds 0 / 1 (commit 5ed1564) | 48,000 | 1.95 / 1.90 h | 0.146 / 0.143 s |
| pixel share 1% and 10%, seeds 0 / 1 (commit f1a7c03) | 24,000 | 0.87–0.88 h | 0.130–0.132 s |

- Training time is from the run-dir timestamp to the last write of `metrics.jsonl`.
- The 48k runs were slower per step, probably because a CPU readout ran alongside part of each
  (the guard comparison's).
- `eval --split test --split dev` took about 2 min per run (1.4–1.8 min after training ended, in
  the chained queue).
- **Planning figure: ≈ 2 GPU-h per 48k arm per seed**, eval included. It doesn't count dataset
  builds, which only §3 and §5 need, nor readouts, which run on CPU.
- **Timing basis:** these runs used `latent_reduce: sum`. `mean` only divides each term by a
  constant, so the step time shouldn't change. The pixel-share runs already ran the 10% share's
  weighting, which is now the default. Re-check against m5's reference re-run.

The 2026-11-17 extrapolation it replaces was ≈ 32–37 min per 10k-step run. The full model measures
0.13–0.15 s/step, so that is about 22–25 min per 10k steps; the estimate was high.

**Dataset builds:**
- One full set of splits is dev 10×4 + train 500×4 + test 50×4 = 2240 episodes (`data.splits` in
  `configs/base.yaml`).
- M2 measured the whole base set at **4 min 39 s wall**, 69.3 MB (train 214 s, or 107 ms per
  episode). That was at a load average of about 150, so it's an upper bound ([M2.md](M2.md)).
- The build CLI (`python -m jepa_world.data.build --split all`) takes the GPU lock, so it queues
  like a run.
- Only §3 (track_full) and §5 (obstacles) need new sets.

## 1. Pixel weight: 0, low, high (as shares of the loss)

**Done at 24k, under the old defaults** ([2026-11-18-pixel-share-ablation.md](../../experiments/2026-11-18-pixel-share-ablation.md),
PR #28). A fixed `w_pix` left the pixel term at ≤ 3e-5 of the loss, so the levels became **target
shares** via `loss.pix_share` ([pixel-ablation-target-share](../decisions/2026-11-18-pixel-ablation-target-share.md)):
- **The levels:** ≈ 0, which is the existing headline runs (fixed `w_pix` 0.05); 1%
  (`configs/m7_pixel_share01.yaml`); and 10% (`configs/m7_pixel_share10.yaml`).
- **How they ran:** 24k steps, seeds 0 and 1, all at `latent_reduce: sum`. Those configs pin `sum`
  and stay at 24k, because they record what ran.
- **Result:** at 10% the k = 20 ratio was 0.416, against 0.430 at ≈ 0, so no worse. k = 1–5
  improved by 10–15%, and 1% sat within the seed spread of ≈ 0. **10% became the base.yaml default**
  ([pixel-share-default](../decisions/2026-11-19-pixel-share-default.md)).

These are shares of the loss **value**; the gradient share in the encoder is not measured.

**At 48k, relative to the 10% default: not scheduled.** A re-run is the user's call. If it runs:
- **Question:** does moving the pixel share away from the 10% default, down to 0 or up, help or hurt
  the latent world model?
- **Switch:** `loss.pix_share`, default 0.10.
  - `pix_share: 0` is refused (`check_pix_share`). The "0" arm is `pix_share: null` with
    `w_pix: 0`.
  - `pix_share: null` with the fixed `w_pix` 0.05 is ≈ 0, not 0. Under `mean` that weight is divided
    by C(L1) = 64, so the term stays tiny.
- **Settings:**
  - the reference: **10%**, skips on;
  - **0**;
  - **high: 30%**, which is the pixel-share note's proposal 1. It was never tested.
  - 1% is optional. At 24k it was within the seed spread of ≈ 0.
- **Hypothesis:** raising the share from 10% to 30% worsens the k = 20 goal-head position error.
  The pixel term pulls encoder capacity toward reconstruction, which is SPEC §4.7's "blur"
  concern. Dropping it to 0 loses the k = 1–5 gain that 10% showed at 24k.
- **Deciding metric:** the goal-head position error at **k = 20**, as a ratio to persistence, on
  the **test** split. Compare 30% with 10%.
- **Refuted if:** at 30% the ratio is no worse than at 10%, beyond the seed spread.
- **Secondary:**
  - 0 vs 10% at k = 1–5;
  - linear-probe ball-position error at L1 and L3 (SPEC §7.2);
  - embedding std per level;
  - the logged `pix_share` and `w_pix_eff`, as a check that each arm held its target.
- **Baselines:** the shared table.
- **Runs:** 0 and 30%, × seeds 0 and 1 = **4 new runs**, ≈ 8 GPU-h. Adding 1% is 2 more runs, ≈ 4
  GPU-h. 10% is the reference.
- **Depends on:** the reference. No new code.
- **Config:** `configs/m7_pixel_off_48k.yaml` (`pix_share: null`, `w_pix: 0`) and
  `configs/m7_pixel_share30_48k.yaml` (`pix_share: 0.30`), each with `base: exp_headline_48k.yaml`.

## 2. Skip connections on/off (U-Net decoder skips)

- **Question:** when the pixel decoder can't take a shortcut through L1/L2, does L3 come to hold
  more of the scene, and is that good for prediction?
- **Switch:** `model.unet_skips` (default `true`). It **exists**: m5 added it to the decoder path in
  `models/pixel.py` with M5. With `false`, the decoder upsamples from L3 alone
  ([decision](../decisions/2026-11-17-skip-connections-meaning.md)).
- **Settings:** skips on or off, at the **10% default share**. Skips on is the reference.
  - The switch acts only through the pixel loss, so it is a no-op with the pixel term off.
  - **What stays fixed:** the share holds the pixel term at 10% of the rest of the loss in both arms.
    With the skips off, `loss/pix` should rise, and `w_pix_eff` falls to keep the term at 10%. So
    the arms differ in how hard the decoder finds L3-only reconstruction, not in the pixel term's
    share of the loss.
- **Hypothesis:** with the skips off, the pixel loss has to be satisfied through L3, so L3 holds
  more ball-position information.
- **Deciding metric:** the **L3 linear-probe ball-position error** (M6 `probes.py`) on the **test**
  split, skips off vs on, at 10%.
- **Refuted if:** the difference is within the seed spread, or the off setting is worse.
- **Secondary:**
  - the k = 20 goal-head error ÷ persistence, to see whether the richer L3 helps or costs far
    prediction;
  - `loss/pix`, which should be worse with the skips off. That is a sanity check that the switch
    did something. Compare `loss/pix` itself, not the pixel term, which the share holds fixed.
- **Runs:** skips off × seeds 0, 1 = **2 new runs** at 48k steps, ≈ 4 GPU-h. Skips on is the
  reference.
  - The plan's earlier second level (`w_pix` 0.5) becomes 30%. Off at 30% is 2 more runs, ≈ 4
    GPU-h. The on arm is §1's 30% arm, if §1 runs it at the same sha.
- **Depends on:** M5 (the decoder), the switch, M6, the reference.
- **Config:** `configs/m7_skips_off.yaml` (`model.unet_skips: false`, `base: exp_headline_48k.yaml`).
  Add `configs/m7_skips_off_share30.yaml` for the 30% level.

## 3. Camera mode: track_lagged vs track_full

- **Question:** does the model's uncertainty track *visibility*, or just trajectory difficulty?
  Out-of-view events under `track_lagged` come only from deep drops (M1.md), and deep drops are
  also dynamically hard. `track_full` keeps the ball in frame on exactly the same trajectories,
  which separates the two.
- **Switch:** `sim.camera_mode`. It **exists** in `jepa_world/sim/camera.py`, merged in M1
  ([decision](../decisions/2026-11-17-camera-mode-track-lagged.md)). No new code. Setting it to
  `track_full` changes the `sim` hash, so `build_dataset_once` builds a separate track_full dataset
  (under 5 min, per M2's build). Every existing dataset stays as it is.
- **Evaluation windows:** the "OOV windows" are test windows whose last context frame t is
  `out_of_view` under `track_lagged`, flagged from the lagged dataset's GT. The same (episode, t)
  pairs are scored in both models, each on its own frames.
  - M2 counted 161–202 of 7400 test windows out of view at t+k (2.2–2.7%) ([M2.md](M2.md)). The
    count for t is similar, so the CIs will be wide. Bootstrap over episodes.
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
- **Runs:** track_full × seeds 0, 1 = **2 new runs** at 48k steps, plus a dataset build. Lagged =
  the reference. ≈ 4 GPU-h.
- **Depends on:** M2 (the build pipeline), M5 (the goal head's variance), M6 (calibration, event
  metric).
- **Config:** `configs/m7_camera_full.yaml` (`sim.camera_mode: track_full`).

## 4. Top-down prediction

- **Question (SPEC §9.2):** if the far heads must also predict the next-finer future target, do
  the coarse levels become better world models at long horizons?
- **Switch:** `model.top_down` (default `false`) and `loss.w_td` (0.5). They **exist**, from the
  M7 top-down lane (`models/heads.py` `TopDownHead`, `models/jepa.py`; `models/DESIGN.md`, "Top-down
  prediction"). The terms are logged as `loss/td/k10` and `loss/td/k20`, plus `sqerr/td/k<k>`,
  `logvar/td/k<k>` and `relerr/td/k<k>`. The targets are set by the
  [decision](../decisions/2026-11-17-top-down-targets.md):
  - k = 20 on L3 also predicts the target **L2(t+20)**, through up×4;
  - k = 10 on L2 also predicts the target **L1(t+10)**, through up×2;
  - both use Gaussian NLL at `w_td = 0.5`, with the gradient flowing into context L3/L2;
  - under `latent_reduce: mean`, each branch's NLL is divided by its target level's channel count,
    like that level's own terms. Where it enters, though, it gains 2× against the far NLL of the
    level it reads ([latent-loss-normalization](../decisions/2026-11-19-latent-loss-normalization.md),
    item 6). Both arms inherit `mean`, so they compare at one reduction. Don't compare this
    ablation with a `sum` run.
- **Hypothesis:** the top-down terms make context L3 keep finer information about the ball, which
  lowers the k = 20 goal-head position error.
- **Deciding metric:** the goal-head position error at **k = 20**, ÷ persistence, on the **test**
  split, top-down on vs the reference.
- **Refuted if:** the on setting isn't lower by more than the seed spread.
- **Guards (a win only counts if these hold):**
  - the near heads' `relerr/k1`–`relerr/k5` and the far heads' own `relerr/k10`, `relerr/k20`
    (dev, via `experiments/latent_relerr.py`) are no worse than the reference, beyond the seed
    spread. These are scale-free; `loss/k` and `sqerr/k` would also do between two `mean` runs;
  - the embedding std on every level stays above the floor, and the far `logvar/k` doesn't run
    away.
- **Secondary:** the L3 and L2 linear-probe ball-position error, and probes for the hidden
  parameters at L3.
- **Runs:** top-down on × seeds 0, 1 = **2 new runs** at 48k steps. ≈ 5.5 GPU-h (2 h × 1.35 per
  run). The step is about 35% slower, measured at 201 ms vs 146–151 ms on M4's synthetic bench
  (`models/DESIGN.md`, "Top-down prediction"). That is more than the decision's +10–20 ms estimate.
- **Depends on:** M4, M5, M6. The code has landed.
- **Config:** `configs/m7_top_down.yaml` (`model.top_down: true`), which the experiment lane writes
  with its run, per `/experiment` Step 2.

## 5. Obstacles and occlusion

- **Question:** when the ball is hidden behind a foreground object, while the camera still points
  at it, does the model's uncertainty rise and then fall again, and is it calibrated?
- **Switch:** `sim.obstacles`, a block of keys. It **exists**, from PR #17: `sim/obstacles.py`,
  `sim/scene.py`, `sim/episode.py` (`sim/DESIGN.md`, "Obstacles and occlusion"). Its tuned values
  are in `configs/m7_obstacles.yaml`; that config's datasets aren't built yet.
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
- **Runs:** obstacles × seeds 0, 1 = **2 new runs** at 48k steps, plus a dataset build. ≈ 4 GPU-h.
- **Depends on:** M2, M5, M6, the sim lane, and **the timing below**.
- **Config:** `configs/m7_obstacles.yaml` (`base: exp_headline_48k.yaml`, plus
  `sim.obstacles: {…}`, with the camera left at `track_lagged`, which is SPEC's
  `track_lagged_obstacles`).

### Flag: the obstacle work changes `sim/` and touches dataset hashes

- M2's dataset key hashes the **whole** `sim` config (`jepa_world/data/build.py:split_key`).
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

Every arm is **48,000 steps, seeds 0 and 1**, at ≈ 2 GPU-h per run (Compute basis), per the
[48k decision](../decisions/2026-11-19-m7-ablations-at-48k.md).

| § | Ablation | Switch | Exists now? | Code owner | New runs | Est. GPU-h |
|---|---|---|---|---|---|---|
| — | Reference at the 2026-11-19 defaults (10% share, `latent_reduce: mean`); assigned to m5 | `configs/exp_headline_48k.yaml` | yes | — | 2 | ≈ 4 |
| 1 | Pixel share vs the 10% default: 0 / 30% (1% optional) | `loss.pix_share` (`w_pix: 0` for 0) | **yes** (PRs #25, #30) | — | **done at 24k** (PR #28); at 48k not scheduled: 4, +2 for 1% | 0 (≈ 8, +4 for 1%) |
| 2 | Decoder skips off, at 10% | `model.unet_skips` | **yes** (M5) | — | 2; +2 for a 30% level | ≈ 4 (+4) |
| 3 | track_lagged vs track_full | `sim.camera_mode` | **yes** (M1) | — (dataset build) | 2 | ≈ 4 |
| 4 | Top-down (A) | `model.top_down`, `loss.w_td` | **yes** (M7 top-down lane) | — | 2 | ≈ 5.5 |
| 5 | Obstacles | `sim.obstacles` | **yes** (PR #17) | — (dataset build) | 2 | ≈ 4 |
| | **Total remaining** | | | | **10** | **≈ 21.5** |

- **The total counts** the reference, §2 at 10%, and §3–§5. It covers training plus eval; the two
  dataset builds (§3, §5) add under 5 min each.
- **After m5's reference re-run:** 8 runs, ≈ 17.5 GPU-h.
- **Optional:** §1 at 48k (≈ 8, or ≈ 12 with 1%) and §2's 30% level (≈ 4) bring the total to
  ≈ 33.5–37.5 GPU-h.
- On the one GPU, ≈ 21.5 GPU-h is most of a day of back-to-back runs.

**Order:**
1. The reference, at the 2026-11-19 defaults (m5).
2. §2 at 10%, §3, §4 and §5, back to back, keeping one queued job per lane.
3. If the user approves §1 at 48k, run its 30% arm before §2's 30% level, so the two share it.
