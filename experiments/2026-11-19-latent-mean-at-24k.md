# Per-channel latent losses at 24k steps, with the pixel share held at 10%

- **Date:** 2026-11-19   **Lane/branch:** worktree-wt-exp-latent-mean-24k
- **Commit:** the commit that adds this note, which is pre-registered before the evals. The evals
  run at it, clean. The sha is filled in under Results.
- **Asked for:** proposal "Next 2" in [2026-11-19-headline-48k-new-defaults.md](2026-11-19-headline-48k-new-defaults.md),
  approved by the user 2026-11-19 (task E, relayed by the orchestrator session).
- **No training.** Three evals, each `uv run python -m jepa_world.eval --run <dir> --checkpoint
  <dir>/checkpoints/step_0024000.pt --split test --split dev`, and a JSON-only scorer
  (`experiments/latent_mean_24k_check.py`).

## Question
The 48k reference at the new defaults beat the old reference at every horizon, including the
deciding k = 20: 0.330 against 0.350 of persistence's error, on test. Three things changed at once:
1. per-channel latent losses (`latent_reduce: mean`);
2. a 10% pixel share;
3. a goal weight 4× smaller relative to L1's latent terms, and 2× relative to L2's
   (`w_goal_eff` = 0.5 / 256). This follows from change 1 in code.

The 24k pixel-share ablation isolated change 2: k = 20 stayed within the seed spread, and k = 1–5
improved. **Does changes 1 + 3, with the pixel share held at 10%, move the goal head at 24k?**

## The comparison: two runs that differ only in `latent_reduce`
| side | runs | `latent_reduce` | pixel | scored at |
|---|---|---|---|---|
| **sum** | `m7_pixel_share10` s0 / s1 (commit f1a7c03, [note](2026-11-18-pixel-share-ablation.md)): `20261119T115353Z-…-pixshare10-s0-aad2ee`, `20261119T134114Z-…-pixshare10-s1-880ec0` | `sum` (key absent) | share 0.10 | step 24,000, their final checkpoint; their existing evals |
| **mean** | the 48k reference at the new defaults (commit 6fd7059, [note](2026-11-19-headline-48k-new-defaults.md)): `20261119T224253Z-…-headline48k-new-s0-0f2411`, `20261120T003253Z-…-headline48k-new-s1-ada237` | `mean` | share 0.10 | `checkpoints/step_0024000.pt` |

- **Configs:** the recorded `config.yaml` of each seed pair differs in exactly two leaves:
  `loss.latent_reduce` (absent → `mean`) and `train.steps` (24,000 → 48,000). This was checked by
  diffing every leaf of the parsed files.
- **A step-24,000 checkpoint of a 48k run is a 24k run:**
  - there is no LR schedule;
  - `train.steps` sets only the loop length, the stage-start check and the final checkpoint
    (`jepa_world/train.py`);
  - training is deterministic on MPS. The old 48k runs matched the 24k headline runs bitwise
    through step 24,000 ([2026-11-18-headline-48k.md](2026-11-18-headline-48k.md)).
- **The training code is the same up to the switch.** The `sum` path at 6fd7059 is bitwise
  f1a7c03's (`experiments/latent_reduce_bitwise.py`,
  [decision](../docs/decisions/2026-11-19-latent-loss-normalization.md)).
- **Eval code:** `jepa_world/eval.py` changed between f1a7c03 and this commit only by an optional
  pixel output (off by default) and a refactored file write. The model path's `sum` branch is
  bitwise unchanged. A control eval checks this with an artifact instead of code reading
  (step 2 below).

## Hypothesis (pre-registered)
**H: the per-channel latent change (1 + 3) produced the reference's k = 20 gain, not the pixel
share.** If so, it shows at 24k with the pixel share fixed: the `mean` side beats the `sum` side at
k = 20.

- **Deciding metric:** goal-head position error at **k = 20**, as a ratio to persistence, on the
  **test** split (200 episodes, 7400 windows), at step 24,000.
- **Verdict rule** (per horizon, as in the reference's note):
  - **better** if both `mean` seeds are below both `sum` seeds;
  - **worse** if both are above;
  - otherwise **within the seed spread**.
- **Outcomes at k = 20:**
  - better → **H supported at 24k**;
  - worse → **H refuted at 24k**;
  - within → **inconclusive** (no difference claimed).
- **Reported without a prediction:**
  - k = 1–5 and 10, with the same rule;
  - dev, separately (SPEC §7.3);
  - against constant velocity and the mean-parameter simulator;
  - calibration (50% / 90%), visibility `std_ratio` and hidden coverage;
  - probes (`pos_cam` L1–L3 and pooled, `radius`, world `pos`);
  - training-batch `relerr`, `emb_std`, far `logvar`, `loss/pix`, `w_pix_eff`, `w_goal_eff` at
    23,001–24,000, and `goalratio` per 4k window.

**My expectation, stated separately from H:** k = 20 within the seed spread, or worse.
- **Disclosed: I had already seen the training-batch tables.** A dry run of the scorer printed them
  before I wrote this.
- **What they showed on training batches:** the `mean` runs escaped persistence later. Their k1
  `goalratio` over steps 12,001–16,000 was 0.927 / 0.925, against 0.854 / 0.845 for `sum`. Over
  20,001–24,000, k20 was 0.464 / 0.468 for `mean`, against 0.464 / 0.416.
- **Why that is a weak guide:** at 48k the reference's k = 20 gain showed on test and dev, but not
  in training-batch `goalratio` (its note).

**The limit of a 24k comparison.** At 24k the goal head is still on the steep part of its curve.
That is the reason M7 moved to 48k ([decision](../docs/decisions/2026-11-19-m7-ablations-at-48k.md)).
If `mean` learns more slowly early but ends better, a 24k comparison shows the timing, not the end
point. So an inconclusive or worse result at 24k does **not** rule out H at 48k. Only a 48k run of
`sum` + 10% would isolate the change there (≈ 4 GPU-h, a proposal).

## Procedure
Evals run one at a time, under the GPU lock, only with no other project job running and ≥ 10 GB
free. The worktree stays clean throughout: `eval.py` records the git state at the end.

1. Eval `mean` s0 and `mean` s1 at `step_0024000.pt`.
2. **Control:** re-eval `sum` s0 at its step-24,000 checkpoint, at this commit. The scorer compares
   every number in the two reports.
   - **Identical:** the original `sum` evals (f1a7c03) stand for both seeds.
   - **Any difference:** re-eval `sum` s1 too, and score both `sum` seeds from the re-evals
     (`--sum-eval latest`). Report the size of the difference.
3. Score with
   `uv run python experiments/latent_mean_24k_check.py --mean <mean s0> --mean <mean s1> --sum <sum s0> --sum <sum s1>`.

**Budget:** 3 evals (4 if the control differs), about 3 minutes each on MPS. No training.

## Results
(Filled in after the evals.)

## Conclusion
(Filled in after the evals.)

## Next
(Filled in after the evals.)
