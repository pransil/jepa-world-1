# Default pixel weight: hold the pixel term at 10% of the rest of the loss

- **Date:** 2026-11-19   **Decided by:** user (2026-11-19)   **SPEC ref:** §5.3, §5.4
- **Status:** accepted

## Question
SPEC §5.4's starting weight `w_pix = 0.05` leaves the pixel term at about 3e-5 of the loss or less.
The latent losses are channel sums in the hundreds, and `loss/pix` is a mean absolute error in
[0, 1]. So the default model effectively trains with no pixel term at all. M7 §1's rescaled ablation
([experiments/2026-11-18-pixel-share-ablation.md](../../experiments/2026-11-18-pixel-share-ablation.md))
compared holding the term at a target share of the rest of the loss, using `loss.pix_share`
([decision](2026-11-18-pixel-ablation-target-share.md)), at ≈ 0, 1% and 10%. Should the default
change?

## Options
- **A: keep the fixed `w_pix = 0.05`** (`pix_share: null`). This is SPEC §5.4 as written, and in
  effect no pixel term.
- **B: `pix_share: 0.10`**, the ablation's best arm. On the test split, at 24k steps, seeds 0 and 1:
  - **The goal head's k = 20 ratio to persistence (the deciding metric) is not worse:** 0.416 at
    10%, against 0.430 at ≈ 0.
  - **k = 1–5 improve by 10–15%.** Both 10% seeds are below both ≈ 0 seeds at every k from 1 to 5.
  - **The decoder reconstructs better:** `loss/pix` is 22% lower.
  - **Calibration and the visibility response don't change**, and nothing collapses (the
    embedding std is 1.8–1.9 at every level).
- **C: a share above 10%.** Not tested. A 30% arm is proposed in the experiment note.

## Decision
**B: `loss.pix_share: 0.10` in `configs/base.yaml`.** It was the only arm that improved anything
beyond the seed spread, and it cost nothing on the deciding metric. `w_pix` stays in base.yaml as
the fallback that `pix_share: null` uses.

This is a share of the loss **value**, not of the encoder's gradient, which isn't measured
(see the ablation decision).

## Consequences
- **base.yaml:** `loss.pix_share: 0.10`, with `w_pix: 0.05` kept for `null`.
- **SPEC §5.4's `w_pix` line** now points here, a minimal edit the user approved with this
  decision (2026-11-19, relayed by the orchestrator session). The formula is unchanged: w_pix
  becomes the per-step share weight.
- **Configs with a pixel stage:**
  - `m7_pixel_share01/10.yaml` set their own share, so they keep their meaning.
  - `exp_headline.yaml`, `exp_headline_48k.yaml` (the M7 reference) and `m7_obstacles.yaml` (an
    M7 ablation on top of it) **inherit the new default**, as
    [m7-ablations-at-48k](2026-11-19-m7-ablations-at-48k.md) requires: the reference is base.yaml's
    defaults at the commit it runs at, and is re-run once this and the latent normalization land.
    `test_the_m7_reference_and_its_ablations_inherit_the_base_pixel_weighting` pins that.
  - The existing headline runs (24k, commit ba81066; 48k, 5ed1564) trained at the fixed `w_pix`
    0.05. To reproduce them, check out their commit or add `--set loss.pix_share=null`. The
    ≈ 0 arm of the pixel ablation ([decision](2026-11-18-pixel-ablation-target-share.md)) is
    those runs.
- **The near/far-only configs** (M3, M4, `exp_first_real_run`, `exp_guard_mean`) are unaffected.
  With no pixel stage there is no decoder and no pixel term, exactly as `w_pix` was unused before.
  So `train.check_config` no longer refuses a share without a pixel stage. That check
  (PR #25) would have made every near/far-only config unrunnable under this default.
  `test_every_committed_config_passes_check_config` now covers every file in `configs/`.
- **M7's reference** is handled by [m7-ablations-at-48k](2026-11-19-m7-ablations-at-48k.md): its
  reuse check now includes `configs/base.yaml`, so this change forces the re-run. M7-plan
  §1/§2 still describe `w_pix` 0.05 as "the reference" / "the base" for the pixel and skip
  ablations. That is the m7 lane's file, so it is not edited here; the m7 lane and the
  orchestrator were told.
- **An explicit share without a pixel stage** (e.g. `--set train.stages=[near, far, goal]` on
  a share config) is accepted silently, as `w_pix` always was. The stage list is the switch
  that turns the pixel term off, and an inherited default can't be told apart from an explicit
  setting.
- **Revisit if** a 30% arm or a longer run shows the "blur" cost at k = 20, or if the latent losses
  are normalized (a SPEC §5 change, under way on 2026-11-19). That rescales L_rest, but the share
  adapts by construction, so the pixel term stays at 10% of whatever the rest is.
