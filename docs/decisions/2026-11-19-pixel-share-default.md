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
- **SPEC §5.4's `w_pix` line** now points here. The formula is unchanged: w_pix is the per-step share weight.
- **Configs whose runs already exist keep their meaning.** `configs/exp_headline.yaml` pins
  `pix_share: null`, the fixed `w_pix` 0.05 its runs trained with. `exp_headline_48k.yaml`
  inherits that, and `m7_pixel_share01/10.yaml` set their own share on top. No other config has
  a pixel stage except `m7_obstacles.yaml`, which has no runs yet. It is "base.yaml plus
  obstacles", so it now trains at 10%. Its comparison needs a reference run at 10% too.
- **The near/far-only configs** (M3, M4, `exp_first_real_run`, `exp_guard_mean`) are unaffected.
  With no pixel stage there is no decoder and no pixel term, exactly as `w_pix` was unused before.
  So `train.check_config` no longer refuses a share without a pixel stage. That check
  (PR #25) would have made every near/far-only config unrunnable under this default.
  `test_every_committed_config_passes_check_config` now covers every file in `configs/`.
- **M7's reference run** (M7-plan "Shared protocol": "the base config after M5, `w_pix` 0.05")
  no longer matches base.yaml. An ablation not yet run that diffs against base.yaml needs a
  reference at 10%. That is the user's call on GPU time, outside this decision.
- **Revisit if** a 30% arm or a longer run shows the "blur" cost at k = 20, or if the latent losses
  are normalized (a SPEC §5 change, under way on 2026-11-19). That rescales L_rest, but the share
  adapts by construction, so the pixel term stays at 10% of whatever the rest is.
