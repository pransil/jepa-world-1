# Pixel ablation: hold the pixel term at a target share of the loss

- **Date:** 2026-11-18   **Decided by:** user (2026-11-18)   **SPEC ref:** §5.3, §5.4, §8 M7
- **Status:** accepted

## Question
M7-plan §1 compares `loss.w_pix` at 0, 0.05 and 0.5. The headline runs
(experiments/2026-11-17-headline-full-model.md, proposal #3) show that at w_pix = 0.05 the pixel
term peaks at about 3e-5 of |loss/total| and averages about 2e-7. The latent losses are channel
sums in the hundreds that grow with the embedding spread, while `loss/pix` is a [0, 1] mean
absolute error that falls about 20× over a run. So any fixed weight in that range leaves the pixel
term effectively at zero, and 0 / 0.05 / 0.5 would compare three near-zero settings.

## Options
- **A: fixed weights, as planned** (0 / 0.05 / 0.5). This needs no code, but every level is
  effectively zero.
- **B: much larger fixed weights.** The share would still drift by orders of magnitude during a run,
  because the ratio |rest| / L_pix grows about 250× over the headline runs. "A level" would not mean
  one share.
- **C: a target share.** `loss.pix_share = s` sets the pixel weight each step to
  w_eff = s · |L_rest| / L_pix, from detached values, so the pixel term is s · |L_rest|. L_rest is
  loss/total without the pixel term. The gradient flows only through L_pix.

## Decision
**C**, at three levels, each a 24k-step headline run with seeds 0 and 1:

| Level | Config | Pixel share of the rest of the loss |
|---|---|---|
| ≈ 0 | the existing headline runs (`configs/exp_headline.yaml`, w_pix 0.05) | ≤ 3e-5 (measured) |
| 1% | `configs/m7_pixel_share01.yaml` (`loss.pix_share: 0.01`) | 0.01 |
| 10% | `configs/m7_pixel_share10.yaml` (`loss.pix_share: 0.10`) | 0.10 |

## Consequences
- `loss.pix_share` exists in `configs/base.yaml`, default null.
  - **Null is bitwise the old fixed-weight loss.** A test pins it, and the PR records a
    side-by-side check against the previous code.
  - When a share is set, `w_pix` is ignored.
- Every run with the pixel decoder logs `w_pix_eff` and the achieved `pix_share` every step.
- The weighting is per step and stateless (no EMA). It uses |L_rest|, and floors L_pix at 1e-6, so
  the share can fall short of the target but never exceed it. See models/DESIGN-aux.md.
- The share is a share of the loss **value**, not of the gradient norm.
- M7-plan §1's hypothesis and deciding metric (k = 20 goal-position error vs persistence on the
  test split) carry over, with "high" = 10% and "low" = 1%.
