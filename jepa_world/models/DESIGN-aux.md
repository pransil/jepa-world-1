# models/: auxiliary heads (M5): goal head and pixel decoder

This is a separate note from `DESIGN.md`, which covers the JEPA core, so the two lanes didn't
collide. The wiring PR links it from there.

## Purpose / non-goals

SPEC §4.6, §4.7, §5.2 and §5.3, as **standalone modules**:
- `GoalHead`: pooled encoder levels → a diagonal 2-D Gaussian over the ball's displacement, for
  each horizon;
- `PixelDecoder`: the U-Net decoder path → the RGB frame at t+1.

Both take every size as a constructor argument. Neither reads a config.

The following are not here yet; they belong to the wiring PR, after M2 and M4 merge:
- calling these modules from `JEPA` / `compute_losses`;
- the `loss.w_goal`, `loss.w_pix` and `model.goal_*` / `model.pix_*` keys in `base.yaml`;
- the `goal` and `pixel` stages of §5.5;
- SPEC M5's **"pixel term does not dominate the logged total"** test, which needs the train loop.

Out of scope for v1: an action input for the goal head, full covariance, and skip on/off for M7.

## Data flow and shapes (SPEC sizes; the tests use tiny ones)

| Unit | In | Out |
|---|---|---|
| `GoalHead(l3_ch, horizons, …, l2_ch=0)` | L3 (B,256,8,8) [, L2 (B,128,32,32)] | mean, logvar: (B, K, 2) each; K = len(horizons) |
| `displacement_targets(ball_pos, C, horizons)` | ball_pos (B, C+k_max, 2), world m | (B, K, 2): ball_pos[:, C−1+k] − ball_pos[:, C−1] |
| `goal_nll(mean, logvar, target)` | (B, K, 2) ×3 | (K,): NLL per horizon |
| `PixelDecoder(levels, widths, …)` | [L1 (B,64,64,64), L2 (B,128,32,32), L3 (B,256,8,8)] | (B, 3, 64, 64) in [0, 1] |
| `pixel_l1(pred, target)` | (B,3,S,S) ×2; target = `frames[:, C]` | scalar |

```
L3 ─ avgpool ─┐
L2 ─ avgpool ─┴ cat ─ ×K, cat k_embed[j] ─ Linear─GN─SiLU ─ Linear─GN─SiLU ─ Linear(4) ─► μ (2), logvar = b·tanh(raw/b) (2)

L3 ─ up ─ cat L2 ─ ConvBlock×2 ─ up ─ cat L1 ─ ConvBlock×2 ─ 1×1 conv ─ sigmoid ─► frame t+1
```

## Key decisions

- **The goal target is the displacement from t, in world metres.** Absolute world x can't be seen
  through a tracking camera. Camera-relative position would carry a localisation error that the
  persistence baseline doesn't have.
  [goal-target-displacement](../../docs/decisions/2026-11-17-goal-target-displacement.md)
- **Global average pooling** (SPEC literal: "pooled L3"). It throws away where things are in the
  frame. That matters less for displacement than for absolute position, because the camera keeps
  the ball near the centre. If goal error plateaus, the fallback is a learned attention pool. An
  8×8×256 flatten lost because it costs 16k inputs per horizon.
- **One shared MLP plus a learned vector per horizon**, rather than seven separate heads. It
  follows M4's "one learned k-vector per head", but the trunk is cheap and shared across horizons.
  All K horizons are scored in one batched pass.
- **GroupNorm after each hidden Linear** (norm_groups, as in M4), not BatchNorm, so there are no
  buffers for an EMA to handle and no coupling across the batch. It also keeps the hidden
  activations bounded when inputs are extreme.
- **Diagonal variance, with logvar = b·tanh(raw/b)**: M4's far-head bound. The variance stays
  positive and finite, and the gradient never vanishes. SPEC allows diagonal for v1. A full 2×2
  covariance (Cholesky) would be the next step if the (x, z) errors turn out correlated.
- **`goal_nll` is local** and has the same formula as M4's `gaussian_nll` (including ½·log 2π, so
  it is a true NLL that can go negative). It sums over (x, z), averages over the batch, and returns
  one value per horizon, so the train loop can log `loss/goal/k<k>`. Importing it from `jepa.py`
  lost: it would tie the auxiliary heads to the JEPA core (SPEC §10), and M4 was still changing it.
- **The pixel decoder owns the U-Net decoder path.** SPEC §4.3 says the decoder path exists only
  for the pixel decoder. Keeping it here leaves `unet.py`'s encoder untouched. Its last stage reads
  L1 (the skip) plus the upsampled decoder features, as §4.7 asks.
- **Nearest upsample + 3×3 conv**, not ConvTranspose, which produces checkerboard artefacts.
  Upsampling goes straight to the next level's size (8 → 32 in one step).
- **Sigmoid output**, so the prediction is in [0, 1], the same range as the target. A clamp lost
  because it has no gradient outside the range.
- **`pixel_l1` is the mean over every element**: mean absolute error in [0, 1] intensity units. So
  w_pix · L_pix ≤ w_pix (0.05 at the SPEC weight), which helps the "does not dominate" requirement.
  It is the wiring PR's job to test that.

## Seams

| Unit | Pure? | Touches | Tested by |
|---|---|---|---|
| `goal.GoalHead`, `goal_nll`, `displacement_targets` | yes | — | `tests/test_goal.py` (CPU; SPEC-shape test at B=2; `@mps` forward/backward) |
| `pixel.PixelDecoder`, `pixel_l1` | yes | — | `tests/test_pixel.py` (same) |

## Files

| File | Responsibility |
|---|---|
| `goal.py` | `GoalHead`, `goal_nll`, `displacement_targets` |
| `pixel.py` | `PixelDecoder` (U-Net decoder path → RGB), `pixel_l1` |
