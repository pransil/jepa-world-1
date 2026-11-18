# models/: auxiliary heads (M5): goal head and pixel decoder

This note is the companion to [`DESIGN.md`](DESIGN.md), which covers the JEPA core.

## Purpose / non-goals

SPEC §4.6, §4.7 and §5.2–§5.5:
- `GoalHead`: pooled encoder levels → a diagonal 2-D Gaussian over the ball's displacement, for
  each horizon;
- `PixelDecoder`: the U-Net decoder path → the RGB frame at t+1;
- `WorldModel` (`world.py`): the JEPA core plus these two heads, built for the `goal` / `pixel`
  stages of `train.stages`;
- `compute_world_losses`: the JEPA terms plus the goal and pixel terms.

`goal.py` and `pixel.py` take every size as a constructor argument. `world.py` reads the config.

Out of scope for v1: an action input for the goal head, and full covariance.

## Data flow and shapes (SPEC sizes; the tests use tiny ones)

| Unit | In | Out |
|---|---|---|
| `GoalHead(l3_ch, horizons, …, l2_ch=0)` | L3 (B,256,8,8) [, L2 (B,128,32,32)] | mean, logvar: (B, K, 2) each; K = len(horizons) |
| `displacement_targets(ball_pos, C, horizons)` | ball_pos (B, C+k_max, 2), world m, any float dtype | (B, K, 2) float32: ball_pos[:, C−1+k] − ball_pos[:, C−1] |
| `goal_nll(mean, logvar, target)` | (B, K, 2) ×3 | (K,): NLL per horizon |
| `PixelDecoder(levels, widths, …, skips=True)` | [L1 (B,64,64,64), L2 (B,128,32,32), L3 (B,256,8,8)] | (B, 3, 64, 64) in [0, 1] |
| `pixel_l1(pred, target)` | (B,3,S,S) ×2; target = `frames[:, C]` | scalar |
| `WorldModel.forward(frames, action, active, aux_active)` | frames (B, C+k_max, 3, S, S) | JEPA's dict + `"goal"`: (μ, logvar) + `"pixel"`: (B,3,S,S) |
| `compute_world_losses(model, batch, loss_cfg, …)` | frames tensor, or `{"frames", "ball_pos" (B, C+k_max, 2)}` | every term (below), 0-d each |

```
L3 ─ avgpool ─┐
L2 ─ avgpool ─┴ cat ─ ×K, cat k_embed[j] ─ Linear─GN─SiLU ─ Linear─GN─SiLU ─ Linear(4) ─► μ (2), logvar = b·tanh(raw/b) (2)

L3 ─ up ─ cat L2 ─ ConvBlock×2 ─ up ─ cat L1 ─ ConvBlock×2 ─ 1×1 conv ─ sigmoid ─► frame t+1

batch ─┬─ frames ─► WorldModel ─► JEPA.forward (context L1..L3, heads, targets) ─┬─► JEPA terms (losses_from_output)
       │                          ├─ goal head  (context L3 [, L2]) ─► μ, logvar ─┼─► loss/goal/k, goalerr/k, goalratio/k, loss/goal
       │                          └─ pixel dec. (context L1, L2, L3) ─► frame ────┼─► loss/pix  vs frames[:, C] (t+1)
       └─ ball_pos (a label, never a model input) ─► displacement_targets ───────┘
loss/total = JEPA total + w_goal·loss/goal [goal stage started] + w_pix·loss/pix [pixel stage started]
```

### Logged terms (every step, for every BUILT aux head)

| Key | Meaning |
|---|---|
| `loss/goal/k<k>` | goal NLL of the displacement at horizon k (world m) |
| `goalerr/k<k>` | mean ‖μ − Δ‖ in metres (detached) |
| `goalratio/k<k>` | goalerr / mean ‖Δ‖: error relative to persistence, which predicts Δ = 0; < 1 beats it (SPEC §7.2, §11) |
| `loss/goal` | Σ_k `loss/goal/k<k>`, the term weighted by `loss.w_goal` |
| `loss/pix` | mean \|frame − frame t+1\|, the term weighted by `loss.w_pix` |

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
- **One NLL formula, in `losses.py`** (`gaussian_nll_terms`, including ½·log 2π, so it is a true
  NLL that can go negative). The far heads' `gaussian_nll` and `goal_nll` both use it, each with
  its own reduction. `goal_nll` sums over (x, z), averages over the batch, and returns one value
  per horizon, so the train loop can log `loss/goal/k<k>`. Importing it from `jepa.py` lost: that
  would tie the auxiliary heads to the JEPA core (SPEC §10).
- **The pixel decoder owns the U-Net decoder path.** SPEC §4.3 says the decoder path exists only
  for the pixel decoder. Keeping it here leaves `unet.py`'s encoder untouched. Its last stage reads
  L1 (the skip) plus the upsampled decoder features, as §4.7 asks.
- **`skips` switch (M7 ablation):** `skips=False` drops the L2/L1 concatenations, so the frame is
  decoded from L3 alone. `forward` still takes and checks all three levels, so callers don't
  branch, but L1 and L2 get no gradient. It acts only through the pixel loss, so it does nothing
  at `w_pix = 0`. [skip-connections-meaning](../../docs/decisions/2026-11-17-skip-connections-meaning.md)
  (user decision, option A: the decoder skips, not FPN laterals)
- **`displacement_targets` subtracts at the input's precision, then returns float32.** M2's
  batches already carry float32 (rounding < 2e-6 m, held by an m2 test to 1e-5 m of the float64
  truth). A float64 caller still keeps full precision for the difference, and the cast gives the
  model's fp32 (MPS has no float64).
- **Nearest upsample + 3×3 conv**, not ConvTranspose, which produces checkerboard artefacts.
  Upsampling goes straight to the next level's size (8 → 32 in one step).
- **Sigmoid output**, so the prediction is in [0, 1], the same range as the target. A clamp lost
  because it has no gradient outside the range.
- **`pixel_l1` is the mean over every element**: mean absolute error in [0, 1] intensity units. So
  w_pix · L_pix ≤ w_pix (0.05 at the SPEC weight). `test_pixel_term_does_not_dominate_the_logged_total`
  holds it under 10% of `loss/total` over a short run.
- **The goal head's first Linear runs K times on the same pooled vector.** Splitting it into W_g·g
  (once per sample) plus W_e·k_embed (once per horizon) would save (l3_ch + l2_ch)·hidden·(K−1)
  MACs per sample, about 300k at the base sizes. That is noise next to the encoder's
  convolutions, so one plain `nn.Linear` is kept for clarity.

### Wiring (M5)

- **`WorldModel` subclasses `JEPA` instead of wrapping it.** So the core's `state_dict` keys are
  unchanged, and pre-M5 checkpoints (e.g. the first real run's) load strictly. The aux heads add
  only `goal.*` and `pixel.*` keys. A wrapper lost: it would prefix every key with `jepa.` and
  break those checkpoints. `jepa.py` still never imports the aux heads (SPEC §10). It gains two
  generic hooks: `extra_levels` (levels to always encode) and `grad_levels` (levels an active
  outside consumer trains on).
- **The aux heads read the CONTEXT levels only.** That means frames t−C+1 … t, never the target
  encoder's output. So the pixel decoder predicts t+1 from the past, and the goal head never sees
  the future. `ball_pos` is a label: `compute_world_losses` passes only the frames tensor to
  `forward`. Tests hold both (`test_aux_heads_read_only_the_context_frames`,
  `test_the_model_never_receives_ground_truth`).
- **Stages.** `goal` and `pixel` follow the same within-run schedule as near and far
  ([stage-schedule](../../docs/decisions/2026-11-17-stage-schedule-within-run.md)). A built aux
  head runs and is logged every step. Before its `stage_start`, it runs under `no_grad` and stays
  out of `loss/total`. Once active, its levels keep autograd even if no active horizon head reads
  them. For example, the goal head trains the L3 encoder stage while only near heads are active.
  The start steps and the head sizes are **proposed placeholders**:
  [m5-stage-starts-and-aux-sizes](../../docs/decisions/2026-11-17-m5-stage-starts-and-aux-sizes.md).
- **Active aux losses train the context encoder.** This is deliberate, and it is the point of
  the heads. SPEC §4.7's pixel decoder exists "to keep the scene grounded", which only a gradient
  into the embedding can do. "It must not drive the embedding toward blur" is handled by its low
  weight (w_pix = 0.05) and by keeping it a separate, logged term, not by detaching it. M7's
  "pixel weight at 0, low, and high" ablation and the
  [skip decision](../../docs/decisions/2026-11-17-skip-connections-meaning.md) ("reconstruction
  must pass through L3"; L3 probe as the deciding metric) both assume this gradient. The goal
  term likewise enters SPEC §5.4's total. To train an aux head without shaping the encoder, set
  its weight to 0 or leave its stage out of `train.stages`.
- **A built aux head whose stage hasn't started still runs every step**, under `no_grad`, to be
  logged. This follows the accepted
  [stage-schedule decision](../../docs/decisions/2026-11-17-stage-schedule-within-run.md)
  ("every term logged every step", SPEC §5.4), the same as an inactive far head. The cost is one
  no-grad pixel-decoder forward plus the L2/L3 encoder continuation per step until the stage
  starts. A run that doesn't want it should leave the stage out of `train.stages`.
- **A goal horizon past every model horizon lengthens every batch window** (`batch_k_max`), which
  drops start times t near the episode end for ALL heads. With the base stages it changes nothing,
  because far's k = 20 already sets k_max = 20, and goal_horizons = the model horizons. A run
  that compares near-only with near + goal should cap `goal_horizons` at 5 (or include far), so
  both see the same windows.
- **`loss/goal` sums the per-horizon NLLs** (Σ_k, like the latent terms). The variance guard and
  `emb_std` still cover only the levels horizon heads read: the guard protects the latent
  targets from collapse, which the aux heads don't affect.
- **Batches.** The goal stage switches the loader to `with_gt=True` (`train.needs_gt`). Train
  keeps only `frames` and `ball_pos`. A batch reaches `max(model horizons, goal horizons)` past t
  (`train.batch_k_max`). The synthetic source's `ball_pos` is in frame units (x = u/S, z = 1 −
  v/S), and is for smoke runs only.

## Seams

| Unit | Pure? | Touches | Tested by |
|---|---|---|---|
| `goal.GoalHead`, `goal_nll`, `displacement_targets` | yes | — | `tests/test_goal.py` (CPU; SPEC-shape test at B=2; `@mps` forward/backward) |
| `pixel.PixelDecoder`, `pixel_l1` | yes | — | `tests/test_pixel.py` (same) |
| `world.WorldModel`, `compute_world_losses` | yes | — | `tests/test_world.py` (CPU, tiny) |
| `train` goal/pixel stages, `batch_k_max`, `active_aux`, dict batches | mutates model + optimizer | device, run dir | `tests/test_train.py` (incl. the SPEC M5 dominance test, `@mps` all-stages step), `tests/test_train_episodes.py` |

## Files

| File | Responsibility |
|---|---|
| `goal.py` | `GoalHead`, `goal_nll`, `displacement_targets` |
| `pixel.py` | `PixelDecoder` (U-Net decoder path → RGB), `pixel_l1` |
| `world.py` | `WorldModel` (JEPA + aux heads), `compute_world_losses`, `split_batch` |
| `losses.py` | `gaussian_nll_terms`: the one diagonal-Gaussian NLL formula |
