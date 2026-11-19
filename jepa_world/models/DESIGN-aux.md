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
loss/total = JEPA total + w_goal·loss/goal [goal stage started] + w_pix_eff·loss/pix [pixel stage started]
```

### Logged terms (every step, for every BUILT aux head)

| Key | Meaning |
|---|---|
| `loss/goal/k<k>` | goal NLL of the displacement at horizon k (world m) |
| `goalerr/k<k>` | mean ‖μ − Δ‖ in metres (detached) |
| `goalratio/k<k>` | goalerr / mean ‖Δ‖: error relative to persistence, which predicts Δ = 0; < 1 beats it (SPEC §7.2, §11) |
| `loss/goal` | Σ_k `loss/goal/k<k>`, the term weighted by `loss.w_goal` |
| `loss/pix` | mean \|frame − frame t+1\|, the term weighted by `loss.w_pix` (or by `loss.pix_share`) |
| `w_pix_eff` | the pixel weight used this step: `w_pix`, or the share weight; 0 before the pixel stage starts |
| `pix_share` | `w_pix_eff·loss/pix / \|loss/total without it\|`: the pixel term's achieved share |

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
  a fixed weight bounds the term, w_pix · L_pix ≤ w_pix (`test_fixed_pixel_weight_term_is_bounded_by_w_pix`).
  At base.yaml's 10% share the term is 0.10 of |the rest of `loss/total`| by construction (the
  logged `pix_share`), so 0.10/1.10 of `loss/total` while the rest is positive, as it was
  throughout the headline and ablation runs (`test_pixel_term_does_not_dominate_the_logged_total`).
- **Pixel weight: fixed, or a target share** (`loss.pix_share`, M7).
  [pixel-ablation-target-share](../../docs/decisions/2026-11-18-pixel-ablation-target-share.md)
  - **null** keeps `w_pix`, through exactly the old expression. It is bitwise the
    pre-2026-11-18 loss and gradients (`test_pix_share_null_is_bitwise_the_fixed_weight`).
    It was the default until 2026-11-19. Since then base.yaml sets **0.10**, and the configs
    whose runs predate that pin null
    ([pixel-share-default](../../docs/decisions/2026-11-19-pixel-share-default.md)). Without a pixel stage the share is a
    no-op, like w_pix: there is no decoder.
  - **A share s** sets `w_eff = s·|L_rest| / max(L_pix, 1e-6)` (`pix_share_weight`), where
    L_rest is `loss/total` without the pixel term.
  - **Both sides are detached**, so the gradient flows only through L_pix, and the rest keeps its
    own gradient, unscaled. A non-detached L_pix would cancel the decoder's gradient outright,
    because w·L_pix = s|L_rest|.
  - **|L_rest|**, because the far and goal NLLs can push the rest small or negative.
  - **The 1e-6 floor** is the divide-by-zero guard. It can only make the share fall short of s,
    never exceed it.
  - **Per step and stateless: no EMA, no clamp on w_eff.** Evidence: the two headline runs'
    `metrics.jsonl` (`runs/20261118T231506Z-worktree-wt-exp-headline-headline-s0-32ff77` and `runs/20261119T001708Z-worktree-wt-exp-headline-headline-s1-d2c5c1`), with L_rest = loss/total − 0.05·loss/pix, over steps 6001–24000:
    - L_rest stayed within 431–1640 and was never negative.
    - |L_rest|/L_pix moved about 11% step to step from step 6501 on (median; p95 33%).
    - It rose about 50× in the first 500 pixel steps, as the decoder learned: L_pix fell from about
      0.4 to about 0.01. After that it rose only about 5× (about 8e4 → 4.5e5).
    - The smallest L_pix was 0.0012, three orders of magnitude above the 1e-6 floor.

    So w_eff ramps up as the decoder learns, then moves slowly, about 11% per step, since both
    values are batch means. An EMA would smooth only that 11%, but would add cross-step state that
    checkpoints don't carry. A cap would silently break "share = s". The stateless weight is exact
    every step and is reproducible from the step alone.
  - The share is of the loss **value**, not of the gradient norm. Adam rescales per parameter, so
    the pixel decoder's own parameters barely notice w_eff. The share matters where the pixel
    gradient meets the latent gradients, in the shared encoder, and the pixel term's share of that
    gradient is **not measured**. It is about s × (L_pix's relative gradient / L_rest's), which
    may be well above or below s. Logging the per-term encoder gradient norm on log steps would
    measure it; that is a proposed follow-up, not built.
  - Every run with a pixel decoder logs `w_pix_eff` (0 before the stage starts) and the achieved
    `pix_share`. Older runs predate these keys; for them, compute the share as
    w_pix·loss/pix / |loss/total − w_pix·loss/pix|.
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
  The start steps and the head sizes are placeholders, **accepted by the user on 2026-11-18**:
  [m5-stage-starts-and-aux-sizes](../../docs/decisions/2026-11-17-m5-stage-starts-and-aux-sizes.md).
- **Active aux losses train the context encoder.** This is deliberate, and it is the point of
  the heads. SPEC §4.7's pixel decoder exists "to keep the scene grounded", which only a gradient
  into the embedding can do. "It must not drive the embedding toward blur" is handled by its low
  weight and by keeping it a separate, logged term, not by detaching it. (The weight was w_pix = 0.05
  until 2026-11-19, and is a 10% value share since then. M7's ablation found that 10% doesn't
  hurt k = 20: experiments/2026-11-18-pixel-share-ablation.md.) M7's
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
