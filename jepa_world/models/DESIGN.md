# models/: JEPA core (M3 near heads, M4 far heads)

## Purpose / non-goals

The JEPA loop (SPEC §4.3–4.5, §5.1): a U-Net context encoder, an EMA target encoder, one
predictor head per horizon, and the latent loss with a variance guard.
- **M3** added the **near heads**: k = 1..5, reading L1, deterministic mean, squared-L2 loss.
- **M4** adds the **far heads**: k = 10 reading L2 and k = 20 reading L3, each predicting a
  mean and a log-variance, scored by a Gaussian NLL.

All of it is trained by `jepa_world/train.py`, with the §5.5 stage schedule.

Not here yet:
- the U-Net decoder path, which exists only for the pixel decoder (M5);
- the goal head and pixel decoder (M5);
- top-down prediction (SPEC §9.2);
- checkpoint resume;
- the real episode dataset (M2).

Until M2 lands, the synthetic moving-disc source (`data/synthetic.py`) feeds smoke runs and
benchmarks. It is not evidence about the task.

## Batch contract (the M2 ↔ model interface)

`frames: (B, C + k_max, 3, H, W)`, float32 in [0, 1]. C = `model.context_frames` = 4, and index
C−1 is time t. k_max is the largest built horizon: 5 for near only (9 frames), 20 with far
(24 frames).

| Tensor | Slice / module | Shape (SPEC sizes) |
|---|---|---|
| context input | `frames[:, :C]` → stack | (B, 12, 64, 64), RGB of t−3 … t in time order |
| target input, horizon k | `frames[:, k:k+C]` → stack | (B, 12, 64, 64), frames t+k−3 … t+k ([decision](../../docs/decisions/2026-11-17-target-input-stack.md)) |
| context / target levels | `UNetEncoder(x, upto)` | L1 (B,64,64,64) · L2 (B,128,32,32) · L3 (B,256,8,8) |
| near head k = 1..5 | `HorizonHead(L1)` | mean (B,64,64,64) |
| far head k = 10 | `GaussianHead(L2)` | mean, logvar (B,128,32,32) each |
| far head k = 20 | `GaussianHead(L3)` | mean, logvar (B,256,8,8) each |

```
frames ─┬─ [:, :C] ─► context_encoder ─► L1 ─► near heads k=1..5 ─► mean ───────────► loss/k = mean_{b,h,w} Σ_c (μ−tgt)²
        │                               ├─ L2 ─► far head k=10 ─► mean, logvar ─┐
        │                               └─ L3 ─► far head k=20 ─► mean, logvar ─┴──► loss/k = mean_{b,h,w} Σ_c ½[(tgt−μ)²e^{−logvar} + logvar + log 2π]
        │                  every level an ACTIVE head reads ─► variance_guard ─► loss/var
        └─ [:, k:k+C] grouped by the level the head reads ─► target_encoder (no_grad, upto that level) ─► detach ─► tgt
loss/total = Σ_{k active} w_k·loss/k + w_var·loss/var        after opt.step():  θ_target ← m·θ_target + (1−m)·θ_context
```

## Key decisions

- **Target input is the stack ending at t+k**, not a lone frame. A lone frame would be out of
  distribution for an encoder (and its EMA copy) trained on 4-frame stacks.
  [target-input-stack](../../docs/decisions/2026-11-17-target-input-stack.md)
- **Downsampling: 3×3 stride-2 conv** per 2× step, not pooling. L2→L3 (32→8) is two steps, with
  an unexposed 16×16 stage. [downsample-stride2](../../docs/decisions/2026-11-17-downsample-stride2.md)
- **Latent distance: squared L2 on raw embeddings**, summed over channels and averaged over
  positions (SPEC §5.1), not cosine or normalized targets.
  [latent-distance-sq-l2](../../docs/decisions/2026-11-17-latent-distance-sq-l2.md)
- **Far heads read only their own level** (k = 10 → L2, k = 20 → L3; SPEC §9.6). They do not get
  pooled finer levels or a global summary.
  [far-heads-read-their-level](../../docs/decisions/2026-11-17-far-heads-read-their-level.md)
- **Far loss: the full diagonal Gaussian NLL**, including ½·log 2π, summed over channels and
  averaged over positions. That is the same reduction as the near loss: with logvar = 0 it is
  ½·(the near loss) + a constant. Including the constant makes the logged value a true NLL (it
  can go negative). `torch.nn.functional.gaussian_nll_loss` lost because it takes a variance,
  clamps it with an eps, and averages over channels rather than summing.
- **Log-variance is soft-bounded:** logvar = b·tanh(raw / b), b = `model.far_logvar_bound` = 10.
  So the variance is always positive and finite (e^±10). It is the identity near 0 and keeps a
  gradient everywhere. A hard `clamp` lost because it zeroes the gradient once outside the range.
- **Far-head width = the level's channels** (`l2_head_hidden: 128`, `l3_head_hidden: 256`, 3×3).
  SPEC fixes only the L1 width. At 32×32 and 8×8 these heads are cheap: the measured step time
  is in the PR and `docs/milestones/M4.md`. 32 wide like L1 lost: it would squeeze 128/256
  channels through a much narrower layer, for no compute reason.
- **Each far head also logs `sqerr/k`** (its mean's squared L2, the same metric as the near
  loss) **and `logvar/k`** (mean predicted log-variance). NLL can fall just by tightening the
  variance. `sqerr` shows whether the *prediction* improved, and `logvar` shows the uncertainty
  that SPEC §7.2 later calibrates.
- **Stage schedule within a run** (SPEC §5.5): every built head runs and is logged every step,
  but only *active* horizons (stage started, `train.stage_start`) enter `loss/total`, and the
  variance guard covers only levels read by active heads. An inactive head gets no gradient.
  [stage-schedule-within-run](../../docs/decisions/2026-11-17-stage-schedule-within-run.md)
- **The target encoder runs once per level group:** the stacks of all heads reading a level are
  concatenated along the batch and encoded `upto` that level. Near stacks stop at L1, and only the
  two far stacks pay for L2/L3. GroupNorm is per sample, so grouping can't change a result.
- **GroupNorm, not BatchNorm.** BatchNorm's running stats are buffers the EMA would have to
  handle, and its batch statistics would couple the target stacks to each other. GroupNorm has
  neither problem, so the EMA covers parameters only. `JEPA` raises if the encoder gains buffers.
- **One head per horizon, each with its own learned k-vector** (SPEC §4.5), plus an action slot
  (a (B, action_dim) vector broadcast over positions; absent when `action_dim: 0`).
- **EMA after `optimizer.step()`**, once per step: `pt.mul_(m).add_(pc, alpha=1−m)`. Target
  parameters have `requires_grad=False` and are never in the optimizer.
- **Logging without per-step syncs:** `train.py` buffers each step's loss vector on device and
  flushes every `train.log_every` steps (and on a crash). Every step is still logged.

## Seams

| Unit | Pure? | Touches | Tested by |
|---|---|---|---|
| `unet.UNetEncoder`, `heads.HorizonHead`, `heads.GaussianHead` | yes | — | `tests/test_unet.py`, `tests/test_heads.py` (CPU, 16×16) |
| `jepa.JEPA`, `latent_loss`, `gaussian_nll`, `variance_guard`, `compute_losses`, stack slicing | yes | — | `tests/test_jepa.py` |
| `data/synthetic.moving_disc_batch` | yes (explicit `torch.Generator`) | — | `tests/test_synthetic.py` |
| `train.train_step`, `train.active_horizons` | mutates model + optimizer | device | `tests/test_train.py` |
| `train.run` / `train.main` | no | run dir (`train.start_run`), GPU lock (always taken by `main`) | `tests/test_train.py` (CPU, tmp artifact root) |
| `bench.bench_heads` / `bench.main` | no | MPS timing, GPU lock, run dir | `tests/test_bench.py`; `@mps` test in `tests/test_train.py` |

## Files

| File | Responsibility |
|---|---|
| `unet.py` | `ConvBlock` (conv → GroupNorm → SiLU), `UNetEncoder` (levels from config, `upto` early stop), `encoder_from_cfg` |
| `heads.py` | `HorizonHead` (features + k-vector + action slot → mean map), `GaussianHead` (→ mean, bounded logvar) |
| `jepa.py` | `JEPA` (context/target encoders, heads, `ema_update`), stack slicing, `latent_loss`, `gaussian_nll`, `variance_guard`, `compute_losses` |
