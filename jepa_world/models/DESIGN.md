# models/: JEPA core (M3: near-horizon)

## Purpose / non-goals

M3 builds the JEPA loop for the **near horizons k = 1..5** (SPEC §4.3–4.5, §5.1): a U-Net context
encoder, an EMA target encoder, one predictor head per horizon reading L1, and the latent loss with
a variance guard. All of it is trained by `jepa_world/train.py`.

Not in M3: the U-Net decoder path (it exists only for the pixel decoder, M5), far heads and their
Gaussian NLL (M4), the goal head and pixel decoder (M5), checkpoint resume, and the real episode
dataset (M2). Until M2 lands, the synthetic moving-disc source (`data/synthetic.py`) feeds smoke
runs and benchmarks. It is not evidence about the task.

## Batch contract (the M2 ↔ M3 interface)

`frames: (B, C + k_max, 3, H, W)`, float32 in [0, 1]. C = `model.context_frames` = 4, and index
C−1 is time t. k_max is the largest active horizon (5 in M3, so 9 frames; 20 in M4).

| Tensor | Slice | Shape (SPEC sizes, B = batch) |
|---|---|---|
| context input | `frames[:, :C]` → stack | (B, 12, 64, 64), RGB of t−3 … t in time order |
| target input, horizon k | `frames[:, k:k+C]` → stack | (B, 12, 64, 64), frames t+k−3 … t+k ([decision](../../docs/decisions/2026-11-17-target-input-stack.md)) |
| context / target levels | `UNetEncoder(x, upto)` | L1 (B,64,64,64) · L2 (B,128,32,32) · L3 (B,256,8,8) |
| near head k | `HorizonHead(L1, action)` | (B,64,64,64), deterministic mean |

```
frames ─┬─ [:, :C] ──► context_encoder ──► L1 ──► head_k (+ k-vector, + action slot) ──► pred_k ─┐
        │                                   └──► variance_guard ──► loss/var                      ├─► loss/k = mean_{b,h,w} Σ_c (pred−tgt)²
        └─ [:, k:k+C] for all k, one (B·K) batch ──► target_encoder (no_grad, upto L1) ──► detach ┘
after opt.step():  θ_target ← m·θ_target + (1−m)·θ_context
```

## Key decisions

- **Target input is the stack ending at t+k**, not a lone frame. A lone frame would be out of
  distribution for an encoder (and its EMA copy) trained on 4-frame stacks.
  [2026-11-17-target-input-stack.md](../../docs/decisions/2026-11-17-target-input-stack.md)
- **Downsampling: 3×3 stride-2 conv** per 2× step, not pooling. L2→L3 (32→8) is two steps, with
  an unexposed 16×16 stage. [2026-11-17-downsample-stride2.md](../../docs/decisions/2026-11-17-downsample-stride2.md)
- **Latent distance: squared L2 on raw embeddings**, summed over channels and averaged over
  positions (SPEC §5.1), not cosine or normalized targets.
  [2026-11-17-latent-distance-sq-l2.md](../../docs/decisions/2026-11-17-latent-distance-sq-l2.md)
- **Variance guard on by default**, on the batch std of each (channel, position) feature of every
  level a head reads. Variance uses `correction=0`, so B = 1 gives a constant instead of NaN.
  [2026-11-17-collapse-guard-on.md](../../docs/decisions/2026-11-17-collapse-guard-on.md)
- **GroupNorm, not BatchNorm.** BatchNorm's running stats are buffers that the EMA would have to
  copy or average, and its batch statistics would couple the B·K target stacks to each other
  (a perturbed frame at one horizon would shift another horizon's target). GroupNorm has neither
  problem, so the EMA covers parameters only. `JEPA` raises if the encoder ever gains buffers.
  SiLU is the nonlinearity.
- **One head per horizon, each with its own learned k-vector** (SPEC §4.5). With separate heads,
  a shared `nn.Embedding` would add nothing. The action slot is a (B, action_dim) vector
  broadcast like the k-vector. With `action_dim: 0` it is absent. With action_dim > 0 and no
  action given, zeros are used.
- **The target encoder runs once per step** on all K target stacks concatenated along the batch,
  and stops at the deepest level a head reads (`upto="L1"` in M3, which skips L2/L3 entirely).
  The context encoder stops there too, so in M3 the L2/L3 stages get no gradient and stay at
  their initialization. AdamW skips parameters whose grad is None, which includes weight decay.
  They start training when M4's far heads read them.
- **EMA is applied after `optimizer.step()`**, once per step, as `pt.mul_(m).add_(pc, alpha=1−m)`.
  The target starts as a deep copy of the context encoder, and its parameters have
  `requires_grad=False` and are never given to the optimizer (`trainable_parameters()`).
- **Logging without per-step syncs:** `train.py` buffers each step's detached loss vector on
  device and writes them to `metrics.jsonl` every `train.log_every` steps. Every step is still
  logged (SPEC §5.4), and MPS is synced once per flush, not once per term per step.

## Seams

| Unit | Pure? | Touches | Tested by |
|---|---|---|---|
| `unet.UNetEncoder`, `heads.HorizonHead` | yes | — | `tests/test_unet.py`, `tests/test_heads.py` (CPU, 16×16) |
| `jepa.JEPA`, `latent_loss`, `variance_guard`, `compute_losses`, `context_stack`, `target_stack` | yes | — | `tests/test_jepa.py` |
| `data/synthetic.moving_disc_batch` | yes (explicit `torch.Generator`) | — | `tests/test_synthetic.py` |
| `train.train_step` | mutates model + optimizer | device | `tests/test_train.py` |
| `train.run` / `train.main` | no | run dir (`paths.new_run_dir`), GPU lock on MPS | `tests/test_train.py` (CPU, tmp artifact root) |
| `bench.bench_heads` / `bench.main` | no | MPS timing, GPU lock, run dir | `tests/test_bench.py`; `@mps` test in `tests/test_train.py` |

## Files

| File | Responsibility |
|---|---|
| `unet.py` | `ConvBlock` (conv → GroupNorm → SiLU), `UNetEncoder` (levels from config, `upto` early stop), `encoder_from_cfg` |
| `heads.py` | `HorizonHead`: features + k-vector + action slot → mean map |
| `jepa.py` | `JEPA` (context/target encoders, heads, `ema_update`), stack slicing, `latent_loss`, `variance_guard`, `compute_losses` |
