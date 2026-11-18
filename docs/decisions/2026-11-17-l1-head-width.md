# L1 head: 32 channels wide, 3×3 convolutions

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §9.4, §11 ("Benchmark the L1 heads on the Mac early")
- **Status:** accepted

## Question
The five near heads run at full 64×64 resolution, so their cost could dominate a step. Should the
L1 head stay at SPEC's starting width of 32 with 3×3 convs, be narrowed, or switch to 1×1 convs?

## Evidence
`uv run python -m jepa_world.bench --config configs/m3_near_synthetic.yaml` on the M4 (MPS, fp32,
B = 32, SPEC shapes, 20 timed iterations after 5 warmup, under the GPU lock). Synthetic frames:
timing depends on shapes, not content. **No op fell back to CPU.**

| hidden | kernel | 5 heads fwd+bwd (ms) | full train step (ms) | head params | MPS driver MB |
|---:|---:|---:|---:|---:|---:|
| 16 | 3×3 | 27.3 / 28.2 | 55.9 / 73.2 | 69,320 | 2105 |
| **32** | **3×3** | **40.3 / 49.0** | **74.7 / 92.5** | **161,320** | **2105** |
| 64 | 3×3 | 80.0 / 97.0 | 101.5 / 130.0 | 414,440 | 3137 |
| 16 | 1×1 | 23.4 / 24.6 | 55.2 / 59.4 | 13,000 | 2105 |
| 32 | 1×1 | 33.3 / 32.2 | 65.3 / 68.5 | 28,200 | 2105 |
| 64 | 1×1 | 46.2 / 47.2 | 79.4 / 83.9 | 66,280 | 3129 |

Each cell is run 2 / run 1. Run 2: `runs/20261118T042046Z-worktree-wt-m3-near-jepa-bench-l1-heads-a92e58`
(after the review fixes to the memory measurement). Run 1:
`runs/20261118T021043Z-worktree-wt-m3-near-jepa-bench-l1-heads-a5add4`. Step times varied by up
to 25% between the two runs. The ranking did not change.

## Options
- **32 wide, 3×3** (SPEC's starting value): 75–93 ms per step, so 10k steps take 12–15 min.
  The two 3×3 head convs widen the L1 receptive field from the encoder's 5×5 px to 9×9 px.
- **16 wide, 3×3**: about 20% faster, same receptive field, half the head capacity.
- **32 wide, 1×1**: about 15–25% faster, but the head adds no spatial context. Each prediction
  sees only 5×5 px, less than the ball's travel by k = 2–5 (it can move ~4 px per frame).
- **Keep [OPEN]** until real episodes exist.

## Decision
**32 wide, 3×3.** Compute is affordable at this size, and 1×1 heads would cut the receptive field
below the ball's travel over the near horizons.

## Consequences
- `configs/base.yaml`: `model.l1_head_hidden: 32` and `model.l1_head_kernel: 3` lose `[OPEN]`.
- The receptive-field point limits the SPEC design itself. Even at 9×9 px, an L1 head can't see
  a fast ball arriving from more than 4 px away by k = 5. If the per-horizon loss in real-data
  runs grows sharply from k = 1 to k = 5, that is the first suspect (SPEC §9.6, horizon ordering,
  is the related open question).
- Re-benchmark if the encoder or batch size changes materially.
