# U-Net downsampling: stride-2 3×3 convolution

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §4.3 ("Choose one and keep it fixed")
- **Status:** accepted

## Question
Should the encoder downsample with stride-2 convolutions or with pooling? L1→L2 is 64→32 (one 2×
step). L2→L3 is 32→8, so it takes two 2× steps, with a 16×16 stage that is not exposed as a level.

## Options
- **A: stride-2 3×3 conv**, one per 2× step. Learned, so it can keep a small ball (about 4–6 px
  at 64×64) alive at 8×8 instead of averaging it away. Standard in modern conv U-Nets. Adds a few
  parameters.
- **B: 2×2 max-pool.** No parameters and cheaper. Keeps the ball's peak through 32→8, but the
  downsampling is fixed rather than learned.

## Decision
**A**, because a learned downsample can choose what survives the 4× drop to L3.

## Consequences
- `configs/base.yaml`: `model.downsample: stride2`. `jepa_world/models/unet.py` accepts only
  `stride2` and raises on anything else, so a config can't silently switch it.
- Each 2× step is Conv2d(c, c, 3, stride=2, padding=1) → GroupNorm → SiLU, followed by the
  level's two 3×3 conv blocks.
- The M7 ablations could revisit this, but SPEC asks for it to stay fixed.
