# "Skip connections on/off" means the U-Net decoder skips

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §8 M7, §4.3, §4.7
- **Status:** accepted

## Question
SPEC §8 M7 lists "Skip connections on and off". In SPEC §4 the only U-Net skips run from the
encoder to the decoder, and "the decoder path is used only by the pixel decoder" (§4.3). The
pixel decoder "reads L1 plus the skip from the U-Net decoder path" (§4.7). m5 is building that
decoder now, so it needs to know what the switch turns off.

## Options
- **A: the decoder skips only.** A switch turns off the encoder L1/L2 → decoder skips feeding the
  pixel decoder. With them off, no encoder feature other than L3 reaches the pixel decoder, so
  pixel reconstruction must pass through L3. It's a small switch inside m5's decoder.
- **B: encoder top-down laterals (FPN style).** The context levels L1/L2 also receive upsampled
  coarser features before the heads read them, and "off" is the plain SPEC §4.3 encoder. This
  changes what every head sees, so it is a larger architecture change that needs its own build lane.

## Decision
**A.** It is the meaning SPEC's architecture supports, and it costs m5 one switch.

## Consequences
- **m5 (the `models/unet.py` decoder path and `models/pixel.py`):** please expose
  `model.unet_skips: true` (the proposed name). It's on by default, as SPEC §4.7 describes. With
  `false`, the decoder upsamples from L3 alone and gets no L2 or L1 input. If m5 merges without it,
  a later M7 lane adds it through `/builder`.
- The switch only acts through the pixel loss, so it is a **no-op at `w_pix = 0`**. The ablation
  runs at `w_pix` 0.05 (the base) and 0.5. See §2 of the [M7 plan](../milestones/M7-plan.md).
  (2026-11-19: base.yaml's default is now a 10% pixel share, not `w_pix` 0.05, so an arm that
  means "the base" must say which: [pixel-share-default](2026-11-19-pixel-share-default.md).)
- Deciding metric: the L3 linear-probe ball-position error on the test split (M6 `probes.py`).
- B stays available as a separate future ablation if the hierarchy results call for it.
