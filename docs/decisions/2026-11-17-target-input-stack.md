# Target input: the 4-frame stack ending at t+k

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §4.4 (with §4.2)
- **Status:** accepted

## Question
SPEC §4.4 says the target encoder "encodes the frame at t + k with the same architecture". That
architecture takes a 4-frame, 12-channel stack (§4.2). What exactly does the target encoder see
at horizon k? M3 needed it fixed before the target path could be written.

## Options
- **A: the stack ending at t+k** (frames t+k−3 … t+k). Same input distribution as the context
  encoder, so the EMA weights stay in distribution, and the target carries motion at t+k. For
  k ≤ 3 the stack overlaps context frames.
- **B: frame t+k repeated ×4.** Closer to the SPEC's wording, but the encoder never sees a
  static stack in training, so the targets are out of distribution and carry no velocity.
- **C: frame t+k plus zero padding.** Same problem as B, plus an artificial "missing frames"
  signal.

## Decision
**A.** The target for horizon k is the encoder's embedding of frames t+k−3 … t+k, because only A
keeps the target encoder in the distribution its EMA weights were trained on.

## Consequences
- Batch contract (`jepa_world/models/DESIGN.md`): `frames (B, C + k_max, 3, H, W)`, with
  context = `frames[:, :C]` and target stack k = `frames[:, k:k+C]`.
- The overlap for k ≤ 3 is not leakage. The leakage rule applies to the *context*, which must
  never contain a frame at or after t+1 (and so never one at or after t+k).
  `tests/test_jepa.py::test_context_ignores_future_frames` and
  `::test_target_k_ignores_frames_after_t_plus_k` pin both directions.
- Revisit if near-horizon prediction turns out trivially easy because of the overlap (e.g. k = 1
  loss near zero while position probes stay poor).
