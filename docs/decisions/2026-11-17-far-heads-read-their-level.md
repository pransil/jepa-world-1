# Horizon ordering: each far head reads its own context level

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §9.6, §4.5
- **Status:** accepted

## Question
Should the far heads (k = 10, k = 20) read the same context as the near heads, or a summary of
it? M4 builds the far heads, so it needed an answer.

## Options
- **A: their level, as §4.5 says.** k = 10 reads context L2 (B, 128, 32, 32), and k = 20 reads
  L3 (B, 256, 8, 8). This is the same context encoder output the near heads use, one level each.
- **B: their level plus the finer levels pooled to its resolution** (k = 10: L2 + pool(L1);
  k = 20: L3 + pool(L2) + pool(L1)). The far heads see fine detail directly, but there are more
  input channels and the hierarchy test is less clean.
- **C: a global summary vector** broadcast over positions. A strong bottleneck that loses
  *where* things are, which hurts predicting a spatial target map.

## Decision
**A.** Each far head reads exactly one context level, as SPEC §4.5 says. It is the simplest
option, and it keeps the claim being tested ("coarse levels predict far futures") unconfounded.

## Consequences
- `jepa_world/models/jepa.py`: a head's input is `context[head_level[k]]`, with `head_level` from
  `configs/base.yaml` (`10: L2`, `20: L3`). There is no cross-level input.
- B and C remain candidate M7 ablations. Revisit if far-head error barely beats persistence while
  the near heads do well. That would suggest L2/L3 lose information the far horizons need.
