# Latent distance: squared L2 on raw embeddings

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §9.5, §5.1
- **Status:** accepted

## Question
Which distance should the latent loss L_k use? M3 implements it for the near heads. The far
heads switch to a Gaussian NLL in M4.

## Options
- **A: squared L2, raw.** Exactly the SPEC §5.1 formula: the sum over channels of
  (pred − sg(target))², averaged over positions and batch. Pairs naturally with the far heads'
  Gaussian NLL (same squared-error family).
- **B: squared L2 on per-position LayerNorm'd targets** (as in I-JEPA). Bounds the target scale
  and makes collapse harder, at the cost of another design choice.
- **C: cosine distance** per position. Scale-invariant, but ignores magnitude and doesn't pair
  with the Gaussian NLL.

## Decision
**A**, because it is the SPEC formula and is consistent with the far heads' NLL. The collapse
risk it leaves is handled by the variance guard
([2026-11-17-collapse-guard-on.md](2026-11-17-collapse-guard-on.md)).

## Consequences
- `configs/base.yaml`: `loss.latent_distance: mse` keeps its name and loses `[OPEN]`. "mse" here
  means *summed* over channels, not averaged: L_k = mean_{b,h,w} Σ_c (pred − tgt)². Any other
  value raises.
- `jepa_world/models/jepa.py::latent_loss`, tested by
  `tests/test_jepa.py::test_latent_loss_matches_hand_value`.
- Revisit if L_k scales badly across levels in M4 (L1 has 64 channels, L3 has 256), or if
  embeddings drift in norm without collapsing.
