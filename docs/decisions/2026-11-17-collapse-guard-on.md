# Collapse guard: VICReg variance term, on by default

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §9.3, §5.1
- **Status:** accepted

## Question
Should the VICReg-style variance term on the context embeddings be part of the default training
loss, or added only once a run shows collapse?

## Options
- **A: on by default.** One term, logged every step, switchable off for the ablation.
- **B: off by default, on in tests.** Closer to a vanilla JEPA, which relies on EMA and predictor
  asymmetry alone. Turned on only once collapse shows (embedding std → 0 in `metrics.jsonl`).

## Decision
**A**, because with raw squared-L2 targets (§9.5) a collapsed embedding drives the loss to zero,
and a guard that is always on and logged costs little.

Definition: for each context level read by an active head (L1 in M3),

    loss/var = mean over (c, h, w) of relu(γ − sqrt(Var_batch[z_{c,h,w}] + ε))

That is, the batch standard deviation of each feature *at each position*, which follows SPEC's
"batch standard deviation" literally. Pooling over positions would let a map that varies in
space but ignores the input pass. γ = `loss.var_gamma` = 1.0, ε = `loss.var_eps` = 1e-4,
weight `loss.w_var` = 1.0.

## Consequences
- `configs/base.yaml`: `loss.collapse_guard: true` loses `[OPEN]`. New keys `var_gamma`, `var_eps`
  and `w_var`.
- The loss dict also carries `emb_std/<level>` (mean batch std, detached), so collapse is
  visible whether or not the guard is on.
- Tested by `tests/test_jepa.py::test_variance_guard_*`.
- Revisit γ and w_var once real-data runs show the typical embedding scale (GroupNorm + SiLU
  outputs are O(1), so γ = 1 may keep the term always active).
