# Top-down prediction: each far head also predicts the next-finer target at its horizon

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §9.2, §8 M7
- **Status:** accepted

## Question
SPEC §9.2: "As an ablation, coarse heads also predict lower-level future targets, and errors
propagate upward. Define the exact targets before implementing." This is one of the M7
ablations ([plan](../milestones/M7-plan.md)), so its targets had to be fixed before any code lane
picks it up.

All three options share two facts:
- **The targets are free.** The target encoder already encodes each far stack `upto` L2 (k = 10)
  or L3 (k = 20) and returns every level on the way: M4's `models/DESIGN.md`, "the target encoder
  runs once per level group", on the M4 branch pending merge. So the target L1 at t+10 and the
  target L2/L1 at t+20 cost no extra encoder passes.
- **Errors propagate upward by backprop.** The new loss terms flow through the coarse head into
  the context encoder's L2/L3 stages. That is what makes the coarse levels hold what the finer
  levels will contain.

## Options
- **A: the next-finer level, from coarse features only.** The k = 20 head (reads context L3,
  8×8) gains an up×4 branch that predicts the target **L2(t+20)** (128 ch, 32×32). The k = 10 head
  (reads context L2, 32×32) gains an up×2 branch that predicts the target **L1(t+10)** (64 ch,
  64×64). No finer context is input, so the test of what L2/L3 hold stays clean. Extra cost:
  about one more 64×64 output, estimated at +10–20 ms on a 158 ms step. The basis is one L1 head
  ≈ 8–10 ms fwd+bwd, from [l1-head-width](2026-11-17-l1-head-width.md), and the M4 near+far step
  of 158 ms (`runs/20261118T050637Z-worktree-wt-m4-far-heads-bench-l1-heads-337448`).
- **B: the finer targets average-pooled to the head's own resolution.** k = 20 predicts
  pool(L2(t+20)) and pool(L1(t+20)) at 8×8, and k = 10 predicts pool(L1(t+10)) at 32×32. Only
  extra output channels, with no decoder, about +1–2% per step. It tests content, not spatial layout.
- **C: a coarse-to-fine cascade.** At each far horizon, the coarse prediction is upsampled and
  concatenated with the finer context level, and a new finer head predicts the finer target.
  Every level at that horizon is supervised. Closest to predictive coding, but the finer heads see
  fine context, so attribution to the coarse level is less clean. About +20% per step (2 extra L1
  heads and 1 L2 head).

## Decision
**A.** It is the cleanest test of the hierarchy claim, because the coarse level alone must
predict the finer future. The targets are free, and the extra compute is about 10%.

The exact specification for the implementing lane:

| Head | Reads (context) | Existing output | New top-down output | New target (detached) |
|---|---|---|---|---|
| k = 20 | L3 (B,256,8,8) | mean, logvar of L3(t+20) | mean, logvar (B,128,32,32) via up×4 | target-encoder **L2** of `frames[:, 20:24]` |
| k = 10 | L2 (B,128,32,32) | mean, logvar of L2(t+10) | mean, logvar (B,64,64,64) via up×2 | target-encoder **L1** of `frames[:, 10:14]` |

- **Loss:** `gaussian_nll` with the same reduction as the far heads (sum over channels, mean over
  batch and positions), and log-variance bounded by `model.far_logvar_bound` as for the far heads.
  `L += w_td · (NLL_td,20 + NLL_td,10)`, with **`loss.w_td = 0.5`**.
- **The branch reads coarse features only:** the far head's own trunk or the context level it
  already reads, never a finer context level. The implementing lane picks the upsampling op and
  records it in `models/DESIGN.md`.
- **Gradient path:** the new terms backprop into the head and the context encoder up to that head's
  level. The targets are detached, and the target encoder stays under `no_grad`, as for every head.
- **Logging:** every step, as `loss/td/20`, `loss/td/10`, plus `sqerr/td/<k>` and
  `logvar/td/<k>`, the same as the far heads (SPEC §5.4).
- **Schedule:** the terms join `loss/total` when the `far` stage starts
  (`train.stage_start.far`). They are logged from step 0.
- **Switch:** `model.top_down: false` by default, the names proposed for the implementing lane. With
  it off, no extra parameters are built, so the reference model is unchanged.

## Consequences
- No code now. Implementation waits until M4 (far heads) and M5 have merged. It comes in a later
  M7 code lane through `/builder`, touching `models/heads.py`, `models/jepa.py`, `train.py`
  (logging) and `configs/base.yaml` (`model.top_down`, `loss.w_td`). These are model and loss keys,
  so they don't change any dataset hash, which is keyed on `sim` and `data` only.
- Tests that lane needs, seen failing first:
  - the top-down output shapes match the L2 and L1 targets;
  - no gradient reaches the target encoder;
  - the top-down loss gives a nonzero gradient to the context L3 (k = 20) and L2 (k = 10) stage
    parameters, which is "errors propagate upward";
  - with `top_down: false`, the parameter set and loss terms are identical to the reference;
  - log-variance stays bounded.
- The ablation itself is §4 of the [M7 plan](../milestones/M7-plan.md). Revisit `w_td`, or
  consider B, if A wins on the deciding metric but hurts the far heads' own `sqerr`.
  Consider C if A shows no effect and the question becomes whether a cascade helps at all.
