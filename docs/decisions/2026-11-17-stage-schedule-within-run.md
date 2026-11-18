# Training schedule: stages join the loss at a start step, within one run

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §5.5
- **Status:** accepted

## Question
SPEC §5.5 says: "Train the near heads and the context and target encoders first. Add the far
heads. Add the goal head. Add the pixel decoder last." Is that a schedule inside one training
run, or only the order the milestones build things in?

## Options
- **A: a within-run start step per stage.** Every head in `train.stages` is built and logged
  from step 0, but its loss joins the total only from `train.stage_start.<stage>`.
- **B: everything from step 0.** `train.stages` only says which heads exist, and §5.5 describes
  the milestone order (M3 near, M4 far, M5 goal/pixel). One less hyperparameter.
- **C: the mechanism of A, with every start step defaulting to 0.**

## Decision
**A**, because it follows §5.5 inside every run. `train.stage_start: {near: 0, far: 2000}` in
`configs/base.yaml`. The 2000 is a placeholder, to be tuned on real data.

## Consequences
- A head whose stage hasn't started is still run, and its terms are logged every step (SPEC
  §5.4), but they are left out of `loss/total`. So it gets no gradient, and AdamW leaves it alone,
  weight decay included.
- The variance guard covers only levels read by *active* heads. Before `far` starts, the
  context L2/L3 stages get no gradient at all.
- `train.check_config` requires a start step for every stage in `train.stages`, and at least one
  stage starting at 0.
- An experiment that wants B sets `train.stage_start.far: 0`.
- M5 adds `goal` and `pixel` entries.
