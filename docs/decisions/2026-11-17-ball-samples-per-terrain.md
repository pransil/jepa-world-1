# Ball samples per terrain: 4

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §3.5 ("each with several ball-parameter samples")
- **Status:** accepted

## Question
SPEC §3.5 asks for "several" ball-parameter samples per training terrain, and `base.yaml` left it
`null`, "TBD in M2". M2's dataset build needs a number.

## Options
- **2 per terrain:** train is 1000 episodes. The fastest build, but it gives fewer hidden-parameter
  samples for the M6 probes.
- **4 per terrain:** train is 2000 episodes, about 66 MB on disk, about 1.5 GB of uint8 frames in
  RAM, and about 2–3 min to render. M1's size estimates (docs/milestones/M1.md) assumed this.
- **8 per terrain:** train is 4000 episodes, about 130 MB on disk, about 3 GB in RAM, and about
  5 min to build.

## Decision
4 per terrain, in every split: dev 40, train 2000 and test 200 episodes. It is the middle option,
already sized by M1, and it fits comfortably in RAM.

## Consequences
`data.splits.<split>.ball_seeds: [first, 4]` in `configs/base.yaml`. Changing the count changes
the split's dataset hash, so a new dataset is built and the old one stays readable. Revisit if
M6's hidden-parameter probes are limited by the number of samples.
