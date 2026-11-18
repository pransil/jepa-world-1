# Test split balls: new draws from the same ranges

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §3.5 ("Test: 50 held-out terrain seeds, with a separate set of ball parameters")
- **Status:** accepted

## Question
What does "a separate set of ball parameters" mean for the test split: independent draws from
the training distribution, or parameter ranges that training never sees?

## Options
- **A: same ranges, new draws.** Test ball seeds (1000–1003) are disjoint from train's (0–3).
  Ball parameters come from `SeedSequence([BALL_STREAM, terrain_seed, ball_seed])`, so they are
  already independent across terrains. Disjoint ball seeds make the separation explicit, and
  `check_split_config` enforces it. The test stays in-distribution for the ball, and out of
  distribution only for the terrain.
- **B: held-out parameter ranges.** Test radius, friction and restitution come from intervals that
  training never sees. That is an out-of-distribution test. It needs per-split `sim.ball` ranges,
  and so a different sim config hash for the test split.

## Decision
A. The v1 question is generalisation to unseen terrain. Held-out ball ranges would confound it
with extrapolation in the hidden parameters.

## Consequences
`data.splits.test.ball_seeds: [1000, 4]`. `splits.check_split_config` raises `LeakageError` if
train and test share a ball seed (`test_test_ball_seeds_are_disjoint_from_train`). B remains a
possible M7 ablation: add per-split ball ranges, and a test split keyed on them.
