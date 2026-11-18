# Obstacle config keys: absent means off (no dataset re-key)

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §3.2 (`track_lagged_obstacles`), §8 M7
- **Status:** accepted

## Question
The M7 occlusion ablation needs foreground obstacles in `sim/`, which means new `sim.*` config
keys. M2's dataset key (`jepa_world/data/build.py:split_key`) hashes the
**whole** `plain_json(cfg["sim"])`. So a new key in `configs/base.yaml`, even one that is off,
would re-key every split and orphan the datasets m2 is about to build. How should the obstacle
keys be added?

The occluders are non-colliding foreground geoms between the camera and the ball's plane. So
physics and the ground-truth positions are unchanged. Only the frames and the `occluded` flag
change.

## Options
- **A: absent means off.** base.yaml documents the keys only in a commented-out `obstacles:`
  block, and comments don't reach the parsed config or the hash. `sim/` reads
  `sim_cfg.get("obstacles")` and treats a missing value as no obstacles. The obstacle experiment's
  yaml adds the block (`config.deep_merge` accepts new keys, but `--set` rejects unknown keys, so
  it must be in the yaml). Existing dataset hashes don't change, and there is no
  `generator_version` bump.
- **B: a default-off key in base.yaml, and accept a rebuild.** This follows the usual convention,
  but every split re-keys once. A rebuild of all 2240 episodes took 4 min 39 s wall in M2 (an upper
  bound, taken at a load average of about 150; docs/milestones/M2.md). Runs trained before then
  point at the old dataset dirs, including the base datasets M2 just built.
- **C: a default-off key, plus a hash exclusion.** Change `split_key` to drop the obstacle keys
  when they're off. No rebuild, but it adds a special case to m2's file.

## Decision
**A.** The datasets m2 builds stay valid, and no other lane's file changes. The cost is one
documented exception to "every key lives in base.yaml".

## Consequences
- **The later sim lane:**
  - adds a commented `# obstacles:` block to `configs/base.yaml`, with each key explained;
  - reads the keys with `sim_cfg.get("obstacles")`;
  - adds tests that:
    - the base config's `sim` section has no `obstacles` key, so `split_key` is unchanged;
    - a scene built without the key has exactly the geoms it has today;
    - `occluded` stays all False without obstacles (`test_occluded_is_all_false_without_obstacles`
      already pins this).
- **No `generator_version` bump**, because the output with the key absent is unchanged. If the
  change alters that output after all, bumping becomes necessary. That would re-key everything,
  which is option B, and it goes back to the user.
- **Obstacle placement** must come from a seed stream derived from the terrain seed, so the
  dataset's split by terrain seed still holds. The M7 plan's §5 sketches the rest (placement, the
  `occluded` rule, the target rate).
- **Timing:** the sim work starts only after M2's datasets are built and the `sim/` from-state
  entry point (needed for the M6 simulator-aware baseline) has merged, so two lanes aren't in
  `sim/episode.py` at once.
