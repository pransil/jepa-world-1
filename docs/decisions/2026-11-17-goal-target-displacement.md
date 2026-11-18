# Goal head target: displacement from time t, in world metres

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §4.6, §5.2, §7.2
- **Status:** accepted

## Question

SPEC §4.6 says the goal head outputs a distribution over "the ball's world (x, z) at horizon k".
Under `track_lagged`, the camera follows the ball, and terrain is random per seed. So the frames
show where the ball is relative to the camera, and how it moves. They do not show the ball's
absolute world x. We have to choose what the head predicts before the goal head can be built (M5).

## Options

- **A: displacement** `ball_pos[t+k] − ball_pos[t]`, world metres (x, z). It can be inferred from
  motion and the visible terrain. Persistence predicts 0. Position error in world units is the same
  as for an absolute prediction once the true pos(t) is added back. The persistence baseline adds
  back that same pos(t), so the ratio to persistence (§7.2) compares like with like.
- **B: camera-relative** `ball_pos[t+k] − cam_pos[t]`. It is observable. The model must also find
  where the ball is now, so its error includes a localisation error that the persistence baseline
  (which knows pos(t)) does not have.
- **C: absolute world** `ball_pos[t+k]` (SPEC literal). The x part can't be observed, so the head
  would learn the dataset mean with a huge variance.

## Decision

A, displacement in world metres. It is observable, and it makes the persistence comparison exact.

## Consequences

- `jepa_world/models/goal.py`: `displacement_targets(ball_pos, context_frames, horizons)` builds the
  (B, K, 2) target from a per-frame `ball_pos` aligned with `frames`. The head itself is
  frame-agnostic: it outputs a 2-D Gaussian.
- M2 must carry `ball_pos: (B, C + k_max, 2)`, world metres, on the batch's frame axis (index
  C−1 = t). This was sent to the orchestrator on 2026-11-17. M2 stores it as float64, and
  `displacement_targets` subtracts at that precision and returns float32.
- SPEC.md is not edited. This file records how §4.6 is read.
- M6: position error for horizon k is |Δ̂ − Δ|, and persistence error is |Δ|.
- Revisit if the goal head becomes a planning target that needs absolute positions (then predict
  pos(t) + Δ̂ using known state), or if the camera stops tracking the ball.
