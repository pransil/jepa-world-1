# Camera mode: track_lagged (with a track_full switch)

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §9.1 / §3.2
- **Status:** accepted

## Question
SPEC §3.2 offers three camera tracking modes. M1 (the generator) has to pick the default.

## Options
- **track_full:** the camera follows the ball in x and z, so the ball is always in frame. The only
  hidden information is off-screen terrain.
- **track_lagged:** smoothed x tracking, plus a deadzone in z. High bounces and deep drops take
  the ball out of frame, which produces `out_of_view` events.
- **track_lagged_obstacles:** as track_lagged, plus foreground occluders, which give real
  `occluded` events. This is more scope: occluder placement and an occlusion test.

## Decision
The default is `track_lagged`, the SPEC default. It produces out-of-view events, which SPEC §7.2
needs to test uncertainty. `track_full` is built as a config switch (`sim.camera_mode`), because it
is the same pure function plus a clamp. Obstacles are deferred to M7 (ablations).

## Consequences
- `jepa_world/sim/camera.py` implements both modes. Any other `camera_mode` raises.
- `occluded` is recorded every frame but is all False until obstacles exist. A test pins this.
- Revisit in M7, when the camera-mode and obstacle ablations run.
