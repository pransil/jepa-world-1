# Camera projection: orthographic

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** §3.2 (projection unspecified)
- **Status:** accepted

## Question
SPEC §3.2 fixes a side-on camera looking along y, but doesn't say whether the projection is
perspective or orthographic.

## Options
- **Orthographic:** a truly 2D view. A pixel is a linear map of world (x, z), the ball is a disc
  of constant size, and only the terrain's front face is visible.
- **Perspective:** a pinhole camera. The extruded terrain's top surface shows at an angle, and the
  ground-truth-to-pixel map depends on depth.

## Decision
Orthographic. The pixel map is exact and linear, so "ground truth matches rendered geometry"
(SPEC §8 M1) is a sharp test, and no depth cue leaks into a world that is meant to be 2D.

## Consequences
- The scene uses a model camera with `projection="orthographic"`, where `fovy` is the view height
  in metres (checked on MuJoCo 4.0.1). The camera is mounted on a mocap rig that moves every frame.
  The free camera's orthographic mode isn't used, because its extent depends on distance and fovy.
- `sim.camera.view_height` (metres) sets the scale: px/m = frame_size / view_height.
