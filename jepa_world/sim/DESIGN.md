# sim/: the episode generator (M1)

## Purpose / non-goals

Turn `(sim cfg, terrain_seed, ball_seed)` into one deterministic episode: 60 RGB frames of
64×64 at 15 fps, showing a ball rolling and bouncing over a y-constant heightfield, plus exact
per-frame ground truth (SPEC §3).

**Not here:** datasets, the seed→split mapping and loaders (M2, `data/`, which calls
`simulate_episode` and `save_episode`). Obstacles and occlusion (M7; `occluded` is recorded
but always False). Physics baselines (M2/M6, which can reuse `build_model`).

## Data flow and shapes

```
cfg["sim"], terrain_seed ─► sample_terrain ─► Terrain(x (N,), heights (N,), params)
cfg["sim"], terrain_seed, ball_seed ─► sample_ball ─► BallParams(radius, friction, restitution, x0, drop, vx0)
Terrain, BallParams ─► build_model ─► MjModel (hfield, 3-joint ball, mocap camera rig)
simulate_episode: per frame, `substeps` mj_steps ─► contact/bounce flags ─► camera.track ─► render
          ─► Episode
```

| Episode field | shape, dtype | meaning |
|---|---|---|
| `frames` | (T, S, S, 3) uint8, or None | RGB. T = `episode_frames` = 60, S = `frame_size` = 64 |
| `t` | (T,) float64 | seconds, frame t at t / fps |
| `ball_pos`, `ball_vel` | (T, 2) float64 | world (x, z), (vx, vz) |
| `ball_spin` | (T,) float64 | ω about +y (rad/s); rolling to +x has ω = vx / r |
| `cam_pos` | (T, 2) float64 | camera centre (x, z); orientation is fixed |
| `ball_px` | (T, 2) float64 | projected centre (u, v), in pixels; may lie outside [0, S) |
| `contact`, `bounce`, `out_of_view`, `occluded` | (T,) bool | event flags, defined below |
| `terrain`, `ball`, `meta` | Terrain, BallParams, dict | seeds, generator_version, cfg hash, view |

**Event flags.** `contact[t]`: the ball touched the terrain at any physics substep in (t−1, t]
(at t = 0, at the initial state). `bounce[t]`: a contact onset in (t−1, t] after at least
`min_air_s` airborne, with normal approach speed ≥ `bounce_min_speed`, i.e. a landing.
`out_of_view[t]`: the projected ball disc lies entirely outside the image.

## Key decisions

- **Three joints** (slide x, slide z, hinge y) rather than a channel. Free flight is then
  exactly ballistic, and the tests check it to 1e-9.
  [decision](../../docs/decisions/2026-11-17-ball-planar-joints.md)
- **Orthographic model camera on a mocap rig.** px = S/view_height per metre, exact and linear.
  The free camera's orthographic extent scales with its distance, so it lost.
  [decision](../../docs/decisions/2026-11-17-orthographic-camera.md).
  A moved rig needs `mj_forward` before rendering: `mj_kinematics` alone leaves `cam_xpos` stale
  (checked on 4.0.1).
- **Camera:** `track_lagged` by default, with a `track_full` switch.
  [decision](../../docs/decisions/2026-11-17-camera-mode-track-lagged.md)
- **Friction = sliding μ + rolling drag** from one draw, condim 6.
  [decision](../../docs/decisions/2026-11-17-friction-sliding-plus-rolling.md)
- **Restitution is calibrated, not formula-mapped.** Soft-contact `dampratio` gave e = 0.34 for
  a target of 0.8, and it depended strongly on impact speed. Direct-mode solref
  `(-stiffness, -damping)` at stiffness 3e4 is nearly independent of speed and radius
  (0.69 / 0.65 / 0.64 at 1 / 4 / 8 m/s). A drop simulation measures e over a damping grid, and
  `contact.damping_for` inverts that monotone curve at `restitution_ref_speed`. The table is
  cached per process. What remains: e at other impact speeds differs by a few hundredths. M1.md
  reports the measured spread.
- **Contact is tracked per substep, not just at frame instants.** A 20 ms impact falls between
  two frames, so per-frame sampling would miss bounces. It also makes "no contact in (t−1, t]
  ⇒ ballistic step" an exact test.
- **Terrain = red-noise sinusoids (relief drawn, slope-capped) + 2–5 smoothed ledges + quadratic
  end walls.** Smooth hills alone never produced the out-of-view events that `track_lagged` exists
  for. At most 4% of episodes left the frame, whatever the camera settings: rolling over 1.5–3 m
  of relief is too slow to outrun a lagging camera. Ledges (1–3 m steps, sign random) give
  deep drops in one direction and walls in the other. Ledges sit in separate bins at least
  1 m apart, so they never merge into one taller cliff. The walls keep the ball on the terrain:
  max |x| over 100 episodes was 18.3 m, against 20 m. The geometry oracle is the piecewise-linear
  interpolation of the hfield samples. That is exactly MuJoCo's triangulated surface, so
  `Terrain.gap` is an exact check against MuJoCo's contacts, ledges included
  (`test_mujoco_contact_agrees_with_the_terrain_polyline`).
- **Camera tuning** (view 3.6 m, deadzone 0.6 m, lerp_z 0.04; measured in `docs/milestones/M1.md`).
  These give out-of-view in about 17% of episodes, with a median blackout of 4 frames. A 1.4 m
  deadzone lost: after a drop it parked the ball on the frame's bottom edge with no terrain
  visible. lerp_z trades event count against blackout length: 0.03 gives 21% of episodes but a
  p90 blackout of 23 frames, and 0.05 gives 12.5%.
- **Frame 0 frames the landing.** The camera starts min(drop, deadzone_z) below the ball. Centring
  on a ball that starts up to 2 m in the air made the initial drop leave the view at about
  frame 11 in a quarter of all episodes. That was an artefact of initialisation, not an event.
- **Seeds.** Terrain comes from `SeedSequence([TERRAIN_STREAM, terrain_seed])`, and the ball
  from `SeedSequence([BALL_STREAM, terrain_seed, ball_seed])`, so a terrain does not change when
  the ball seed changes.
- **Determinism scope.** Ground truth is bitwise-identical for the same seeds on any machine
  with the same MuJoCo build. Frames are byte-identical on the same machine and GL stack, but
  not guaranteed across GPUs.
- **Background is black, and shading is flat:** a strong ambient headlight, no specular, no
  shadows. Visual detail should not dominate the loss (SPEC §3.3).

## Seams

| unit | pure? | touches | tested by |
|---|---|---|---|
| `terrain`, `ball`, `camera` | pure numpy | none | fast tier |
| `contact`, `scene`, `episode(render=False)` | MuJoCo physics, CPU | none (no GL) | fast tier, also in Linux CI |
| `render`, `episode(render=True)` | creates a GL context lazily | GL (CGL on macOS) | `render` tier, Mac only |
| `io`, `preview` | | disk (`preview` writes to `paths.new_run_dir`) | fast tier (io, in a tmp dir) |

## Files

| file | responsibility |
|---|---|
| `terrain.py` | `Terrain`, `sample_terrain`: heights (waves, ledges, walls), `height_at`, `slope_at`, `gap`, `hfield` |
| `ball.py` | `BallParams`, `sample_ball`: hidden parameters and the initial state |
| `contact.py` | restitution calibration, `contact_params`: solref, solimp, friction triple |
| `scene.py` | `build_model`: MJCF, hfield data, ids |
| `camera.py` | `initial`, `track`, `project`, `out_of_view`, `px_per_m` (pure) |
| `episode.py` | `Episode`, `simulate_episode`: the stepping loop and event flags |
| `render.py` | `FrameRenderer`: RGB and segmentation (GL) |
| `io.py` | `save_episode` / `load_episode` (.npz plus JSON metadata) |
| `preview.py` | CLI: contact sheets and `stats.json` into a run dir |
