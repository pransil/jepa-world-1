# sim/: the episode generator (M1)

## Purpose / non-goals

Turn `(sim cfg, terrain_seed, ball_seed)` into one deterministic episode: 60 RGB frames of
64×64 at 15 fps, showing a ball rolling and bouncing over a y-constant heightfield, plus exact
per-frame ground truth (SPEC §3).

**Not here:** datasets, the seed→split mapping and loaders (M2, `data/`, which calls
`simulate_episode` and `save_episode`). Obstacles and occlusion (M7; `occluded` is recorded
but always False). Physics baselines themselves (M2/M6 in `baselines.py`). They re-simulate known
terrain through `simulate_from_state`, below.

## Data flow and shapes

```
cfg["sim"], terrain_seed ─► sample_terrain ─► Terrain(x (N,), heights (N,), params)
cfg["sim"], terrain_seed, ball_seed ─► sample_ball ─► BallParams(radius, friction, restitution, x0, drop, vx0)
Terrain, BallParams ─► build_model ─► MjModel (hfield, 3-joint ball, mocap camera rig)
simulate (simulate_episode = seeded wrapper): per frame, `substeps` mj_steps ─► contact/bounce
          flags ─► camera.track ─► render ─► Episode
simulate_from_state(Terrain, BallParams, pos (2,), vel (2,), spin, frames) ─► the same loop ─► Episode
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

**Event flags.**
- `contact[t]`: the ball touches the terrain in some physics state from frame t−1's instant to
  frame t's instant, both included. Those are exactly the states whose contact forces act during
  (t−1, t], so `contact[t]` False means the step from t−1 to t is pure free flight. A ball that
  lifts off exactly at frame t−1 still has `contact[t]` True. At t = 0 it is the initial touch
  state.
- `bounce[t]`: a contact onset (a landing) at a state in (t−1, t], after at least `min_air_s` of
  airborne substeps (counted as an integer), with normal approach speed ≥ `bounce_min_speed`.
  `ContactTracker` implements it, and `test_tracker_*` pins each threshold.
- `out_of_view[t]`: the projected ball disc lies entirely outside the image.

## Re-simulating from a state (`simulate_from_state`)

The simulator-aware physics baseline (SPEC §7.1, for M6;
[scope decision](../../docs/decisions/2026-11-17-m2-baseline-scope.md)) restarts known terrain from
the ball's state at frame t. `simulate_from_state(sim_cfg, terrain, ball, pos, vel, spin, frames,
cam_pos=None, render=False)` returns an `Episode` of `frames` frames. Frame 0 is the given state, and
`t` counts from it.

- **One loop.** `simulate` and `simulate_from_state` share `_run`, and both build the scene with
  `build_model(terrain, ball, sim_cfg)`. So whatever later reads `sim_cfg.get("obstacles")` there
  applies to both, and a missing key means off
  ([decision](../../docs/decisions/2026-11-17-obstacles-dataset-keying.md)). Seeded placement can
  use `terrain.params["seed"]`. A from-state Episode is a rollout, not a dataset episode: its
  top-level `terrain_seed` and `ball_seed` are None, and the terrain's seed is in
  `meta["from_state"]`, so no seed-keyed lookup can mistake one for the other. `simulate` itself is unchanged: its ground truth is bitwise equal
  to before the refactor (80 episodes, checked in this lane's PR).
- **Ball parameters are an argument**, the true `Episode.ball` or `mean_ball(sim_cfg)`, which is
  the midpoint of each sampled range (sampling is uniform). Only radius, friction and restitution
  matter here. Which one M6 uses is the user's call. With a radius other than the true one, the
  given centre can start inside the terrain. `meta["from_state"]["start_gap"]` (`Terrain.gap`, in
  metres, negative = penetrating) reports it, and the state is used as given. What to do about
  penetration is the baseline's policy, not the simulator's.
- **How exactly it reproduces an episode.** Started from frame t of an episode, with the true ball
  and the episode's `cam_pos[t]`:
  - **If MuJoCo has no contact at the start** (the result's `contact[0]` is False, which is clearly
    the case wherever `Terrain.gap` is well above 0), every later frame's ground truth is bitwise
    identical. Rendered frames are too, except within 1.5 px of the ball. The ball's
    rotation angle (`qpos[2]`) isn't in the ground truth, only its spin rate, so a restart begins
    at angle 0, and MuJoCo's tessellated sphere shades and outlines differently as it turns. From
    frame 0 the original's angle is also 0, and the frames are bitwise identical. Storing the angle
    would change the episode format, which is M2's. Baselines use ground truth, so it lost.
  - **If it has a contact**, MuJoCo's constraint solver starts from a warm start (`qacc_warmstart`,
    the previous substep's accelerations), which the saved ground truth doesn't contain. The new
    solve converges to the same contact forces only within solver tolerance, and later contacts
    can amplify the difference. Scratch measurement over 181 touching starts (80 episodes × 7 t):
    median drift 5e-10 m, p90 9e-8 m, p99 3.5e-4 m, max 3.8 mm, which is 0.07 px. The test
    allows 1 cm (0.18 px) on any frame, and a median under 1e-6 m. Bitwise equality would need
    `warmstart` off in both simulations. That changes `simulate`'s output, so it would mean a
    `generator_version` bump and a dataset rebuild. It lost.
- **Event flags near the start.** `contact[0]` is the touch state at the start instant only.
  There is no frame t−1 in the new run. `ContactTracker` starts with no airborne history, so a
  landing within `min_air_s` of the start isn't a bounce, though it may have been one in the
  original. From frame 1 (contact) and frame 3 (bounce, at 40 substeps per frame), the flags match
  the original wherever the trajectory does. `test_airborne_start_reproduces_the_episode_exactly`
  pins this.
- **Fast by default:** `render=False` skips GL. Each call builds its own model (`build_model`,
  2.3 ms), then simulates (about 9 / 28 / 38 ms for 5 / 20 / 60 frames, measured on the M4 at
  load average ~55). Reusing a built model across restarts would save about 8% at a 20-frame
  horizon. That isn't worth widening the API yet.

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
- **Restitution is calibrated, not formula-mapped.** Two things ruled out a formula. Standard
  solref `(timeconst, dampratio)` with dampratio from the textbook e(ζ) gave e = 0.34 for a target
  of 0.8, and e varied strongly with impact speed. That was a scratch measurement during design,
  not reproduced by a committed command. Direct mode `(-stiffness, -damping)` is a mass-normalised
  spring-damper. A drop simulation measures e over a damping grid, and `contact.damping_for`
  inverts the curve at `restitution_ref_speed`. The curve is quantised by the separation substep,
  and at stiff settings it can tie, so the table is its monotone envelope. It is cached per
  process. The measured spread with impact speed is in M1.md (preview `stats.json`), and
  `test_restitution_holds_in_the_real_scene_for_both_radius_extremes` holds e to ±0.03 at the
  reference speed.
- **Contact is tracked per substep, not just at frame instants.** A 20 ms impact falls between
  two frames, so per-frame sampling would miss bounces. It also makes "no contact in (t−1, t]
  ⇒ ballistic step" an exact test.
- **Terrain = red-noise sinusoids (relief drawn, slope-capped) + 2–5 smoothed ledges + quadratic
  end walls.** Smooth hills alone rarely produced the out-of-view events that `track_lagged`
  exists for. With ledges off, 3% of episodes left the frame (M1.md). Rolling over 1.5–3 m of
  relief is too slow to outrun a lagging camera. Ledges (1–3 m steps, sign random) give deep
  drops in one direction and walls in the other. Ledges sit in separate bins at least
  `ledge_gap` apart, so they never merge into one taller cliff, and a config too short for its
  ledges raises. The walls keep the ball on the terrain: `max_abs_x` in M1.md, and
  `test_ball_stays_on_the_terrain`. The geometry oracle is the piecewise-linear
  interpolation of the hfield samples. That is exactly MuJoCo's triangulated surface, so
  `Terrain.gap` is an exact check against MuJoCo's contacts, ledges included
  (`test_mujoco_contact_agrees_with_the_terrain_polyline`).
- **Camera tuning** (view 3.6 m, deadzone 0.6 m, lerp_z 0.04). The out-of-view rate and
  blackout lengths, and the lerp_z 0.03 / 0.05 alternatives, are in
  [M1.md](../../docs/milestones/M1.md), each with its preview run dir. A 1.4 m deadzone lost:
  after a drop it parked the ball on the frame's bottom edge with no terrain visible (seen in
  preview contact sheets during tuning). lerp_z trades event count against blackout length.
- **Frame 0 frames the landing.** The camera starts min(drop, deadzone_z) below the ball. Centring
  on a ball that starts up to 2 m in the air made the initial drop leave the view in about a
  quarter of episodes, nearly all at the same frame (scratch measurement during tuning). That was
  an artefact of initialisation, not an event.
- **Seeds.** Terrain comes from `SeedSequence([TERRAIN_STREAM, terrain_seed])`, and the ball
  from `SeedSequence([BALL_STREAM, terrain_seed, ball_seed])`, so a terrain does not change when
  the ball seed changes.
- **Determinism scope.** Proven, on the same machine and software stack: ground truth is
  bitwise-identical for the same seeds (`test_same_seeds_regenerate_identically`), and so are
  frames (`test_same_seeds_render_identical_frames`). Not guaranteed across machines. numpy's
  sin, exp and log can differ by an ulp between platforms and SIMD paths, and contacts amplify
  such differences. Frames also depend on the GPU and GL stack. Datasets are keyed by config
  hash only, so M2 should build each dataset on one machine and record
  `utils.repro.run_metadata()` (platform, library versions) next to its `config.json`.
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
| `ball.py` | `BallParams`, `sample_ball`: hidden parameters and the initial state; `mean_ball` |
| `contact.py` | restitution calibration, `contact_params`: solref, solimp, friction triple |
| `scene.py` | `build_model`: MJCF, hfield data, ids |
| `camera.py` | `initial`, `track`, `project`, `out_of_view`, `px_per_m` (pure) |
| `episode.py` | `Episode`, `simulate_episode` (seeded) / `simulate` (given terrain and ball) / `simulate_from_state` (given a mid-episode state), all through `_run`; `ContactTracker`, `plain_json` |
| `render.py` | `FrameRenderer`: RGB and segmentation (GL) |
| `io.py` | `save_episode` / `load_episode` (.npz plus JSON metadata) |
| `preview.py` | CLI: contact sheets and `stats.json` into a run dir |
