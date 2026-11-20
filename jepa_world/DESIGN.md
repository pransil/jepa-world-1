# jepa_world/ top-level evaluation: metrics, probes, baselines, the eval command (M6)

## Purpose / non-goals

`metrics.py`, `probes.py` and `baselines.py` compute SPEC §7.2's metrics and §7.3's split
discipline as **pure library code**: arrays go in, and JSON-safe numbers come out (str keys and
float, int or None values).
- `metrics.py` computes:
  - position error by horizon;
  - the ratio to persistence;
  - 50%/90% calibration of a diagonal Gaussian;
  - rollout drift;
  - uncertainty around out-of-view and occlusion events, both as event-locked curves and as a
    hidden-versus-visible split per window.
- `probes.py` fits scikit-learn linear probes on frozen embeddings.
- `baselines.py` (M2's persistence and constant velocity) also holds SPEC §7.1's
  **simulator-aware physics baseline**. It re-simulates the known terrain from the true state at
  t with `sim.episode.simulate_from_state`. That is MuJoCo on the CPU, with no GL.
- `eval.py` is the only part that touches a model, the GPU or the disk. It is described
  [below](#the-eval-command-evalpy).

**Not here:** model code. `eval.py` builds the model with `train.build_model` and only calls it.

## The eval command (`eval.py`)

```
python -m jepa_world.eval --run <run dir> [--split test] [--split dev] [--checkpoint <.pt>]
                          [--config configs/base.yaml] [--sim-params [true mean ...]]
                          [--set eval.probes.enabled=false]
```

```
run dir ─► latest checkpoints/step_*.pt (or --checkpoint, one of the run's own)
        ─► its saved config ─► train.build_model ─► strict load
eval config (--config) ─► check_eval_split(run cfg, eval cfg, split)   # leakage across configs
split ─► EpisodeSet ─► windows (ep, t): t ∈ valid_starts(T, C, max(model.horizons ∪ goal_horizons))
  ─► window_frames(k_max=0): frames t−C+1..t only ─► context_encoder (every level) ─► goal head
       goal (mean, logvar) (M, K, 2): pred_k = ball_pos[t] + mean_k,  var_k = exp(logvar_k)
       pooled context levels {L1, L2, L3}: (M, channels), for the probes
  ─► goal_head  = evaluate_positions(...) + ratio to each baseline         ┐
     baselines  = baselines.evaluate (+ simulator, eval.sim_params)        │ SplitResult(split,
     visibility = visibility(pred_k, var_k, truth_k, flags at t), per k    │   {family: ...})
     events     = event_behaviour(std series, flags), per k                │
     probes     = fit once on the run's training split, scored per split   ┘
  ─► metrics.report ─► <run dir>/eval/<UTC time>-<splits>-<hex>.json   (test = headline, dev separate)
```

- **The run's own checkpoint config builds the model and picks the data.** The eval config
  (`--config`, default base.yaml) supplies only the `eval:` block and the canonical split seeds.
  Runs started before this block existed are evaluated with base.yaml's.
- **Leakage across configs** (data/DESIGN.md asks M6 for this). `check_eval_split` refuses
  - an eval split that is the split the run trained on;
  - an eval split that shares any terrain seed with it;
  - an eval split whose seeds differ between the run's config and the eval config, because
    that would be a different test set.
- **Same windows as the baselines.** t runs over `valid_starts(T, C, k_max)`, with k_max the
  largest model or goal horizon, which is t = 3..39 for base.yaml. The baselines are recomputed
  in the same eval on exactly these windows, so every ratio is like-for-like. They are also
  `eval --baselines`'s windows whenever max(goal_horizons) ≤ max(model.horizons).
- **Only the context frames reach the model.** `model_outputs` sends frames t−C+1 … t to the
  device and runs the context encoder and the goal head, the same path `WorldModel.forward` takes
  for the goal (`test_context_path_matches_forward`). The alternative, calling `forward` on the
  full window, lost:
  - it moved 24 frames per window to MPS instead of 4;
  - it ran the target encoder on every horizon, every predictor head and the pixel decoder, none
    of which eval reads, all while holding the shared GPU lock.

  Goal outputs are moved to the CPU before being widened to float64, because MPS has none
  (`test_model_outputs_on_mps_match_cpu`).
- **Flags at t.** Visibility and events use `out_of_view` / `occluded` at the context's last frame
  t, which is what the model sees (the M7 plan's definition). `goal_head`'s in-view/out-of-view
  error split uses t+k, as the baselines do.
- **Probes.** They are fit once, on `eval.probes.train_windows` random windows of the run's
  training split, and scored on every window of each eval split.
  - Targets: world position at t, position relative to the camera at t (`pos_cam`), and the
    hidden parameters. The camera tracks the ball, so world position is barely visible in a
    frame, while `pos_cam` is what the frame shows.
  - Ball parameters come from the episode seeds (`sim.ball.sample_ball`, exact by M1's
    determinism). Each split checks the first, the last and 8 seeded episodes against their
    stored files.
  - Without a goal head (a pre-M5 run), only `probes` and `baselines` are reported, and the
    report says why. With neither a goal head nor probes, the model isn't run at all.
- **Device and lock.** Data loading, the seed checks, probe fitting, the simulator and the
  metrics all run outside `gpu_lock`. Only the forward passes run inside it, and only when
  `eval.device` is MPS. The model is moved to the device inside the lock and back to the CPU
  before it is released.
- **Simulator params.** They come from `eval.sim_params`, the accepted `["true", "mean"]`.
  `--sim-params` replaces them, and with no values it turns the simulator off.
- **Validation first.** `check_eval_block` validates the whole `eval:` block before any model,
  data or lock. That covers sim params (an unquoted YAML `true` is a bool), coverage levels,
  sizes and the probe window count. `--checkpoint` may be a bare file name in the run's
  checkpoints/. The default checkpoint is the highest step by number.
- **Output** is never overwritten. Each invocation writes one new JSON, holding the checkpoint,
  step, git and the eval config with the report. It is serialised first, written to a temp
  file, and hard-linked into place, so a failure leaves no partial file and an existing name
  is never replaced.

## The pred/actual viewer (`viewer.py`)

```
python -m jepa_world.viewer --model "24k headline=<run dir>@24000" --model "48k=<run dir>" \
                            [--split test] [--episodes 24] [--device cpu] [--out <file.html>]
```

It writes one self-contained HTML page that plays held-out episodes frame by frame, one model at
a time. **Not here:** new metrics. The page shows what the model outputs, plus the run's own
`eval/*.json` numbers when one exists for exactly that checkpoint.

```
each --model ─► load_run(run, checkpoint) on the CPU ─► check_eval_split(run cfg, base.yaml, split)
first model's cfg ─► load_split(split) (every model must name the same dataset dir)
pick_episodes: evenly spaced over the manifest, never chosen by model output
per episode: model_outputs(windows t = C−1 .. T−1, goal + pixel), context frames only
   goal  ─► goal_track: pred (W,K,2) = pos[t] + mean,  std (W,K,2) = exp(logvar / 2)
   pixel ─► (W,S,S,3) uint8 ─► one PNG strip; MAE vs real frame t+1, and copy-last-frame's MAE
episode files ─► terrain profile, cam_pos, ball_px, events (bounce, contact, out_of_view, occluded)
payload (JSON) ─► viewer_template.html ─► $JEPA_ARTIFACTS/viewers/<UTC>-<split>-<hex>.html (≤ 16 MB)
```

- **"Predicted position" is the goal head's output, and nothing else.** pred_k = ball_pos[t] +
  mean_k, with each axis's σ from logvar_k: the numbers `eval --run` scores. The 50% and 90%
  regions are the joint χ²(2) ellipses that `metrics.coverage` counts. The alternative, a linear
  probe over the latent heads' predicted embeddings, lost for two reasons. It would be a second
  model fit on training data, so its error would mix predictor and probe. And world position is
  barely decodable from these embeddings: probe R² 0.06 on test for `headline48k-s0` at 48k,
  because the camera tracks the ball. The latent heads are therefore not drawn.
- **Camera overlays use frame t's camera.** Predicted and true positions at t+k are projected
  with `sim.camera.project(·, cam_pos[t])`. The camera is orthographic, so this is exact and σ
  scales by `px_per_m`. Points off the frame are drawn on a margin around it.
- **Every t ≥ C−1 gets a prediction.** Truth exists only where t+k ≤ T−1. The windows `eval`
  scores (t = 3..39) are flagged, and per-episode errors use only those.
- **Held-out data only.** The split must be dev or test, which is checked before any checkpoint
  is read. Then `eval.check_eval_split` runs for every model against base.yaml's seeds, which
  refuses a model trained on that split. The page names the split on every panel.
- **Every model must have something to show:** a goal head, a pixel decoder, or both. A
  latent-only run is refused rather than drawn as an empty page.
- **CPU by default.** Three models on 24 test episodes took 71 s of wall time on the CPU,
  including loading the split, and the GPU is shared. So `--device mps` is opt-in and runs under
  `gpu_lock`.
- **One file, data inline.** Frames go in as PNG strips (data URIs, lossless), with no script or
  stylesheet from the network. The size is checked against 16 MB before anything is written. The
  default output is under the artifact root, and `--out` is refused inside any git work tree (a
  lane's main clone included). The write is `paths.write_new`'s: all or nothing, never over an
  existing file.

## Scores by motion regime (`regimes.py`)

```
python -m jepa_world.regimes --run <run dir> [--split test] [--checkpoint step_N.pt] [--set ...]
```

This asks whether the error lives in the bounces or in smooth rolling too. Each (window t,
horizon k) gets one regime, from the episode files' `contact` and `bounce` flags:

- **bounce:** a bounce in frames t+1..t+k.
- **rolling:** no bounce in t−C+1..t+k, and contact in every one of those frames.
- **airborne:** everything else.

**Not here:** new families in `eval --run`'s report. This is a separate file under
`<run>/eval/regimes/`. It is kept out of `eval/*.json`, because `viewer.eval_summary` and the
experiment scripts read every file there as an eval report (tested).

```
run ─► load_run, the eval --run checks (held-out split first, eval block, episodes, seeds), goal head required
load_split ─► windows (ep, t): window_index over t in [C−1, T−1−k_eval], eval's own set
episode files (eval.stored_episodes) ─► contact, bounce (N,T) ─► regime_labels per k ─► (M,) int8
baselines.predict (closed form) + eval.simulator_baselines (CPU, outside the lock) ─► {name: {k: (M,2)}}
under gpu_lock (MPS only): window_outputs = JEPA core forward + goal head per batch (no pixel decode)
   ─► goal mean/logvar (M,K,2), sqerr {k: (M,)}, spread {k: float} (float64 running moments)
score_regimes ─► {k: {all, rolling, airborne, bounce: share, error_m, mse_m2, ratio, coverage, relerr}}
```

- **The same windows and numbers as `eval --run`.** The "all" row is the eval headline for that
  checkpoint (tested: error, ratios and coverage), so a regime row and the headline differ only
  in which windows are counted. A regime's ratio is a ratio of mean errors, as SPEC §7.2's is.
  MSE is kept too, because the baseline-only cross-check is a ratio of summed squared errors.
- **Latent relerr uses the full set's spread.** A regime's relerr is its windows' mean squared
  error over the target spread of all windows at that k, so the regimes share one scale. Their
  share-weighted mean is then the full set's relerr. Dividing each regime by its own spread lost:
  a regime whose targets vary less would look worse for the same error. Training's per-batch
  relerr divides by a 32-window batch's spread instead, which is about 3% smaller (B−1)/B.
- **The latent pass is training's forward.** The JEPA core's forward computes the targets and
  the per-head means, so the per-window error is the term training averages into `sqerr/k`. The
  goal head runs on its context levels, which is `WorldModel.forward` minus the pixel decoder,
  whose output nothing here reads. Both are tested against `WorldModel.forward`, and the goal
  output equals `eval.model_outputs`'.
- **The span starts at frame t−C+1, as the definition was given.** `contact[f]` and `bounce[f]`
  describe the interval (f−1, f], so the flags at t−C+1 cover one interval before the first
  context frame. A landing in exactly that interval makes the window airborne, not rolling. The
  definition was kept as given, so the baseline-only cross-check it came with reproduces exactly.
- **Flags come from the episode files.** EpisodeSet doesn't carry `contact` or `bounce`, so they
  are read through `eval.stored_episodes`, which the simulator baseline also uses. Each file is
  checked against its row's positions. Adding the flags to EpisodeSet would touch `data/` for
  one consumer.

## Post-hoc calibration of the goal head (`calibration.py`)

```
python -m jepa_world.calibration --run <run dir> [--level 0.9] [--folds 5] [--checkpoint ...]
```

It fits one variance scale per goal horizon and reports what it does to test coverage and NLL.
**Not here:** a change to `eval --run`. Its numbers stay the model's as trained, and whether a
scale becomes the default is the user's call (dev is SPEC §3.5's debugging split).

```
run ─► load_run; the eval --run checks for dev AND test; goal head required
dev, test ─► eval_windows ─► model_outputs (goal only) under gpu_lock (MPS only) ─► goal_positions
per k: r2 = Σ_xz (y−μ)²/σ²  (M,)     s_k = quantile(r2_dev, level) / χ²₂(level)
   test before │ test × s_k (dev fit) │ test × cross-fit s_k (other terrain folds of test)
   ─► 50% / 90% joint coverage (metrics.coverage), mean NLL (goal_nll's formula), per regime too
<run>/eval/calibration/<time>-calibration-<hex>.json
```

- **The scale multiplies the variance; the mean is untouched.** So the error and every ratio stay
  as they are, and only coverage and NLL move. One scale per k: SPEC's goal head has one log-
  variance per horizon and axis, and per k is the coarsest scale that can still follow the
  miss changing with k.
- **Fit to the coverage level, in closed form.** s_k is the level-quantile of r2 over
  χ²₂(level), so the fit set's `level` region holds `level` of its truths, to one window
  (tested). The NLL-optimal scale, mean(r2)/2, is reported beside it as a diagnostic, not
  applied. When the two disagree, the residuals aren't Gaussian with a wrong scale, and no
  single scale fixes both.
- **Two fits, both scored on test.** The dev fit is the proposal. The cross-fit is the check: test
  terrain seeds go round-robin into folds, and each fold gets the scale fit on the others, so no
  window's scale saw its own terrain. If the two agree, the dev fit carries over to new terrains.
- **Regimes from `regimes.py`.** At k ≤ 2 the miss differs by regime: too wide in rolling, too
  narrow in bounces (experiments/2026-11-19-error-by-motion-regime.md). One scale can't fix that,
  and the per-regime rows show how much is left.
- **Output under `eval/calibration/`,** for the reason `regimes` writes under `eval/regimes/`
  (tested: `viewer.eval_summary` still reads the run).

## Data flow and shapes

M is the number of scored windows (episode, t), the same set at every horizon k. N is the number
of episodes, and T = 60 is the episode length. Positions are world (x, z) in metres.

```
model mean/var {k:(M,2)}, persistence {k:(M,2)}, truth {k:(M,2)} ─► evaluate_positions ─┐ "goal_head"
mean/var/truth at one k (M,2), flags {name:(M,)} ─► visibility ────────────────────────┤ "visibility"
predictive_std(var at k) (M,) ─► to_series(.., ep, t) (N,T) ─► event_behaviour(flags) ──┤ "events"
frozen features {level:(M,C,H,W)} ─► pool ─► ProbeData(split) ─► fit_probes(train)      │
                                             ProbeData(held out) ─► score_probes ───────┤ "probes"
                                                       SplitResult(split, {family: ...}) ◄┘
                          report([...]) ─► {headline: test only, development: dev/train}, families merged per split
```

| unit | in → out |
|---|---|
| `position_error` | (..., 2) ×2 → (...,) Euclidean |
| `errors_by_horizon`, `ratio_to_persistence` | `{k: (M,2)}` → `{str(k): {mean, median, p90}}` / `{str(k): float or None}` |
| `coverage` | mean, var, truth (M,2) → `{"0.5": {x, z, joint}, "0.9": {...}}` |
| `rollout_drift` | `{k: mean error}` → growth, OLS slope (m per frame), log-log exponent |
| `event_response` | series (N,T) with NaN allowed, flag (N,T) → onset and offset event-locked curves and paired deltas |
| `by_visibility` / `visibility` | mean, var, truth (M,2) and hidden (M,) or flags `{name: (M,)}` → std, error and coverage on hidden vs visible windows, `std_ratio`, `has_hidden` (per flag and for their union, "hidden") |
| `probes.fit_probes` / `score_probes` | `ProbeData` (features `{level: (M,D)}`, targets `{name: (M,) or (M,d)}`, terrain seeds (M,)) → `SplitResult` of R² and RMSE per level and target |

## Key decisions

- **Same output shapes as `baselines.evaluate`:**
  - `error_m[str(k)] = {mean, median, p90}`;
  - the ratio is a ratio of *mean* errors, and None where persistence's mean error is 0;
  - the out-of-view split is keyed `mean_error_m_by_view_at_t_plus_k`.

  The CLI can then place model and baseline numbers side by side. The alternative was a
  metrics-native layout. It lost because every consumer would need a translation step.
  `test_matches_baselines_evaluate_on_the_same_windows` pins the two to **exact** equality on
  the same windows. The cleaner end state is to have `baselines.evaluate` call these functions,
  but `baselines.py` belongs to the M2 lane, so that is left to the eval CLI follow-up.
- **Horizon keys may be int or str.** This module emits `str(k)` keys, as JSON does, and accepts
  either on input. One dict holding both `1` and `"1"` raises. M = 0 windows raises a clear
  error, not a numpy warning.
- **One window set for every horizon is enforced.** Every function that takes `{k: (M,2)}`
  raises unless all horizons share one M, and `evaluate_positions` also checks that mean, var,
  truth and persistence agree. That is M2's rule (data/DESIGN.md), and it keeps error-vs-k and
  ratio curves comparable. The windows' identity (which (ep, t)) is the caller's job, and the CLI
  must use `baselines.evaluate`'s starts.
- **Calibration reports per-axis intervals and the joint ellipse.**
  - Per axis: the central interval |y−μ| ≤ z·σ, with z = Φ⁻¹((1+p)/2). That is the literal
    "interval" of SPEC §7.2.
  - Joint: the 2-D Gaussian's p-region, Mahalanobis² ≤ −2 ln(1−p). The χ² quantile with 2 dof
    has a closed form, so no scipy is needed.

  A model can be calibrated per axis while its axes are mis-correlated, and only the joint number
  shows that. Reporting only one of the two lost because each misses a failure the other catches.
  Levels are keyed to 6 significant digits (`"0.5"`, `"0.9"`), and two levels with the same key
  raise.
- **The uncertainty scalar is the RMS σ**, √(mean of the variances), in metres. The determinant
  form (geometric mean σ) lost because one collapsed axis drives it to 0.
- **Events are event-locked averages over a series (N,T) indexed by the context's last frame t**,
  which is what the model sees. Onset is False→True at t ≥ 1, and a flag already True at t = 0 is
  not an event, because nothing came before it. The offset (reappearance) is True→False. Before
  covers offsets −W..−1, and after covers 0..W−1. `delta` is the mean over events of each event's
  own after − before, so episodes with different baseline uncertainty don't bias it. A blackout
  shorter than W puts the reappearance inside "after", and this is documented rather than
  filtered. No events (`occluded` on base datasets, which have no obstacles) gives `n_events: 0` and None everywhere, without
  numpy warnings.
- **Hidden versus visible windows (`by_visibility`, `visibility`)**, for M7's camera and
  obstacle ablations (docs/milestones/M7-plan.md):
  - `std_ratio` is the mean `predictive_std` on hidden windows divided by the mean on visible
    ones;
  - 50%/90% coverage and mean error are reported for each side;
  - "hidden" is `out_of_view | occluded`, and each flag is also reported on its own.

  The caller picks the horizon (M7 uses k = 1) and the flag's frame. The recommended frame is
  the context's last frame t, as for events. A side with no windows gets None and
  `has_hidden: False`. This is a second view, not a replacement for the event-locked curves:
  the curves show *when* uncertainty moves around an event, and the split shows *how much* it
  differs overall.
- **Probes: StandardScaler + `RidgeCV(alpha_per_target=True)`, one pipeline per (level, target).**
  Alpha is chosen by RidgeCV's leave-one-out (GCV) on the train rows only. The alternative,
  choosing alpha on held-out data, lost because that is fitting on held-out data. Plain least
  squares lost because L3 pooled is 256 dimensions, and unregularised fits on a few thousand
  correlated windows overfit.
- **Probes can't fit on test, and can't score on train terrains.**
  - `fit_probes` raises `LeakageError` for any split outside `data.splits.TRAINABLE`.
  - `score_probes` raises `LeakageError` if the held-out data shares the train split, or any
    terrain seed, with the probe's training rows.
  - "pooled" is the concatenation of every level the probes were fit on, in name order, so it
    measures what the hierarchy holds jointly. `Probes` records those levels and their widths,
    and held-out data is pooled in that layout and checked against it. Extra held-out levels are
    ignored, and a missing level or a different width raises.
  - `pool` averages in float64 without first copying the map. L1 on the test split is several
    GB as float32.
- **Split tags (SPEC §7.3).** Every result is a `SplitResult(split, {family: metrics})`, and only
  `split == "test"` is `headline`. The families are `goal_head`, `visibility`, `events` and
  `probes`.
  - `report` merges the families of each split. It puts test under `headline` and everything
    else under `development`, so a dev number can't be placed where a headline is read.
  - Two results reporting the same family for one split raise rather than overwrite.
  - The splits come from `data.splits.SPLITS`.

- **The simulator baseline's ball parameters are an argument.** The three choices are `true`
  (the episode's own, an oracle), `mean` (`sim.ball.mean_ball`, each range's midpoint) and
  `true_radius` (the visible radius with mean friction and restitution). It is opt-in:
  `eval --baselines` stays M2's quick closed-form report unless `--sim-params` is given. The
  user [decided](../docs/decisions/2026-11-17-simulator-baseline-params.md) on 2026-11-18:
  report `true` and `mean` (`PROPOSED_SIM_PARAMS`, a name from before the decision), with
  `mean` as the headline bar and `true` as the oracle ceiling.
- **A different radius at the true state** (`match_clearance`). On the test split (run
  `20261118T065246Z-…-baselines-test-590f46`, `baselines.json` → `simulator`):
  - the true ball itself starts inside the terrain at 2648 of 7400 windows, down to −3.4 cm, in
    impacts;
  - the mean-radius ball at the true centre would start inside at 2095, down to −10.7 cm.

  So "lift any penetrating start" would break the `true` oracle, and using the state as given
  launches the bigger ball. Instead the state is mapped to the other ball:
  - the centre moves along the outward surface normal until `Terrain.gap` equals the true ball's
    gap at t, by stepping p ← p + (target − gap(p))·n(p). On smooth terrain one step gives Δr·n,
    beside a face the step is horizontal, and in a floor-and-riser corner the second step clears
    the other face;
  - spin is scaled by r_true / r, so the surface speed, and with it rolling or slipping, is
    unchanged;
  - predictions are the rollout minus the shift, so the radius offset isn't counted as error.

  All three are exact no-ops for the true radius. The first version shifted only vertically,
  with a fallback of Δr when nothing matched within 8·|Δr|. That lost:
  - 307 of 7400 windows found no match (run `20261118T064214Z-…-9fa587`);
  - those were balls against ledge faces, which the fallback launched;
  - they carried about half of `simulator_mean`'s error, 0.253 m mean at k = 20, against
    0.137 m with the normal shift.

  Now |shift| is at most 0.16 m. The 54 windows flagged `unmatched` (tolerance 1e-9 m) all end
  within 4.4e-8 m (`clearance_miss_m`).
- **One rollout per window serves every horizon.** Each rollout has k_max + 1 frames, and windows
  are scored on `evaluate`'s window set. Its `extra` argument scores precomputed predictions with
  the same stats, ratio and out-of-view split as the closed-form baselines.
- **Each rollout uses its episode's own stored `meta["sim_cfg"]`**, not the eval config. Physics,
  and on M7's obstacle datasets the posts, then match the episode.
- **No GPU lock.** It is MuJoCo on the CPU with `render=False`, so no MPS and no GL, and
  `gpu_lock` is for MPS jobs. Each params option took 46–58 s on the 7400 test windows in the
  run above, and 443–655 s at load average ~265 (run `…-9fa587`).

## Seams

| unit | pure? | touches | tested by |
|---|---|---|---|
| `metrics.*` | yes, numpy + stdlib | none | `tests/test_metrics.py` (synthetic arrays with known answers) |
| `probes.*` | yes, numpy + scikit-learn (CPU) | none | `tests/test_probes.py` (synthetic embeddings) |
| `baselines.match_clearance`, `sim_ball` | yes | none | `tests/test_baselines.py` (synthetic terrains) |
| `baselines.simulator_predict` | MuJoCo physics, CPU, no GL. `simulate` is injectable | none | fast: one real short episode, with true params within 1 cm and median ≤ 1e-6 m; stubs for the policy |
| `eval.run_baselines` | no | disk (dataset, run dir) | fast with a stub rollout; slow with real physics end to end |
| `eval.goal_positions`, `check_eval_split` | yes | none | `tests/test_eval.py` |
| `eval.load_run`, `model_outputs` | torch, device | the run dir (read) | fast: tiny CPU model, checkpoint in the tmp artifact root |
| `eval.evaluate_run` / `--run` | no | GPU lock on MPS, run dir (writes `eval/*.json`) | fast end to end on tiny fake datasets (CPU, simulator off) |
| `viewer.goal_track`, `pick_episodes`, `png_strip`, `eval_summary`, `render_html` | yes (eval_summary reads a run's eval/) | none | `tests/test_viewer.py` |
| `viewer.export` / CLI | no | artifact root (reads datasets and runs, writes `viewers/`), GPU lock only with `--device mps` | fast end to end on tiny fake datasets (CPU) |
| `regimes.regime_labels`, `score_regimes` | yes, numpy | none | `tests/test_regimes.py` (hand-built flags at each edge, a brute-force check of the definition, synthetic scores) |
| `regimes.window_outputs`, `episode_flags` | torch, device / disk (episode files) | none | fast: tiny CPU model against one unbatched forward; fake episodes with chosen flags |
| `regimes.evaluate_regimes` / CLI | no | GPU lock on MPS, run dir (writes `eval/regimes/*.json`) | fast end to end on tiny fake datasets (CPU), simulator stubbed |
| `calibration.fit_scale`, `cross_fit_scales`, `terrain_folds`, `nll`, `calibrate` | yes, numpy | none | `tests/test_calibration.py` (Gaussian residuals with a known variance error; exact χ² quantiles; training's goal_nll) |
| `calibration.calibrate_run` / CLI | no | GPU lock on MPS, run dir (writes `eval/calibration/*.json`) | fast end to end on tiny fake dev and test splits (CPU), against eval --run's coverage |

## Files

| file | responsibility |
|---|---|
| `metrics.py` | errors, ratio, `coverage`, `rollout_drift`, `to_series`, `predictive_std`, `event_response` / `event_behaviour`, `by_visibility` / `visibility`, `evaluate_positions`, `SplitResult`, `report` |
| `probes.py` | `pool`, `ProbeData`, `fit_probes`, `score_probes`, `run_probes` |
| `baselines.py` | closed-form baselines (`predict`), `sim_ball`, `match_clearance`, `simulator_predict`, `evaluate` (with `extra` precomputed predictions) |
| `eval.py` | `--baselines`: every baseline on a split, written to a new run dir. `--run`: a trained run's goal head, visibility, events, probes and baselines on held-out splits, written to `<run>/eval/` |
| `viewer.py` | the pred/actual viewer: models' goal and pixel outputs on held-out episodes, as one HTML file |
| `viewer_template.html` | the page: camera view, side view, pixel decoder, controls. `viewer.py` fills in its data |
| `regimes.py` | scores by motion regime (rolling, airborne, bounce): goal head, baselines, coverage and latent relerr per regime and k, written to `<run>/eval/regimes/` |
| `calibration.py` | post-hoc variance scale per goal horizon: fit on dev and cross-fit over test terrain folds, test coverage and NLL before and after (per regime too), written to `<run>/eval/calibration/` |
