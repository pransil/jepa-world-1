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
                          [--config configs/base.yaml] [--set eval.probes.enabled=false]
```

```
run dir ─► latest checkpoints/step_*.pt ─► (its saved config, WorldModel, strict load)
eval config (--config) ─► check_eval_split(run cfg, eval cfg, split)   # leakage across configs
split ─► EpisodeSet ─► windows (ep, t): t ∈ valid_starts(T, C, max(model.horizons ∪ goal_horizons))
  ─► window_frames (frames only) ─► model ─► goal (mean, logvar) (M, K, 2) + pooled context levels
       pred_k = ball_pos[t] + mean_k,  var_k = exp(logvar_k)
  ─► goal_head  = evaluate_positions(...) + ratio to each baseline       ┐
     baselines  = baselines.evaluate (+ simulator, eval.sim_params)      │ SplitResult(split,
     visibility = visibility(pred_k, var_k, truth_k, flags at t), per k  │   {family: ...})
     events     = event_behaviour(std series, flags), per k              │
     probes     = fit on the run's training split, scored on this split  ┘
  ─► metrics.report ─► <run dir>/eval/<UTC time>-<splits>.json   (test = headline, dev separate)
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
  in the same eval on exactly these windows, so every ratio is like-for-like.
- **The model only ever gets frames.** It gets `window_frames`, i.e. frames t−C+1 … t+max(model
  horizons), because `forward` also encodes its targets. The goal head and the probes read only
  the context levels. `test_eval_outputs_ignore_frames_after_t` pins this.
- **Flags at t.** Visibility and events use `out_of_view` / `occluded` at the context's last frame
  t, which is what the model sees (the M7 plan's definition). `goal_head`'s in-view/out-of-view
  error split uses t+k, as the baselines do.
- **Probes.** They are fit on `eval.probes.train_windows` random windows of the run's training
  split and scored on every eval-split window. Ball parameters come from the episode seeds
  (`sim.ball.sample_ball`, exact by M1's determinism). Without a goal head (a pre-M5 run), only
  `probes` and `baselines` are reported, and the report says why.
- **Device and lock.** Forward passes run on `eval.device` under `gpu_lock` when that is MPS. The
  simulator and the metrics run after the lock is released.
- **Output** is never overwritten: one new JSON per invocation, holding the checkpoint, step,
  git and the eval config with the report.

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
  `eval --baselines` stays M2's quick closed-form report unless `--sim-params` is given. Which
  one is the headline bar is the user's call. The
  [proposed decision](../docs/decisions/2026-11-17-simulator-baseline-params.md) recommends
  reporting `true` and `mean` (`PROPOSED_SIM_PARAMS`), with `mean` as the bar.
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

## Files

| file | responsibility |
|---|---|
| `metrics.py` | errors, ratio, `coverage`, `rollout_drift`, `to_series`, `predictive_std`, `event_response` / `event_behaviour`, `by_visibility` / `visibility`, `evaluate_positions`, `SplitResult`, `report` |
| `probes.py` | `pool`, `ProbeData`, `fit_probes`, `score_probes`, `run_probes` |
| `baselines.py` | closed-form baselines (`predict`), `sim_ball`, `match_clearance`, `simulator_predict`, `evaluate` (with `extra` precomputed predictions) |
| `eval.py` | `--baselines`: every baseline on a split, written to a new run dir. `--run`: a trained run's goal head, visibility, events, probes and baselines on held-out splits, written to `<run>/eval/` |
