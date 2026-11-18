# data/: splits, dataset build, loader, leakage checks (M2)

## Purpose / non-goals

Turn M1's episodes into the three terrain-seed splits of SPEC §3.5. Build each split once into the
shared artifact root, and feed the JEPA batches in the layout M3 consumes. Guarantee two things:
no frame at or after t+k reaches the context, and no terrain crosses a split. The position
baselines that go with it are in `jepa_world/baselines.py` (SPEC §7.1), and their CLI is
`python -m jepa_world.eval --baselines`.

**Not here:**
- The episode format and the simulator. Those are `sim/` (M1), used through `simulate_episode`,
  `save_episode` and `load_episode`, and never edited from this lane.
- Model metrics, probes and calibration (M6).
- The simulator-aware baseline (M6). It needs a `sim/` entry point that starts from a mid-episode
  state ([decision](../../docs/decisions/2026-11-17-m2-baseline-scope.md)).
- `synthetic.py`, M3's moving-disc stand-in, kept for benchmarks and smoke runs.

## Data flow and shapes

```
cfg.data.splits ─► splits.split_episodes ─► [(terrain_seed, ball_seed), ...]
   └► build.build_split ─► simulate_episode ─► save_episode
        ─► datasets/episodes-<split>-<hash>/{t*-b*.npz, manifest.json, meta.json, config.json}
           └► dataset.load_split ─► EpisodeSet ─► dataset.batches ─► (B, C+k_max, 3, S, S) float32
```

| object | shape, dtype | notes |
|---|---|---|
| episode file | M1's `.npz` (`sim/io.py`) | frames (T, S, S, 3) uint8, plus the GT fields |
| `EpisodeSet.frames` | (N, T, 3, S, S) uint8, torch, CPU | the whole split in RAM. Train: 2000·60·12288 B ≈ 1.5 GB |
| `EpisodeSet.ball_pos`, `.ball_vel`, `.cam_pos` | (N, T, 2) float64 | world (x, z), (vx, vz), camera centre |
| `EpisodeSet.out_of_view`, `.occluded` | (N, T) bool | `occluded` is all False until M7's obstacles |
| `EpisodeSet.terrain_seed`, `.ball_seed` | (N,) int64 | checked against the manifest and each file's meta |
| `valid_starts(T, C, k_max)` | (n_t,) int | t ∈ [C−1, T−1−k_max]. T=60, C=4, k_max=20 gives 3..39 |
| batch | (B, C+k_max, 3, S, S) float32 in [0, 1] | index C−1 is time t. Context `[:, :C]`, target stack k `[:, k:k+C]` |
| batch, `with_gt=True` | dict: `frames` as above, plus `GT_KEYS`: `ball_pos`, `ball_vel`, `cam_pos` (B, C+k_max, 2) float32; `out_of_view`, `occluded` (B, C+k_max) bool | the same window and time axis as `frames`. For M5's `goal.displacement_targets` and M6's visibility metrics (flags at index C−1) |

## Key decisions

- **Split by terrain seed** (SPEC §3.5): dev is terrains 0–9, train 1000–1499 and test
  100000–100049. Each terrain gets 4 balls
  ([decision](../../docs/decisions/2026-11-17-ball-samples-per-terrain.md)). Train and test
  terrains sit outside 0–199, where M1 tuned its event rates (M1.md). Dev is for debugging only,
  so the overlap there doesn't matter.
- **Test balls are new draws from the same ranges.** Their ball seeds are 1000–1003, disjoint
  from train's 0–3 ([decision](../../docs/decisions/2026-11-17-test-split-ball-params.md)).
  Held-out parameter ranges, an out-of-distribution test, lost for v1 and are a possible M7
  ablation.
- **Leakage is checked at three levels:**
  - the config: `check_split_config` refuses overlapping terrain seeds, or ball seeds shared by
    train and test;
  - the disk: `load_split` checks every file's stored seeds against its manifest, and each
    manifest's terrain and ball seeds against its split. It also checks terrain seeds, terrain
    hashes, and train/test ball seeds across every built split's manifest, so the same terrain
    under two seeds would still be caught;
  - the batch: one shared index check (`_window_index`, used by `window_frames`, `window_gt` and
    `batches`) refuses any frame index outside [0, T) and any episode index outside [0, N). A
    negative index would otherwise wrap silently: to the end of the episode, putting future
    frames in the context, or to another episode.
- **Training never reads test.** `train.check_config` refuses `data.split: test` with
  `LeakageError`, and only `dev` and `train` are trainable (`splits.TRAINABLE`).
- **Not checked here: leakage across configs.** The on-disk check sees only the splits of the
  config it is given. A model trained under another config's splits (other terrain seeds, say)
  could overlap this config's test split unseen. M6's model eval must compare the run's
  `config.yaml` splits against the eval config before reporting anything.
- **One window set for every horizon.** Baselines are scored on
  t ∈ [C−1, T−1−max(model.horizons)] (T=60, C=4: t = 3..39), the same windows at every k. That
  keeps error-versus-k curves and persistence ratios comparable across k. **M6 must score the
  model on these same windows**, even for a near-only model, whose training windows run to
  t = 54.
- **One dataset per split**, keyed on (sim cfg, the split's seed list, `format_version`). The
  alternative was one dataset holding all three splits. That lost because changing train's size
  would then rebuild test, and a dev-only build would cost the whole set.
- **Whole split in RAM as uint8.** The alternative was reading npz per batch. That lost because
  each sample would decompress a whole 60-frame file to use 24 frames of it. 1.5 GB fits easily in
  36 GB. Each batch is one advanced-indexing gather plus a cast.
- **Ground truth is opt-in** (`batches(..., with_gt=True)`). The default still yields the bare
  frames tensor, so `train.run` and its callers are unchanged. The alternative, always yielding
  a dict, lost because it would have changed every consumer at once, including M4's lane. Both
  modes draw the same windows from the same seed. GT positions are cast to float32 because MPS
  has no float64. At |x| ≤ 20 m that rounds a position by under 2e-6 m, and
  `test_m5_and_m6_consumers_accept_gt_batches` holds displacements to 1e-5 m.
- **Sampling is uniform over (episode, valid t) windows**, with replacement, from a numpy
  Generator seeded by `cfg.seed`. Every episode has the same number of windows, so this is also
  uniform over episodes.
- **The build takes the GPU lock**, because rendering uses the GPU through CGL, and a build takes
  minutes and would skew a benchmark running beside it. The lock is skipped when the dataset
  already exists.
- **Build record:** `meta.json` holds `repro.run_metadata()` plus the timing.
  Cross-machine determinism isn't guaranteed (sim/DESIGN.md, "Determinism scope"), so the dataset
  says which machine and stack produced it.

## Seams

| unit | pure? | touches | tested by |
|---|---|---|---|
| `splits` | pure | none | fast |
| `dataset.valid_starts`, `window_frames`, `batches` | pure (torch CPU) | none | fast |
| `build.build_split` | no | disk (artifact root), and GL through the default `simulate` | fast, with `simulate` injected (encoded fake frames). One `render` test uses the real renderer |
| `dataset.load_dir`, `load_split` | no | disk | fast, on datasets built in the tmp artifact root |
| `build.main` | no | GPU lock, GL | by hand (M2.md) |

## Files

| file | responsibility |
|---|---|
| `splits.py` | seed lists per split, `LeakageError`, `check_disjoint`, `check_split_config` |
| `build.py` | `split_key`, `build_split` (write-once through `paths.build_dataset_once`), `terrain_sha`, CLI |
| `dataset.py` | `EpisodeSet`, `load_dir`, `load_split`, `valid_starts`, `window_frames`, `window_gt`, `batches` |
| `synthetic.py` | M3's moving-disc stand-in (smoke runs and benchmarks only) |
