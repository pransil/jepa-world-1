# CLAUDE.md

Guidance for Claude Code sessions in this repo. **Several sessions work here at once.** Read "The
floor" before editing anything.

## What this is

A research testbed for a JEPA-style world model with multi-horizon prediction, on MuJoCo episodes
of a ball rolling over terrain z = h(x), constant along y for now (later z = h(x, y)). Episodes are
64×64 RGB, 15 fps, 60 frames. **`SPEC.md` is the design and milestone plan (M1–M7).**
`README.md` has the architecture diagrams. PyTorch on Apple MPS (M4, 36 GB); tests run on CPU.

## The floor: every lane, every change

1. **Work in a worktree, entered with `EnterWorktree`** (name `wt-<topic>`). Never `cd` into one,
   because skills and subagents would keep running in the main clone. The main clone (`world-1/`)
   is for `git fetch` and `git pull --ff-only` only. The `lane_guard` hook **blocks** `git add/commit/push` there.
2. **Push your branch as soon as you create it** (`git push -u origin HEAD`). That's your claim.
   Run `tools/lane-owners.sh <paths>` before starting, and heed the hook's COLLISION warnings.
3. **Tests first, through `./test`.** Every new module arrives with tests that were seen to fail
   first. Never run bare `pytest` as the definition of passing.
4. **PR into `main`, green CI, then self-merge** (`gh pr merge --merge --delete-branch`). `main`
   is protected: no direct pushes.
5. **SPEC §9 `[OPEN]` items and SPEC.md itself are the user's call** (SPEC §11). Ask, then record
   the answer in `docs/decisions/`.

The full rules are in `docs/process/LANES.md`. **For anything beyond a one-line fix, use
`/builder`.** For training runs and comparisons, use `/experiment`. Both call `/verify`.

## Commands

```bash
uv sync                 # create/refresh .venv (per worktree; fast, uses uv's shared cache)
./test                  # lint + fast tier (CPU, < ~20s). This is what CI runs.
./test all              # + slow, render, mps tiers (auto-skip what this machine lacks)
./test -k ema           # fast tier, filtered (any pytest args)
./test fix              # ruff format + autofix
tools/lane-owners.sh [paths]   # who is working where; collision check for paths
tools/gpu-status.sh     # MPS lock holder and running jobs
tools/finish-lane.sh wt-<slug>   # after the merge, from the main clone: remove a merged, clean lane
uv run python -m jepa_world.train --config configs/x.yaml [--set key=value]   # (from M3)
```

Use `uv add <pkg>` to add a dependency (it updates `pyproject.toml` and `uv.lock`; commit both). On
Linux, torch comes from the PyTorch CPU index (see `pyproject.toml`), so CI doesn't pull CUDA.

## Layout

```
jepa_world/
  sim/        M1  MuJoCo scene, heightfield terrain, ball, camera, rendering, ground truth, events
  data/       M2  dataset, terrain-seed splits, leakage checks
  models/     M3–M5  unet.py, jepa.py (EMA + latent loss), heads.py, goal.py, pixel.py
  train.py    M3+  eval.py  M2/M6
  utils/      paths (artifact root, write-once datasets, run dirs) · locks (GPU lock) ·
              config (YAML + base inheritance + --set overrides) · repro (seeds, run metadata) · device
configs/      base.yaml = SPEC defaults; one yaml per experiment, inheriting from it
tests/        conftest.py defines the tiers; test_<module>.py per module
experiments/  one note per experiment       docs/decisions/  one file per SPEC §9 decision
docs/milestones/  M1..M7 summaries          docs/process/LANES.md  parallel-work rules
tools/        lane-owners.sh, finish-lane.sh, gpu-status.sh (tools/DESIGN.md)
```

The code lives under one package, `jepa_world/`, instead of SPEC §10's top-level `sim/ data/
models/`, because top-level packages named `data` and `models` collide with other installs.
Everything else follows SPEC §10.

## Testing: what matters, fast

- **Fast tier (default):** CPU, tiny shapes (B=2, few channels, small spatial size where the code
  allows), fixed seeds, under 1s each. `conftest.py` lists any fast test over 2s at the end of the
  run, so shrink it or mark it `slow`.
- **Markers:** `slow` (real training loops, full episodes), `render` (MuJoCo GL), `mps`. CI runs
  only the fast tier on Linux CPU, and the others run locally on the Mac.
- **Test invariants, not golden numbers:** shapes, no-grad-to-target, the EMA formula, leakage,
  split disjointness, determinism, finite NLL, positive variance, all loss terms logged.
- Tests never touch the shared artifact root: an autouse fixture points `JEPA_ARTIFACTS` at a tmp
  dir.

## Shared resources (multi-agent)

- **Artifacts** live outside git, under `$JEPA_ARTIFACTS` (default `~/devel/jepa/artifacts/world-1`):
  `datasets/` (hash-keyed, write-once; build only with `paths.build_dataset_once`), `runs/` (unique
  per run; create only with `paths.new_run_dir`), and `locks/`. Bump `sim.generator_version` when
  simulator output changes.
- **GPU:** one MPS device shared by every lane. Training, benchmarks and long evals take
  `locks.gpu_lock(label)`, and a second job waits. The lock is a kernel `flock`, so it is released
  when its process dies and can't go stale.

## Working rules

- **Check library APIs against the installed version, never memory** (SPEC §2, §11). The
  installed versions are recorded in `uv.lock`. MuJoCo is 4.x, and many online examples are for
  2.x/3.x. Use `uv run python -c "import mujoco; help(...)"`, or read `.venv/lib/python3.12/site-packages/`.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** is set by `import jepa_world`, so import it before torch in
  entry points. When profiling, log which ops fall back to CPU.
- **fp32 by default.** Don't assume bf16 works on MPS. Test any lower-precision path explicitly.
- **A claim needs an artifact.** "Works", "converges" and "beats persistence" cite a test, a
  `./test` RESULT line, or a run dir's `metrics.jsonl`. Code reading shows what *can* happen,
  and only an artifact shows what *did*. "Never happens" needs a scoped search, reported with its
  scope.
- **One file per experiment, decision and milestone summary.** No shared index tables, no
  sequence numbers: use `<YYYY-MM-DD>-<slug>.md`.
- **Unsure which lane owns something, or which session to message?** Ask the user. Don't guess.
