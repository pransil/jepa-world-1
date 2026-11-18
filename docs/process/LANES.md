# LANES: how parallel agents stay out of each other's way

Several Claude Code sessions work on this repo at once. Each one is a **lane**: one worktree, one
branch, one PR. These rules exist so that lanes never destroy each other's work, never corrupt each
other's data or benchmarks, and never fight over a shared file.

Who is working where is **derived, never recorded**:

```bash
tools/lane-owners.sh                    # every lane: branch, pushed?, PR state, dirty files, live session
tools/lane-owners.sh path/a.py path/b   # ...and does any other lane hold THESE files? (exit 1 if so)
tools/gpu-status.sh                     # who holds the MPS lock, which jobs are running
```

## The rules

1. **One lane = one worktree, entered with `EnterWorktree`.** `cd` does not move a session: its
   skills, subagents and reviews keep running in the main clone. The main clone (`world-1/`) is for
   `git fetch` and `git pull --ff-only` only. The `lane_guard` hook blocks `git add/commit/push`
   there and warns on edits.
2. **Push your branch the moment you create it** (`git push -u origin HEAD`). The push is your
   claim: atomic, instant, and visible to every other lane. A branch nobody can see is a lane nobody
   can avoid.
3. **Check before you touch shared files.** Run `tools/lane-owners.sh <paths>` before starting. The
   hook warns again at the moment of each edit, since a start-of-work check can go stale. If another
   lane holds a file you need, ask the user which lane takes it. Overwriting another lane's
   *uncommitted* change leaves no conflict marker and nothing to recover.
4. **Small PRs into `main`; CI must be green; the lane merges its own PR**
   (`gh pr merge --merge --delete-branch`). Rebase on `origin/main` if the PR is behind. Never
   force-push `main` (branch protection forbids it anyway).
5. **Data and runs live outside git, under `$JEPA_ARTIFACTS`** (default
   `~/devel/jepa/artifacts/world-1`), shared by every worktree:
   - datasets come only from `paths.build_dataset_once(cfg, build)`. They are hash-keyed and
     write-once, so two lanes asking for the same config get the same directory, built once.
     Bump `sim.generator_version` whenever simulator output changes;
   - runs go only to `paths.new_run_dir(name)`, which makes them unique by construction;
   - never hand-build a path into the artifact root, and never delete another lane's run.
6. **The GPU is shared. Heavy MPS jobs take the lock** (`locks.gpu_lock(label)`): training,
   benchmarks, long evals. A second job waits its turn. Benchmarks must never run without it,
   because a timing taken while another job runs is wrong. Tests run on CPU and never take it.
7. **Add files; don't share tables.** Experiments (`experiments/<date>-<slug>.md`), decisions
   (`docs/decisions/<date>-<slug>.md`) and milestone summaries (`docs/milestones/Mx.md`) are one
   file each. There is no index to update and no sequence number to collide on. Photolok lost a
   day to two lanes both taking requirement id FMA-08.
8. **SPEC.md and SPEC §9 open questions belong to the user.** A lane that needs a §9 decision asks
   and records the answer in `docs/decisions/`. A SPEC.md edit is its own small PR, made with the
   user's say-so.

## Which milestones can run in parallel

```mermaid
flowchart LR
  M1["M1 Generator<br/>sim/"] --> M2["M2 Data + baselines<br/>data/, eval baselines"]
  M3["M3 Near-horizon JEPA<br/>models/unet, jepa, heads"] --> M4["M4 Far heads<br/>models/heads"]
  M2 --> M5["M5 Goal + pixel heads<br/>models/goal, pixel"]
  M4 --> M5
  M5 --> M6["M6 Eval + probes<br/>eval.py"]
  M6 --> M7["M7 Ablations<br/>configs/, experiments/"]
```

M1 and M3 are independent: M3 is built and tested on synthetic tensors with the SPEC §4.2 shapes,
and only M3's *loss decreases on the dev set* test waits for M2. Inside a milestone, split lanes by
**file**, not by feature: two lanes in `models/heads.py` at once is a collision by construction.

## Cleaning up

A lane that has merged its PR removes its own worktree, following the order in `/builder` Step J:
1. `ExitWorktree keep`;
2. `git pull --ff-only` in the main clone;
3. `tools/finish-lane.sh wt-<slug>`.

The script removes the worktree and its local branch only when the PR merged at the branch's tip,
nothing is uncommitted or unpulled, and no live session is inside it. Otherwise it removes nothing
and says why. It is the only allowed way to delete a worktree. The user's settings deny a raw
`git worktree remove` (`tools/DESIGN.md`).

Run `tools/lane-owners.sh` to see the rest. Merged, clean trees show as `merged: removable`. A
worktree with no live session might be paused, not abandoned, so ask the user before removing
anyone else's.

**About branch protection.** `main` requires a PR and a green `fast-tests` check, with no
approvals, and admins are included. GitHub rejects a direct push of any commit that hasn't passed
`fast-tests`. One exception: pushing the head of a PR whose checks already passed is accepted,
and GitHub records it as merging that PR. That is equivalent to `gh pr merge`, but use
`gh pr merge --merge --delete-branch` so the PR record is clean.

## Handshakes

Cross-lane questions that need a durable answer go here, newest first, and get deleted once
settled. For anything quick, message the other session directly (ListAgents / SendMessage). If the
target is ambiguous, ask the user rather than guess.

- **M3 → M2: batch layout the model consumes** (2026-11-17). The JEPA takes
  `frames: (B, C + k_max, 3, H, W)`, float32 in [0, 1], where index C−1 is time t
  (C = `model.context_frames` = 4). The context is `frames[:, :C]`, and the target stack for
  horizon k is `frames[:, k:k+C]`. k_max is 5 for the near heads and 20 once M4 lands. The M2
  loader should yield exactly this, so M3's dev-set loss test can plug it into
  `train.run(cfg, run_dir, batches=...)`. Details are in `jepa_world/models/DESIGN.md`.
  Settled when M2's loader merges with a test that pins this layout.
