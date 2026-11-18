# tools/: lane tooling

Shell tools for running several Claude sessions as worktree lanes (CLAUDE.md "The floor",
`docs/process/LANES.md`). They are bash 3.2 compatible, because macOS's `/bin/bash` is 3.2.

## Purpose / non-goals

- `lane-owners.sh` **reports** who works where and checks paths for collisions. It changes nothing.
- `finish-lane.sh wt-<slug>` **removes** one merged lane: its worktree and its local branch. It
  is the only path by which an agent deletes a worktree.
- Non-goals: it never removes a lane that might still hold work, never passes `--force`, and never
  touches remote branches. `gh pr merge --delete-branch` already deletes the remote branch.

## Why a script, and not a permission rule

The user's `~/.claude/settings.json` denies `Bash(git worktree remove *)`, and a deny beats an
allow at every level. Narrowing that rule to the `--force` forms would loosen it for every
project, and flag patterns are easy to slip past (flag order, `-ff`). Instead, the project allows
exactly `Bash(tools/finish-lane.sh *)`. The safety checks live in versioned, tested code, and the
script has no way to pass `--force`. Decision:
[2026-11-17-lane-cleanup-script](../docs/decisions/2026-11-17-lane-cleanup-script.md).

## finish-lane.sh: what must hold before anything is deleted

Every check runs first, and any failure exits 1 with nothing removed. Only then does it run
`git worktree remove` (no force: git itself still refuses a dirty tree), then `git branch -d`.

| check | why |
|---|---|
| run from the main clone | `git branch -d` judges against the main clone's HEAD, and a lane must not remove another lane |
| name is `wt-<slug>`, one argument | the path and branch are derived from it, so no `..`, `/` or flags get through |
| the worktree has `worktree-<name>` checked out | otherwise the branch checks would vouch for a different branch than the tree holds |
| not locked | a locked worktree is in use, or deliberately parked |
| no uncommitted, untracked, or ignored-but-not-disposable files | `git worktree remove` silently deletes ignored files, such as a stray `*.npz`. Disposable means caches (`.venv/`, `__pycache__/`, ...) and the lane's own `.claude/settings.local.json`. `--untracked-files` is explicit, so `status.showUntrackedFiles=no` can't hide anything |
| no commits on the branch missing from `main` | the main clone must have pulled the merge |
| `git branch -d` will accept the branch | it judges against the branch's upstream if that ref exists, else HEAD. The worktree is removed first, so this is checked up front |
| a **merged** PR whose head is exactly the branch tip | a freshly cut branch is also "0 commits ahead", so refs alone can't tell "merged" from "just started" (`lane-owners.sh` says the same). A reused lane name with an old merged PR doesn't match. `gh` failing counts as not merged |
| no process of this user has its cwd inside the worktree, except `caffeinate` | a Claude session or a background job (training, a `run_in_background` shell) still running there. On macOS this is the slow check (one `lsof`, ~0.3 s), so it runs only when everything else passed. Files held open from a cwd elsewhere aren't seen. `caffeinate` is exempt: Claude Code starts `caffeinate -i -t 300` in the session's cwd to keep the Mac awake, so one is usually still inside a worktree right after `ExitWorktree`, and it touches no files |

If the worktree directory is already gone but the branch remains, it deletes the branch after the
same branch checks. If neither exists, it exits 1, so a typo is loud.

## Seams

- `lane-lib.sh` is sourced by both scripts. It finds the main clone. It lists every process's cwd
  and command name for `finish-lane.sh`: from `/proc` plus `ps` where `/proc` exists (Linux, so CI
  covers that path), and from `lsof` on macOS. It lists live Claude sessions
  from `$CC_SOCKS_DIR` (default `/tmp/cc-socks`), for `lane-owners.sh`.
- `tests/test_finish_lane.py` builds one throwaway main clone, with a lane per scenario, under
  `tmp_path` (no origin). It puts a fake `gh` first on `PATH`, and a "running process" is a real
  `sleep` child. The exempt one is the real `caffeinate` on macOS (a copied Apple binary is
  SIGKILLed on launch), or a copy of `sleep` named `caffeinate` on Linux. It never touches this repo's worktrees. The script reports every problem from the
  fast checks through one gate, so one run on a lane with several faults checks each detector by
  its message. That keeps the file at about 3 s in the fast tier.

## Files

| file | responsibility |
|---|---|
| `lane-lib.sh` | shared helpers: main clone path, process cwds, live Claude sessions |
| `lane-owners.sh` | lane report and collision check (read-only). `merged: candidate` is a hint, and `finish-lane.sh` decides |
| `finish-lane.sh` | remove one merged, clean, unowned lane |
| `gpu-status.sh` | MPS lock holder and running jobs |
