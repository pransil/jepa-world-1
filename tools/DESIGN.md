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
| run from the main clone | `git branch -d` judges merged-ness against HEAD; a lane can't remove another lane |
| name is `wt-<slug>`, one argument | the path and branch are derived from it, so no `..`, `/` or flags get through |
| not locked | a locked worktree is in use, or deliberately parked |
| no live Claude session has its cwd inside it | a merged lane's session may still be writing its report |
| no uncommitted, untracked, or ignored-but-not-a-cache files | `git worktree remove` silently deletes ignored files, such as a stray `*.npz` |
| no commits on the branch missing from `main` | the main clone must have pulled the merge |
| a **merged** PR whose head is exactly the branch tip | a freshly cut branch is also "0 commits ahead", so refs alone can't tell "merged" from "just started" (`lane-owners.sh` says the same). `gh` failing counts as not merged |

If the worktree directory is already gone but the branch remains, it deletes the branch after the
same branch checks. If neither exists, it exits 1, so a typo is loud.

## Seams

- `lane-lib.sh` is sourced by both scripts. It finds the main clone, and lists live sessions from
  `$CC_SOCKS_DIR` (default `/tmp/cc-socks`). It gets a session's cwd from `lsof`, or from
  `/proc/<pid>/cwd` on Linux.
- `tests/test_finish_lane.py` builds a throwaway origin, main clone and worktrees under `tmp_path`.
  It puts a fake `gh` first on `PATH`, and points `CC_SOCKS_DIR` at a temp dir holding a real
  socket named by a child process's pid. It never touches this repo's worktrees.

## Files

| file | responsibility |
|---|---|
| `lane-lib.sh` | shared helpers: main clone path, live sessions and their cwds |
| `lane-owners.sh` | lane report and collision check (read-only) |
| `finish-lane.sh` | remove one merged, clean, unowned lane |
| `gpu-status.sh` | MPS lock holder and running jobs |
