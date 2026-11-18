# Lane cleanup: an allowlisted script, not a looser permission rule

- **Date:** 2026-11-17   **Decided by:** user   **SPEC ref:** none (process, `docs/process/LANES.md`)
- **Status:** accepted

## Question
Agents could not finish `/builder` Step J. The user-level `~/.claude/settings.json` denies
`Bash(git worktree remove *)`, and a deny beats an allow at every settings level. So every merged
lane's worktree, and the branch deletion chained after it, was handed back to the user. How
should an agent be allowed to clean up its own lane?

## Options
- **A: Narrow the user-level deny** to the `--force` and `-f` forms. Cheap, but it loosens the rule
  for every project on the machine. Flag patterns are also easy to slip past (flag order, `-ff`).
- **B: A project script, `tools/finish-lane.sh wt-<slug>`, allowlisted in `.claude/settings.json`.**
  The checks live in versioned, tested code: merged PR at the branch tip, nothing unpulled, no
  uncommitted, untracked or non-cache ignored files, not locked, no live session inside. It calls
  `git worktree remove` without force, and `git branch -d`. Permission rules don't look inside
  scripts, so this deliberately routes around the deny for this one checked path. The raw command
  stays denied.

## Decision
B. It keeps the user-level rule intact for other projects and for raw commands, and it moves the
safety from pattern matching into tests (`tests/test_finish_lane.py`).

## Consequences
- `/builder` Step J and `docs/process/LANES.md` call the script instead of raw git.
- `tools/lane-lib.sh` now holds the live-session detection that `lane-owners.sh` used inline.
- Revisit this if the script ever needs a force option. That would be the user's call, not an
  agent's.
- Separately, the user-level allow `Bash(git branch *)` also permits `git branch -D`. That was
  pointed out to the user and is unchanged here.
