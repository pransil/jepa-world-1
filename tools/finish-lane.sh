#!/usr/bin/env bash
# Remove one merged lane: its worktree and its local branch. Nothing else, and never with --force.
#
#   tools/finish-lane.sh wt-<slug>
#
# Run it from the main clone after the lane's PR merged, `ExitWorktree keep`, and
# `git pull --ff-only` (/builder Step J). It removes .claude/worktrees/wt-<slug> and the branch
# worktree-wt-<slug>. Every check runs before anything is deleted, and any failure exits 1 with
# nothing removed. The checks and their reasons are in tools/DESIGN.md.
#
# Exit: 0 removed · 1 refused, nothing removed · 2 usage
set -uo pipefail

usage() {
  echo "usage: tools/finish-lane.sh wt-<slug>   (one lane name, run from the main clone)" >&2
  exit 2
}
[ $# -eq 1 ] || usage
NAME=$1
case "$NAME" in wt-?*) ;; *) usage ;; esac
case "$NAME" in *[!A-Za-z0-9._-]* | *..*) usage ;; esac

. "$(dirname "$0")/lane-lib.sh" || exit 2

die() {
  echo "finish-lane: REFUSED: $*" >&2
  exit 1
}
PROBLEMS=""
problem() { PROBLEMS="$PROBLEMS  - $*"$'\n'; }
first10() { tr ' ' '\n' | sed '/^$/d' | head -10 | tr '\n' ' '; }

MAIN=$(lane_main_clone) || die "not in a git repo"
MAIN=$(cd "$MAIN" && pwd -P)
TOP=$(git rev-parse --show-toplevel 2>/dev/null) && TOP=$(cd "$TOP" && pwd -P)
[ "$TOP" = "$MAIN" ] || die "run this from the main clone ($MAIN), not from ${TOP:-$PWD}"

WT=$MAIN/.claude/worktrees/$NAME
BRANCH=worktree-$NAME
tip=$(git -C "$MAIN" rev-parse -q --verify "refs/heads/$BRANCH^{commit}")
[ -d "$WT" ] || [ -n "$tip" ] || die "there is no worktree $WT and no branch $BRANCH"

if [ -d "$WT" ]; then
  wt_top="" gitdir="" head=""
  { read -r wt_top; read -r gitdir; read -r head; } < <(
    git -C "$WT" rev-parse --show-toplevel --absolute-git-dir --symbolic-full-name HEAD 2>/dev/null
  )
  [ -n "$wt_top" ] && [ "$(cd "$wt_top" && pwd -P)" = "$WT" ] ||
    die "$WT exists but is not a git worktree"
  [ "$head" = "refs/heads/$BRANCH" ] || problem "$NAME has ${head#refs/heads/} checked out, not $BRANCH"
  [ -f "$gitdir/locked" ] && problem "$NAME is locked: $(cat "$gitdir/locked")"

  # `git worktree remove` refuses changes and untracked files, but deletes ignored files
  # silently, so only disposable ones may be ignored: regenerable caches, and this lane's own
  # Claude permission grants. --untracked-files is explicit, so status.showUntrackedFiles=no
  # can't hide anything.
  dirty="" junk=""
  while IFS= read -r -d '' e; do
    p=${e:3}
    case "${e:0:2}" in
      '!!') ;;
      R* | C*) # -z gives a rename or copy's source path as an extra field
        read -r -d '' _ || true
        dirty="$dirty $p"
        continue
        ;;
      *)
        dirty="$dirty $p"
        continue
        ;;
    esac
    case "$p" in
      .venv/ | */.venv/ | __pycache__/ | */__pycache__/ | .pytest_cache/ | */.pytest_cache/) ;;
      .ruff_cache/ | */.ruff_cache/ | .DS_Store | */.DS_Store | *.pyc | *.pyo | *.pyd) ;;
      .claude/settings.local.json) ;;
      *) junk="$junk $p" ;;
    esac
  done < <(git -C "$WT" status --porcelain=v1 -z --ignored=matching --untracked-files=normal)
  [ -n "$dirty" ] && problem "uncommitted or untracked:$(printf '%s' "$dirty" | first10)"
  [ -n "$junk" ] && problem "ignored files that are not disposable:$(printf '%s' "$junk" | first10)"
fi

if [ -n "$tip" ]; then
  ahead=$(git -C "$MAIN" rev-list --count "main..$tip")
  [ "$ahead" = 0 ] || problem "$BRANCH has $ahead commit(s) not in main. If its PR merged, run \
git pull --ff-only in the main clone first"

  # The worktree goes first, so check now that `git branch -d` will agree afterwards. It judges
  # against the branch's upstream if that ref exists, else the main clone's HEAD.
  up=$(git -C "$MAIN" rev-parse -q --verify "$BRANCH@{upstream}" 2>/dev/null)
  judge="the HEAD of the main clone"
  [ -n "$up" ] && judge="its upstream"
  git -C "$MAIN" merge-base --is-ancestor "$tip" "${up:-HEAD}" ||
    problem "git branch -d would refuse: ${tip:0:7} is not in $judge"

  # A freshly cut branch is also 0 commits ahead of main. Only a merged PR whose head is exactly
  # this tip says "merged" rather than "just started". gh failing counts as not merged.
  if heads=$(cd "$MAIN" && gh pr list --head "$BRANCH" --state merged --json headRefOid \
    --jq '.[].headRefOid'); then
    case $'\n'"$heads"$'\n' in
      *$'\n'"$tip"$'\n'*) ;;
      *) problem "no merged PR has $BRANCH at its tip ${tip:0:7}" ;;
    esac
  else
    problem "gh failed (its error is above), so the merge can't be confirmed"
  fi
fi

# Last, because it is the slow check (~0.3 s): any process of this user whose cwd is inside the
# worktree, a Claude session or a background job alike. Skipped when already refusing.
if [ -z "$PROBLEMS" ] && [ -d "$WT" ]; then
  while IFS=$'\t' read -r pid cmd cwd; do
    case "$cwd/" in "$WT"/*) problem "process $pid ($cmd) is running inside $NAME ($cwd)" ;; esac
  done < <(lane_process_cwds)
fi

if [ -n "$PROBLEMS" ]; then
  printf 'finish-lane: REFUSED %s, nothing removed:\n%s' "$NAME" "$PROBLEMS" >&2
  exit 1
fi

if [ -d "$WT" ]; then
  git -C "$MAIN" worktree remove "$WT" || die "git would not remove $WT; the branch was kept"
  echo "finish-lane: removed worktree $WT"
fi
if [ -n "$tip" ]; then
  git -C "$MAIN" branch -d "$BRANCH" || die "git would not delete $BRANCH"
fi
echo "finish-lane: $NAME finished"
