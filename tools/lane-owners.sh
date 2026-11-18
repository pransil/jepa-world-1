#!/usr/bin/env bash
# Who is working where, and is anyone holding the files I'm about to edit?
#
#   tools/lane-owners.sh                     every lane, plus all files in flight
#   tools/lane-owners.sh path/a.py path/b    same, then a collision check for just those paths
#                                            (exit 1 if another lane holds any of them)
#
# Everything is DERIVED at the moment you ask: git worktrees, git status, origin refs, open PRs,
# and live Claude sessions. There is no board file to keep current. Photolok ran one, and measured
# it as the repo's most hand-resolved merge-conflict file while it caught nothing git did not.
#
# Live sessions: /tmp/cc-socks/<pid>.sock names each running Claude Code session's pid, and
# `lsof -d cwd` gives its working directory. Liveness is `kill -0`, not a timestamp, because a
# sleeping laptop suspends sessions rather than killing them.
#
# No owner does NOT mean abandoned. A paused session and an abandoned one look the same, and only
# the user knows which.
set -uo pipefail

. "$(dirname "$0")/lane-lib.sh"

BASE=${BASE:-origin/main}
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not in a git repo"; exit 2; }
MAIN=$(cd "$ROOT" && lane_main_clone)
git -C "$MAIN" fetch -q origin 2>/dev/null || echo "(fetch failed: origin refs may be stale)"

LIVE=()   # "pid<TAB>cwd"
while IFS= read -r e; do LIVE+=("$e"); done < <(lane_live_sessions)
owner_of() {  # pids whose cwd is inside worktree $1 (but not inside a nested worktree of it)
  local out="" e pid cwd
  for e in "${LIVE[@]+"${LIVE[@]}"}"; do
    pid=${e%%$'\t'*}; cwd=${e#*$'\t'}
    if [ "$1" = "$MAIN" ]; then
      case "$cwd" in "$MAIN"/.claude/worktrees/*) continue ;; esac
    fi
    case "$cwd/" in "$1"/*) out="$out${out:+,}$pid" ;; esac
  done
  echo "${out:--}"
}

worktrees() { git -C "$MAIN" worktree list --porcelain | awk '/^worktree /{w=substr($0,10)} /^branch /{sub("refs/heads/","",$2); print w"\t"$2}'; }
held_files() {  # $1=worktree $2=branch : uncommitted + committed-but-not-in-BASE
  { git -C "$1" status --porcelain 2>/dev/null | cut -c4- | sed 's/.* -> //'
    [ "$1" != "$MAIN" ] && git -C "$1" diff --name-only "$BASE...$2" 2>/dev/null
  } | sed 's/^"//; s/"$//' | sort -u | grep -v '^$'
}

# Merged-ness comes from PR STATE, never from refs. With merge commits a merged branch and a
# freshly cut one are both "0 commits ahead of main", so git alone cannot tell them apart.
PRSTATE=$(cd "$MAIN" && gh pr list --state all --limit 200 --json number,headRefName,state \
          --jq '.[] | "\(.headRefName)\t\(.state) #\(.number)"' 2>/dev/null) || PRSTATE=""
pr_of() { printf '%s\n' "$PRSTATE" | awk -F'\t' -v b="$1" '$1==b{print $2; f=1; exit} END{if(!f) print "-"}'; }

printf '%-22s %-30s %-7s %-12s %-6s %-12s %s\n' WORKTREE BRANCH PUSHED PR DIRTY OWNER STATE
while IFS=$'\t' read -r path br; do
  name=$([ "$path" = "$MAIN" ] && echo "(main clone)" || basename "$path")
  dirty=$(git -C "$path" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  owner=$(owner_of "$path")
  if [ "$path" = "$MAIN" ]; then
    state=$([ "$dirty" = 0 ] && echo "ok (fetch/merge only)" || echo "DIRTY main clone: edits belong in a worktree")
    printf '%-22s %-30s %-7s %-12s %-6s %-12s %s\n' "$name" "$br" - - "$dirty" "$owner" "$state"
    continue
  fi
  pushed=no;  git -C "$MAIN" rev-parse -q --verify "origin/$br" >/dev/null && pushed=yes
  own=$(git -C "$MAIN" rev-list --count "$BASE..$br" 2>/dev/null || echo 0)
  pr=$(pr_of "$br")
  if   [ "$owner" != - ] && [ "$pushed" = no ] && [ "$own$dirty" != 00 ]; then state="active, NOT PUSHED: invisible to other lanes"
  elif [ "$owner" != - ];                                        then state="active"
  elif [ "${pr%% *}" = MERGED ] && [ "$dirty" = 0 ];             then state="merged: removable"
  elif [ "${pr%% *}" = MERGED ];                                 then state="merged but $dirty uncommitted: ask user"
  else                                                                state="no live session: paused or abandoned? ask user"
  fi
  printf '%-22s %-30s %-7s %-12s %-6s %-12s %s\n' "$name" "$br" "$pushed" "$pr" "$dirty" "$owner" "$state"
done < <(worktrees)

for e in "${LIVE[@]+"${LIVE[@]}"}"; do
  pid=${e%%$'\t'*}; cwd=${e#*$'\t'}
  case "$cwd" in "$MAIN"/.claude/worktrees/*) ;; "$MAIN"|"$MAIN"/*)
    echo "!! session pid $pid is in the MAIN CLONE. If it edits a worktree via cd, its skills and"
    echo "   subagents still run against the main clone. Lanes should use EnterWorktree." ;;
  esac
done

if command -v gh >/dev/null 2>&1; then
  prs=$(cd "$MAIN" && gh pr list --state open --json number,headRefName,title \
        --jq '.[] | "#\(.number)\t\(.headRefName)\t\(.title)"' 2>/dev/null)
  [ -n "$prs" ] && { echo; echo "open PRs (files: gh pr diff <n> --name-only):"; printf '%s\n' "$prs" | sed 's/^/  /'; }
fi

echo
HIT=0
if [ $# -gt 0 ]; then
  echo "=== collision check for: $* ==="
  while IFS=$'\t' read -r path br; do
    [ "$path" = "$ROOT" ] && continue           # your own lane is not a collision
    files=$(held_files "$path" "$br")
    for want in "$@"; do
      if printf '%s\n' "$files" | grep -qxF "$want"; then
        echo "  !! $want is held by $(basename "$path") ($br, owner $(owner_of "$path"))"; HIT=1
      fi
    done
  done < <(worktrees)
  if [ $HIT = 0 ]; then echo "  clear: no other lane holds those paths."
  else echo "  STOP: coordinate with that lane or ask the user before editing. Writing over an"
       echo "  uncommitted change destroys it with nothing in git to recover."; fi
else
  echo "=== files in flight (uncommitted, or committed but not in $BASE) ==="
  any=0
  while IFS=$'\t' read -r path br; do
    files=$(held_files "$path" "$br"); [ -z "$files" ] && continue
    any=1; n=$(printf '%s\n' "$files" | wc -l | tr -d ' ')
    echo "  $([ "$path" = "$MAIN" ] && echo "(main clone)" || basename "$path") ($br): $n file(s)"
    printf '%s\n' "$files" | head -8 | sed 's/^/      /'
    [ "$n" -gt 8 ] && echo "      ... $((n - 8)) more"
  done < <(worktrees)
  [ $any = 0 ] && echo "  nothing in flight."
fi
echo
echo "Sees worktrees on THIS machine. A lane elsewhere shows only in the open-PR list."
exit $HIT
