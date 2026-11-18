# Shared by lane-owners.sh and finish-lane.sh. Source it: it only defines functions.
# bash 3.2 compatible (macOS /bin/bash). Design: tools/DESIGN.md.

# The main clone: the directory holding the shared .git. Fails outside a git repo.
lane_main_clone() {
  local common
  common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 1
  printf '%s\n' "${common%/*}"
}

# A process's working directory, from lsof (macOS) or /proc (Linux). Empty if neither can tell.
lane_cwd_of() {
  local cwd
  cwd=$(lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)
  if [ -z "$cwd" ] && [ -e "/proc/$1/cwd" ]; then cwd=$(readlink "/proc/$1/cwd" 2>/dev/null); fi
  printf '%s\n' "$cwd"
}

# Live Claude Code sessions, one "pid<TAB>cwd" line each. $CC_SOCKS_DIR/<pid>.sock (default
# /tmp/cc-socks) names each session's pid. Liveness is `kill -0`, not a timestamp, because a
# sleeping laptop suspends sessions rather than killing them.
lane_live_sessions() {
  local sock pid
  for sock in "${CC_SOCKS_DIR:-/tmp/cc-socks}"/*.sock; do
    [ -S "$sock" ] || continue
    pid=${sock##*/}
    pid=${pid%.sock}
    case "$pid" in '' | *[!0-9]*) continue ;; esac
    kill -0 "$pid" 2>/dev/null || continue
    printf '%s\t%s\n' "$pid" "$(lane_cwd_of "$pid")"
  done
}
