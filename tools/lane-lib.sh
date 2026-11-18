# Shared by lane-owners.sh and finish-lane.sh. Source it: it only defines functions.
# bash 3.2 compatible (macOS /bin/bash). Design: tools/DESIGN.md.

# The main clone: the directory holding the shared .git. Fails outside a git repo.
lane_main_clone() {
  local common
  common=$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || return 1
  printf '%s\n' "${common%/*}"
}

# Every process of this user, one "pid<TAB>cwd" line each: the one process backend both tools
# use. Linux reads the /proc cwd links with GNU find, so CI covers this path. Elsewhere it is one
# lsof call, about 0.3 s on macOS, so callers run it last. A cwd containing a newline is not
# supported. An empty or broken list is possible (no GNU find, no lsof, a locked-down /proc), so a
# caller that must not fail open checks that its own pid ($$) is listed with its real cwd.
lane_process_cwds() {
  if [ -e /proc/self/cwd ]; then
    find /proc/[0-9]*/cwd -maxdepth 0 -user "$(id -u)" -printf '%h\t%l\n' 2>/dev/null |
      sed 's|^/proc/||'
  else
    lsof -a -u "$(id -u)" -d cwd -Fpn 2>/dev/null |
      awk '/^p/ { p = substr($0, 2) } /^n/ { print p "\t" substr($0, 2) }'
  fi
}

# Live Claude Code session pids, one per line. $CC_SOCKS_DIR/<pid>.sock (default /tmp/cc-socks)
# names each session's pid. Liveness is `kill -0`, not a timestamp, because a sleeping laptop
# suspends sessions rather than killing them.
lane_session_pids() {
  local sock pid
  for sock in "${CC_SOCKS_DIR:-/tmp/cc-socks}"/*.sock; do
    [ -S "$sock" ] || continue
    pid=${sock##*/}
    pid=${pid%.sock}
    case "$pid" in '' | *[!0-9]*) continue ;; esac
    kill -0 "$pid" 2>/dev/null && printf '%s\n' "$pid"
  done
}

# Live Claude Code sessions with their cwds, one "pid<TAB>cwd" line each.
lane_live_sessions() {
  local pids
  pids=$(lane_session_pids | tr '\n' ' ')
  [ -n "$pids" ] || return 0
  lane_process_cwds | awk -F'\t' -v pids=" $pids" 'index(pids, " " $1 " ")'
}
