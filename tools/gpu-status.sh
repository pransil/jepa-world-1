#!/usr/bin/env bash
# Is the shared MPS GPU busy, and with what?
#
#   tools/gpu-status.sh
#
# The lock (jepa_world/utils/locks.py) is the queue: training and benchmarks take it, and a
# second job waits. This shows who holds it, plus any python training/eval/bench processes
# running anyway, e.g. ones started with the lock disabled.
set -uo pipefail
cd "$(dirname "$0")/.."
uv run --quiet python -m jepa_world.utils.locks
echo
echo "python jobs from this project (train / eval / bench / generate):"
ps -Ao pid,etime,command | grep -E "python.*jepa_world\.(train|eval|bench|sim|data)" | grep -v grep \
  | sed 's/^/  /' || echo "  none"
