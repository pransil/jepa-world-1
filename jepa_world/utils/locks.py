"""Advisory file locks shared by every worktree and every agent on this machine.

WHY. All lanes share one M4 GPU. Two trainings at once each run at roughly half speed, and a
benchmark taken while another job is running is wrong. `gpu_lock()` queues heavy MPS jobs instead.
`file_lock()` also serialises builders of the same dataset (see paths.build_dataset_once).

WHY flock. The kernel releases an flock when its process dies, so a crashed or killed job
never leaves a stale lock behind, and nothing needs cleaning up. The holder metadata in the file
is for display only. Whether the lock is HELD is always decided by trying to acquire it.

Status from a shell:  uv run python -m jepa_world.utils.locks   (or tools/gpu-status.sh)
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class LockTimeout(TimeoutError):
    """The lock was still held when the timeout expired."""


def _holder_info(label: str) -> dict:
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        branch = ""
    return {
        "pid": os.getpid(),
        "label": label,
        "cwd": os.getcwd(),
        "branch": branch,
        "since": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": " ".join(sys.argv)[:200],
    }


def read_holder(path: Path) -> dict | None:
    """The metadata the current (or last) holder wrote. Display only, never proof the lock is held."""
    try:
        text = Path(path).read_text().strip()
        return json.loads(text) if text else None
    except (OSError, ValueError):
        return None


def is_locked(path: Path) -> bool:
    """True if another process holds the lock right now (tested by a non-blocking acquire)."""
    path = Path(path)
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


@contextmanager
def file_lock(
    path: Path, *, label: str = "", timeout: float | None = None, poll: float = 0.1
) -> Iterator[None]:
    """Hold an exclusive advisory lock on `path` for the duration of the block.

    timeout=None waits forever, and timeout=0 fails at once if the lock is held. When it has to
    wait, it prints who holds the lock once, to stderr, so a queued job's log explains the pause.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = None if timeout is None else time.monotonic() + timeout
    announced = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if deadline is not None and time.monotonic() >= deadline:
                    raise LockTimeout(f"{path} is held by {read_holder(path)}") from None
                if not announced:
                    print(
                        f"[lock] waiting for {path.name}, held by {read_holder(path)}",
                        file=sys.stderr,
                    )
                    announced = True
                time.sleep(poll)
        os.ftruncate(fd, 0)
        os.pwrite(fd, json.dumps(_holder_info(label)).encode(), 0)
        try:
            yield
        finally:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def gpu_lock_path() -> Path:
    from jepa_world.utils.paths import artifacts_root

    return artifacts_root() / "locks" / "mps.lock"


@contextmanager
def gpu_lock(label: str, *, timeout: float | None = None, enabled: bool = True) -> Iterator[None]:
    """Take the machine-wide MPS lock. Use it for training, benchmarks, and any long MPS job.

    enabled=False is for quick interactive checks that would not skew anyone else's run. Never
    disable it for a benchmark.
    """
    if not enabled:
        yield
        return
    with file_lock(gpu_lock_path(), label=label, timeout=timeout):
        yield


def _status() -> int:
    path = gpu_lock_path()
    if is_locked(path):
        print(f"MPS lock HELD: {read_holder(path)}")
        return 1
    print(f"MPS lock free ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_status())
