"""Where shared artifacts live. Datasets, runs and locks are OUTSIDE git, and shared by every worktree.

Root: $JEPA_ARTIFACTS, default ~/devel/jepa/artifacts/world-1. Tests redirect it to a tmp dir.

  datasets/<name>-<cfghash>/   immutable and write-once. The same config gives the same dir, so
                               every lane reuses the data instead of regenerating it.
  runs/<utc>-<branch>-<name>-<rand>/   one dir per run, unique by construction.
  locks/                        see locks.py

RULE for dataset configs: put a `generator_version` key in the config, and bump it whenever the
simulator's output changes for the same config. Otherwise the hash would point stale data at new
code.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

ENV_VAR = "JEPA_ARTIFACTS"
DEFAULT_ROOT = Path.home() / "devel" / "jepa" / "artifacts" / "world-1"


def artifacts_root() -> Path:
    root = Path(os.environ.get(ENV_VAR) or DEFAULT_ROOT).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def config_hash(cfg: dict) -> str:
    """Stable 12-hex-digit hash of a JSON-serialisable config (key order does not matter)."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:40] or "x"


def dataset_dir(cfg: dict, name: str = "ds") -> Path:
    """Final location of the dataset generated from `cfg`. It may not exist yet."""
    return artifacts_root() / "datasets" / f"{_slug(name)}-{config_hash(cfg)}"


def build_dataset_once(cfg: dict, build: Callable[[Path], None], name: str = "ds") -> Path:
    """Return the dataset dir for `cfg`, building it with `build(staging_dir)` only if it is missing.

    Concurrency-safe across processes and worktrees:
    - builders of the same config are serialised by a per-hash lock, so the second one waits
      and then reuses the first one's result;
    - `build` writes into a private staging dir, which is renamed into place atomically only on
      success, so a crash never leaves a half-written dataset where readers would find it.
    Writes `config.json` into the dataset dir.
    """
    from jepa_world.utils.locks import file_lock

    final = dataset_dir(cfg, name)
    if final.exists():
        return final
    with file_lock(
        artifacts_root() / "locks" / f"dataset-{final.name}.lock", label=f"build {final.name}"
    ):
        if final.exists():  # someone else built it while we waited
            return final
        staging = final.parent / f".staging-{final.name}-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True)
        try:
            (staging / "config.json").write_text(
                json.dumps(cfg, sort_keys=True, indent=2, default=str)
            )
            build(staging)
            os.rename(staging, final)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return final


def git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "branch", "--show-current"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out or "detached"


def new_run_dir(name: str) -> Path:
    """Create and return a fresh, unique run directory: runs/<utc>-<branch>-<name>-<rand>/."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = (
        artifacts_root()
        / "runs"
        / f"{stamp}-{_slug(git_branch())}-{_slug(name)}-{uuid.uuid4().hex[:6]}"
    )
    run.mkdir(parents=True, exist_ok=False)
    return run
