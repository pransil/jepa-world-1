"""Reproducibility: seeding, and the metadata every run records (SPEC §6: fix seeds, log versions)."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch

LIBRARIES = ("torch", "mujoco", "numpy", "scikit-learn", "pyyaml", "tensorboard")


def seed_everything(seed: int) -> None:
    """Seed python, numpy's global RNG, and torch (CPU and MPS). Prefer passing explicit
    np.random.Generator / torch.Generator objects in data code. This is the safety net."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds MPS and CUDA


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_info() -> dict:
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def library_versions() -> dict:
    out = {}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_metadata() -> dict:
    """Everything needed to say which code, libraries and machine produced a result."""
    return {
        "git": git_info(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": library_versions(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "argv": sys.argv,
    }
