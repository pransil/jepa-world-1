"""Device selection. Tests run on CPU, and training prefers MPS (SPEC §2)."""

from __future__ import annotations

import torch


def pick_device(name: str = "auto") -> torch.device:
    """'auto' picks MPS when available, else CPU. 'mps', 'cpu' or 'cuda' are taken literally,
    and an unavailable explicit choice raises rather than silently falling back."""
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but torch.backends.mps.is_available() is False")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return torch.device(name)
