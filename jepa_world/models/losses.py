"""Loss formulas shared by the JEPA core and the auxiliary heads, defined once.

Each caller keeps its own reduction: the far heads sum over channels and average over (batch,
positions) (jepa.gaussian_nll), and the goal head sums over (x, z) and averages over the batch,
one value per horizon (goal.goal_nll).
"""

from __future__ import annotations

import math

import torch

LOG_2PI = math.log(2 * math.pi)


def gaussian_nll_terms(err2: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Elementwise diagonal-Gaussian NLL, from the squared error and the log-variance:

        0.5 * [err2 * exp(-logvar) + logvar + log(2 pi)]

    The log(2 pi) term makes it a true NLL (it can go negative)."""
    return 0.5 * (err2 * torch.exp(-logvar) + logvar + LOG_2PI)
