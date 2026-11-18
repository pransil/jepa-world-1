"""Synthetic moving-disc clips: the stand-in batch source until M2's episode loader exists."""

from __future__ import annotations

import torch

from jepa_world.data.synthetic import moving_disc_batch


def _g(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_shape_dtype_and_range():
    x = moving_disc_batch(2, 9, 16, generator=_g(0))
    assert x.shape == (2, 9, 3, 16, 16)
    assert x.dtype == torch.float32
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_same_seed_same_clips():
    assert torch.equal(
        moving_disc_batch(2, 9, 16, generator=_g(3)), moving_disc_batch(2, 9, 16, generator=_g(3))
    )


def test_different_seeds_differ():
    assert not torch.equal(
        moving_disc_batch(2, 9, 16, generator=_g(0)), moving_disc_batch(2, 9, 16, generator=_g(1))
    )


def test_disc_moves_over_the_clip():
    x = moving_disc_batch(4, 9, 32, generator=_g(0))
    assert not torch.equal(x[:, 0], x[:, -1])


def test_only_the_disc_moves():
    # The background (sky + terrain) is static within a clip, so the pixels that differ from
    # frame 0 are only where the disc was or is: a small fraction of the image.
    x = moving_disc_batch(2, 9, 32, generator=_g(0))
    changed = (x[:, 1:] != x[:, :1]).any(dim=2)  # (B, T-1, S, S)
    assert 0 < changed.float().mean() < 0.25
