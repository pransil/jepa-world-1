"""Synthetic moving-disc clips: the stand-in batch source until M2's episode loader exists."""

from __future__ import annotations

import torch

from jepa_world.data.synthetic import BALL, moving_disc_batch


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


def test_with_gt_keeps_the_same_frames_and_adds_ball_pos():
    """Smoke runs with the goal stage need a ball_pos; asking for it must not change the clips."""
    frames = moving_disc_batch(2, 9, 32, generator=_g(5))
    batch = moving_disc_batch(2, 9, 32, generator=_g(5), with_gt=True)
    assert set(batch) == {"frames", "ball_pos"}
    assert torch.equal(batch["frames"], frames)
    assert batch["ball_pos"].shape == (2, 9, 2) and batch["ball_pos"].dtype == torch.float32


def test_ball_pos_is_the_disc_centre_in_frame_units_z_up():
    """ball_pos = (u / S, 1 - v / S): x along the image, z up. The pixel there is ball-coloured."""
    s = 32
    batch = moving_disc_batch(3, 9, s, generator=_g(2), with_gt=True)
    frames, pos = batch["frames"], batch["ball_pos"]
    assert (pos >= 0).all() and (pos <= 1).all()
    u = (pos[..., 0] * s).floor().long()
    v = ((1 - pos[..., 1]) * s).floor().long()
    for b in range(3):
        for t in range(9):
            px = frames[b, t, :, v[b, t], u[b, t]]
            torch.testing.assert_close(px, torch.tensor(BALL))
