"""Pixel decoder (SPEC §4.7, §5.3): U-Net decoder path over L3 -> L2 -> L1 -> RGB frame at t+1."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from jepa_world.models.pixel import PixelDecoder, pixel_l1

# tiny: (channels, size), fine -> coarse, with a 4x step at the end like SPEC's 32 -> 8
TINY = [(4, 16), (8, 8), (8, 2)]
SPEC = [(64, 64), (128, 32), (256, 8)]


def _dec(levels=TINY, widths=(4, 4), **kw) -> PixelDecoder:
    torch.manual_seed(0)
    return PixelDecoder(levels, widths=widths, norm_groups=2, **kw)


def _feats(levels=TINY, b=2, scale=1.0, grad=False):
    return [(scale * torch.randn(b, c, s, s)).requires_grad_(grad) for c, s in levels]


def test_output_is_an_rgb_frame_at_l1_resolution():
    out = _dec()(_feats())
    assert out.shape == (2, 3, 16, 16)
    assert out.dtype == torch.float32


def test_spec_shapes():
    out = _dec(SPEC, widths=(8, 8))(_feats(SPEC))
    assert out.shape == (2, 3, 64, 64)


def test_out_ch_is_configurable():
    assert _dec(out_ch=1)(_feats()).shape == (2, 1, 16, 16)


@pytest.mark.parametrize("scale", [1.0, 1e4, -1e4])
def test_output_in_target_range_even_for_extreme_inputs(scale):
    out = _dec()(_feats(scale=scale))
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_gradients_reach_every_level():
    feats = _feats(grad=True)
    pred = _dec()(feats)
    pixel_l1(pred, torch.rand_like(pred)).backward()
    for (c, s), f in zip(TINY, feats, strict=True):
        assert f.grad is not None and f.grad.abs().sum() > 0, (c, s)


def test_l1_skip_reaches_the_output_directly():
    """The last stage reads L1 itself (SPEC §4.7), not only the upsampled coarse features: a change
    confined to one L1 location changes the prediction there."""
    dec = _dec()
    feats = _feats()
    base = dec(feats)
    feats[0] = feats[0].clone()
    feats[0][:, :, 8, 8] += 5.0
    assert not torch.equal(dec(feats)[:, :, 8, 8], base[:, :, 8, 8])


def test_skips_off_reads_the_coarsest_level_only():
    """M7 ablation (docs/decisions/2026-11-17-skip-connections-meaning.md, option A): with skips
    off, L1 and L2 are not consumed, so nothing but L3 reaches the frame."""
    dec = _dec(skips=False)
    feats = _feats(grad=True)
    pred = dec(feats)
    assert pred.shape == (2, 3, 16, 16)
    pixel_l1(pred, torch.rand_like(pred)).backward()
    assert feats[0].grad is None and feats[1].grad is None
    assert feats[2].grad is not None and feats[2].grad.abs().sum() > 0
    other = [torch.randn_like(feats[0]), torch.randn_like(feats[1]), feats[2].detach()]
    torch.testing.assert_close(dec(other), pred.detach(), rtol=0, atol=0)


def test_skips_on_is_the_default_and_reads_every_level():
    assert _dec().skips is True
    assert _dec(skips=False).skips is False


def test_samples_are_independent():
    dec = _dec()
    a = _feats()
    b = [f.clone() for f in a]
    b[2][1] = torch.randn_like(b[2][1])
    torch.testing.assert_close(dec(a)[0], dec(b)[0], rtol=0, atol=0)


def test_bad_inputs_raise():
    dec = _dec()
    with pytest.raises(ValueError, match="levels"):
        dec(_feats()[:2])
    bad = _feats()
    bad[1] = torch.randn(2, 6, 8, 8)
    with pytest.raises(ValueError, match="level 1"):
        dec(bad)
    bad = _feats()
    bad[2] = torch.randn(2, 8, 4, 4)
    with pytest.raises(ValueError, match="level 2"):
        dec(bad)
    bad = _feats()
    bad[0] = torch.tensor(0.0)
    with pytest.raises(ValueError, match="level 0"):
        dec(bad)


def test_bad_config_raises():
    with pytest.raises(ValueError, match="widths"):
        _dec(widths=(4,))
    with pytest.raises(ValueError, match="size"):
        _dec([(4, 16), (8, 8), (8, 16)])  # coarser level must be smaller
    with pytest.raises(ValueError, match="size"):
        _dec([(4, 16), (8, 6), (8, 2)])  # not an integer factor
    with pytest.raises(ValueError, match="two levels"):
        _dec([(4, 16)], widths=())
    with pytest.raises(ValueError, match="GroupNorm"):
        _dec(widths=(3, 4))


def test_no_buffers():
    assert list(_dec().buffers()) == []


def test_pixel_l1_matches_torch_and_is_zero_at_target():
    pred, target = torch.rand(2, 3, 8, 8), torch.rand(2, 3, 8, 8)
    torch.testing.assert_close(pixel_l1(pred, target), F.l1_loss(pred, target))
    assert pixel_l1(target, target) == 0
    assert pixel_l1(pred, target).shape == ()


def test_pixel_l1_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        pixel_l1(torch.rand(2, 3, 8, 8), torch.rand(2, 3, 4, 4))


def test_pixel_l1_detaches_the_target():
    pred = torch.rand(2, 3, 4, 4, requires_grad=True)
    target = torch.rand(2, 3, 4, 4, requires_grad=True)
    pixel_l1(pred, target).backward()
    assert pred.grad is not None and target.grad is None


@pytest.mark.mps
def test_forward_backward_on_mps():
    dec = _dec(SPEC, widths=(16, 16)).to("mps")
    feats = [f.to("mps").requires_grad_(True) for f in _feats(SPEC, b=4)]
    pred = dec(feats)
    loss = pixel_l1(pred, torch.rand_like(pred))
    loss.backward()
    assert torch.isfinite(loss).item()
    assert 0.0 <= pred.min().item() and pred.max().item() <= 1.0
    assert all(f.grad is not None for f in feats)
