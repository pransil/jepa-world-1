"""Predictor heads (SPEC §4.5): near (mean) and far (mean + bounded log-variance)."""

from __future__ import annotations

import math

import pytest
import torch

from jepa_world.models.heads import GaussianHead, HorizonHead


def _head(**kw) -> HorizonHead:
    args = dict(in_ch=8, out_ch=8, hidden=4, kernel=3, embed_dim=2, action_dim=0, norm_groups=2)
    args.update(kw)
    torch.manual_seed(0)
    return HorizonHead(**args)


def test_output_matches_the_level_it_predicts():
    feat = torch.rand(2, 8, 16, 16)
    assert _head()(feat).shape == (2, 8, 16, 16)


@pytest.mark.parametrize("kernel", [3, 1])
def test_kernel_size_keeps_spatial_shape(kernel):
    assert _head(kernel=kernel)(torch.rand(2, 8, 16, 16)).shape == (2, 8, 16, 16)


def test_k_vector_is_learned_and_reaches_the_output():
    head = _head()
    feat = torch.rand(2, 8, 16, 16)
    head(feat).sum().backward()
    assert head.k_embed.requires_grad
    assert head.k_embed.grad is not None and head.k_embed.grad.abs().sum() > 0


def test_action_slot_defaults_to_zeros():
    head = _head(action_dim=3)
    feat = torch.rand(2, 8, 16, 16)
    torch.testing.assert_close(head(feat), head(feat, torch.zeros(2, 3)), rtol=0, atol=0)
    assert not torch.equal(head(feat), head(feat, torch.ones(2, 3)))


def test_bad_action_raises():
    feat = torch.rand(2, 8, 16, 16)
    with pytest.raises(ValueError, match="action"):
        _head(action_dim=3)(feat, torch.zeros(2, 4))
    with pytest.raises(ValueError, match="action"):
        _head(action_dim=0)(feat, torch.zeros(2, 3))


def test_hidden_width_must_fit_norm_groups():
    with pytest.raises(ValueError, match="groups"):
        _head(hidden=6, norm_groups=4)


# ---------------------------------------------------------------- far heads (SPEC §4.5, M4)


def _gauss(bound: float = 10.0, **kw) -> GaussianHead:
    args = dict(in_ch=16, out_ch=16, hidden=4, kernel=3, embed_dim=2, action_dim=0, norm_groups=2)
    args.update(kw)
    torch.manual_seed(0)
    return GaussianHead(**args, logvar_bound=bound)


def test_far_head_outputs_mean_and_logvar_shaped_like_its_level():
    mean, logvar = _gauss()(torch.rand(2, 16, 8, 8))
    assert mean.shape == logvar.shape == (2, 16, 8, 8)
    assert not torch.equal(mean, logvar)


def test_far_head_variance_is_positive_and_finite():
    mean, logvar = _gauss()(torch.rand(2, 16, 8, 8))
    var = logvar.exp()
    assert (var > 0).all() and torch.isfinite(var).all()


@pytest.mark.parametrize("raw_bias", [-1e4, -50.0, 50.0, 1e4])
def test_far_head_logvar_stays_inside_the_bound_at_extremes(raw_bias):
    head = _gauss(bound=10.0)
    with torch.no_grad():
        head.body[-1].bias[16:] = raw_bias  # the logvar half of the output conv
    _, logvar = head(torch.rand(2, 16, 8, 8))
    assert (logvar.abs() <= 10.0).all()
    var = logvar.exp()
    assert (var > 0).all() and torch.isfinite(var).all()


def test_far_head_logvar_keeps_a_gradient_near_the_bound():
    # A hard clamp would give exactly zero gradient here. The tanh bound does not.
    head = _gauss(bound=10.0)
    with torch.no_grad():
        head.body[-1].bias[16:] = 30.0
    _, logvar = head(torch.rand(2, 16, 8, 8))
    assert (logvar > 9.9).all()
    logvar.sum().backward()
    assert head.body[-1].bias.grad[16:].abs().min() > 0


def test_far_head_logvar_is_identity_near_zero():
    head = _gauss(bound=10.0)
    with torch.no_grad():
        head.body[-1].weight.zero_()
        head.body[-1].bias.fill_(0.5)
    _, logvar = head(torch.rand(2, 16, 8, 8))
    torch.testing.assert_close(logvar, torch.full_like(logvar, 10 * math.tanh(0.05)))


def test_far_head_bound_must_be_positive():
    with pytest.raises(ValueError, match="bound"):
        _gauss(bound=0.0)
