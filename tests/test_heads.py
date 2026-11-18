"""Near-horizon predictor heads (SPEC §4.5)."""

from __future__ import annotations

import pytest
import torch
from jepa_world.models.heads import HorizonHead


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
