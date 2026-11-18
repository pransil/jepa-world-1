"""Goal head (SPEC §4.6, §5.2): pooled levels -> diagonal 2-D Gaussian over the ball's displacement
from t, per horizon (docs/decisions/2026-11-17-goal-target-displacement.md)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from jepa_world.models.goal import GoalHead, displacement_targets, goal_nll

HORIZONS = (1, 2, 3, 4, 5, 10, 20)
BOUND = 10.0


def _head(**kw) -> GoalHead:
    args = dict(
        l3_ch=8, horizons=HORIZONS, hidden=8, embed_dim=4, norm_groups=2, logvar_bound=BOUND
    )
    args.update(kw)
    torch.manual_seed(0)
    return GoalHead(**args)


# ---------------------------------------------------------------- shapes


def test_output_is_mean_and_logvar_per_horizon():
    mean, logvar = _head()(torch.rand(2, 8, 4, 4))
    assert mean.shape == logvar.shape == (2, len(HORIZONS), 2)
    assert mean.dtype == logvar.dtype == torch.float32


def test_spec_shapes_with_l2():
    head = _head(l3_ch=256, l2_ch=128, hidden=32, embed_dim=8, norm_groups=8)
    mean, logvar = head(torch.rand(2, 256, 8, 8), torch.rand(2, 128, 32, 32))
    assert mean.shape == logvar.shape == (2, 7, 2)
    assert head.horizons == HORIZONS


def test_horizons_keep_constructor_order():
    head = _head(horizons=[20, 1, 5])
    assert head.horizons == (20, 1, 5)
    assert head(torch.rand(2, 8, 4, 4))[0].shape == (2, 3, 2)


def test_bad_shapes_raise():
    with pytest.raises(ValueError, match="l3"):
        _head()(torch.rand(2, 6, 4, 4))
    with pytest.raises(ValueError, match="l2"):
        _head()(torch.rand(2, 8, 4, 4), torch.rand(2, 4, 8, 8))  # L2 given but not built
    with pytest.raises(ValueError, match="l2"):
        _head(l2_ch=4)(torch.rand(2, 8, 4, 4))  # L2 built but not given
    with pytest.raises(ValueError, match="batch"):
        _head(l2_ch=4)(torch.rand(2, 8, 4, 4), torch.rand(3, 4, 8, 8))


def test_bad_config_raises():
    with pytest.raises(ValueError, match="logvar_bound"):
        _head(logvar_bound=0.0)
    with pytest.raises(ValueError, match="horizons"):
        _head(horizons=[])
    with pytest.raises(ValueError, match="horizons"):
        _head(horizons=[1, 1])
    with pytest.raises(ValueError, match="horizons"):
        _head(horizons=[-1, 2])
    with pytest.raises(ValueError, match="embed_dim"):
        _head(embed_dim=0)  # every horizon would get the same input
    with pytest.raises(ValueError, match="GroupNorm"):
        _head(hidden=6, norm_groups=4)
    with pytest.raises(ValueError, match="GroupNorm"):
        _head(hidden=4, norm_groups=4)  # 1 channel per group would erase the input


def test_no_buffers():
    """Like the encoder: nothing an EMA or a BatchNorm running stat would have to carry."""
    assert list(_head(l2_ch=4).buffers()) == []


# ---------------------------------------------------------------- variance and NLL


@pytest.mark.parametrize("scale", [1.0, 1e4, -1e4])
def test_variance_positive_and_finite_even_for_extreme_inputs(scale):
    mean, logvar = _head()(scale * torch.randn(2, 8, 4, 4))
    assert torch.isfinite(mean).all()
    assert (logvar.abs() <= BOUND).all()
    var = logvar.exp()
    assert (var > 0).all() and torch.isfinite(var).all()


@pytest.mark.parametrize("raw", [1e3, -1e3])
def test_logvar_bound_holds_when_the_raw_output_is_extreme(raw):
    """GroupNorm keeps the hidden layer small, so extreme *inputs* never reach the bound. Push the
    raw log-variance itself to +-1e3 (exp overflows to inf or underflows to 0 without the bound)."""
    head = _head()
    with torch.no_grad():
        head.mlp[-1].bias[2:] = raw  # outputs [2:4] are the raw log-variance
    mean, logvar = head(torch.rand(2, 8, 4, 4))
    assert (logvar.abs() <= BOUND).all()
    var = logvar.exp()
    assert (var > 0).all() and torch.isfinite(var).all()
    nll = goal_nll(mean, logvar, torch.zeros_like(mean))
    assert torch.isfinite(nll).all()


@pytest.mark.parametrize("scale", [1.0, 1e4])
def test_nll_finite_for_extreme_inputs_and_targets(scale):
    head = _head()
    mean, logvar = head(scale * torch.randn(2, 8, 4, 4))
    target = torch.full_like(mean, 1e3)
    nll = goal_nll(mean, logvar, target)
    assert nll.shape == (len(HORIZONS),)
    assert torch.isfinite(nll).all()
    nll.sum().backward()
    for p in head.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()


def test_nll_matches_closed_form():
    mean = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])  # (B=1, K=2, 2)
    logvar = torch.tensor([[[0.0, math.log(4.0)], [0.0, 0.0]]])
    target = torch.tensor([[[1.0, 1.0], [2.0, 3.0]]])
    half_log_2pi = 0.5 * math.log(2 * math.pi)
    k0 = (0.5 * 1.0 + half_log_2pi) + (0.5 * math.log(4.0) + half_log_2pi)
    k1 = half_log_2pi + (0.5 * 9.0 + half_log_2pi)
    torch.testing.assert_close(goal_nll(mean, logvar, target), torch.tensor([k0, k1]))


def test_nll_averages_over_batch():
    m, lv, t = torch.randn(4, 3, 2), torch.randn(4, 3, 2), torch.randn(4, 3, 2)
    per_sample = torch.stack(
        [goal_nll(m[i : i + 1], lv[i : i + 1], t[i : i + 1]) for i in range(4)]
    )
    torch.testing.assert_close(goal_nll(m, lv, t), per_sample.mean(0))


def test_nll_is_minimised_at_the_target():
    target = torch.randn(2, 3, 2)
    logvar = torch.zeros_like(target)
    at = goal_nll(target, logvar, target)
    off = goal_nll(target + 0.1, logvar, target)
    assert (at < off).all()


def test_nll_agrees_with_torch_gaussian_nll():
    """Cross-check against torch's reference (its mean over the 2 coords x2 = our sum, plus const)."""
    m, lv, t = torch.randn(3, 2, 2), torch.randn(3, 2, 2), torch.randn(3, 2, 2)
    ref = F.gaussian_nll_loss(m, t, lv.exp(), full=True, eps=0.0, reduction="none")  # (3, 2, 2)
    torch.testing.assert_close(goal_nll(m, lv, t), ref.sum(-1).mean(0))


def test_nll_detaches_the_target():
    mean = torch.randn(2, 3, 2, requires_grad=True)
    target = torch.randn(2, 3, 2, requires_grad=True)
    goal_nll(mean, torch.zeros(2, 3, 2), target).sum().backward()
    assert mean.grad is not None
    assert target.grad is None


def test_nll_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        goal_nll(torch.zeros(2, 3, 2), torch.zeros(2, 3, 2), torch.zeros(2, 4, 2))
    with pytest.raises(ValueError, match="shape"):
        goal_nll(*[torch.zeros(2, 3, 4)] * 3)  # matching, but not (x, z)


# ---------------------------------------------------------------- gradients and horizons


def test_gradients_reach_inputs_and_every_k_vector():
    head = _head(l2_ch=4)
    l3 = torch.rand(2, 8, 4, 4, requires_grad=True)
    l2 = torch.rand(2, 4, 8, 8, requires_grad=True)
    mean, logvar = head(l3, l2)
    goal_nll(mean, logvar, torch.randn(2, len(HORIZONS), 2)).sum().backward()
    for name, x in [("l3", l3), ("l2", l2)]:
        assert x.grad is not None and x.grad.abs().sum() > 0, name
    assert head.k_embed.requires_grad
    assert (head.k_embed.grad.abs().sum(dim=1) > 0).all()  # every horizon's vector learns


def test_horizons_give_different_predictions():
    mean, logvar = _head()(torch.rand(2, 8, 4, 4))
    for j in range(1, len(HORIZONS)):
        assert not torch.equal(mean[:, 0], mean[:, j])


def test_samples_are_independent():
    """GroupNorm is per sample: changing one sample must not change another's prediction."""
    head = _head()
    x = torch.rand(2, 8, 4, 4)
    y = x.clone()
    y[1] = torch.rand(8, 4, 4)
    torch.testing.assert_close(head(x)[0][0], head(y)[0][0], rtol=0, atol=0)


# ---------------------------------------------------------------- targets


def test_displacement_targets_hand_indexed():
    c = 4
    ball_pos = torch.arange(2 * 25 * 2, dtype=torch.float32).reshape(2, 25, 2) ** 1.5
    tgt = displacement_targets(ball_pos, c, [1, 20, 3])
    assert tgt.shape == (2, 3, 2)
    for j, k in enumerate([1, 20, 3]):
        torch.testing.assert_close(tgt[:, j], ball_pos[:, c - 1 + k] - ball_pos[:, c - 1])


def test_displacement_targets_zero_horizon_is_zero():
    ball_pos = torch.randn(2, 9, 2)
    assert torch.equal(displacement_targets(ball_pos, 4, [0])[:, 0], torch.zeros(2, 2))


def test_displacement_targets_float64_in_float32_out():
    """M2 stores ball_pos as float64: subtract at full precision, return the model's fp32."""
    ball_pos = torch.randn(2, 9, 2, dtype=torch.float64) + 1e6
    tgt = displacement_targets(ball_pos, 4, [1, 5])
    assert tgt.dtype == torch.float32
    expect = (ball_pos[:, [4, 8]] - ball_pos[:, 3:4]).float()
    torch.testing.assert_close(tgt, expect, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("context_frames", "horizons", "match"),
    [(4, [], "horizons"), (4, [-1, 2], "horizons"), (0, [1], "context_frames")],
)
def test_displacement_targets_bad_args_raise(context_frames, horizons, match):
    """Negative indices would silently wrap to the end of the episode."""
    with pytest.raises(ValueError, match=match):
        displacement_targets(torch.randn(2, 30, 2), context_frames, horizons)


def test_displacement_targets_too_short_raises():
    with pytest.raises(ValueError, match="frames"):
        displacement_targets(torch.randn(2, 23, 2), 4, HORIZONS)  # needs 4 + 20 = 24
    with pytest.raises(ValueError, match="ball_pos"):
        displacement_targets(torch.randn(2, 24, 3), 4, HORIZONS)


# ---------------------------------------------------------------- MPS


@pytest.mark.mps
def test_forward_backward_on_mps():
    head = _head(l3_ch=256, l2_ch=128, hidden=32, embed_dim=8, norm_groups=8).to("mps")
    l3 = torch.rand(4, 256, 8, 8, device="mps", requires_grad=True)
    l2 = torch.rand(4, 128, 32, 32, device="mps", requires_grad=True)
    mean, logvar = head(l3, l2)
    nll = goal_nll(mean, logvar, torch.randn(4, 7, 2, device="mps"))
    nll.sum().backward()
    assert torch.isfinite(nll).all().item()
    assert l3.grad is not None and l2.grad is not None


def test_goal_nll_and_the_far_head_nll_share_one_formula():
    """One NLL in the repo (models/losses.py): for K = 1 the goal NLL (sum over x, z; mean over B)
    equals the far heads' NLL on the same numbers laid out as (B, 2 channels, 1, 1)."""
    from jepa_world.models import losses
    from jepa_world.models.jepa import gaussian_nll

    m, lv, t = torch.randn(4, 1, 2), torch.randn(4, 1, 2), torch.randn(4, 1, 2)
    as_map = [x.reshape(4, 2, 1, 1) for x in (m, lv, t)]
    torch.testing.assert_close(goal_nll(m, lv, t)[0], gaussian_nll(*as_map))
    err2 = (t - m).square()
    torch.testing.assert_close(
        goal_nll(m, lv, t), losses.gaussian_nll_terms(err2, lv).sum(-1).mean(0)
    )
