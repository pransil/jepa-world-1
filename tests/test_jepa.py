"""JEPA core: target encoder, EMA, stop-grad, leakage at the model boundary, losses (SPEC §4-5)."""

from __future__ import annotations

import math

import pytest
import torch
from tiny import ALL, FAR, LEVEL, NEAR, TINY_SHAPE, random_frames, tiny_cfg, tiny_far_cfg

from jepa_world.models.jepa import (
    JEPA,
    compute_losses,
    context_stack,
    gaussian_nll,
    latent_loss,
    target_stack,
    variance_guard,
)


def _model(cfg=None) -> JEPA:
    torch.manual_seed(0)
    return JEPA(cfg or tiny_cfg(), NEAR)


def _far_model() -> JEPA:
    torch.manual_seed(0)
    return JEPA(tiny_far_cfg(), ALL)


# ---------------------------------------------------------------- stacking


def test_context_stack_is_frames_before_and_at_t_in_time_order():
    frames = torch.arange(9).float().view(1, 9, 1, 1, 1).expand(1, 9, 3, 2, 2)
    ctx = context_stack(frames, 4)
    assert ctx.shape == (1, 12, 2, 2)
    assert ctx[0, :, 0, 0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


@pytest.mark.parametrize("k", ALL)
def test_target_stack_ends_at_t_plus_k(k):
    frames = torch.arange(24).float().view(1, 24, 1, 1, 1).expand(1, 24, 3, 2, 2)
    tgt = target_stack(frames, 4, k)
    t = 3  # index of time t when C = 4
    assert tgt[0, ::3, 0, 0].tolist() == [t + k - 3, t + k - 2, t + k - 1, t + k]


def test_too_few_frames_raises():
    with pytest.raises(ValueError, match="frames"):
        _model()(random_frames(tiny_cfg())[:, :8])  # needs C + 5 = 9
    with pytest.raises(ValueError, match="frames"):
        _far_model()(random_frames(tiny_far_cfg())[:, :23])  # needs C + 20 = 24


# ---------------------------------------------------------------- shapes (SPEC §8 M3, M4)


def test_near_predictions_match_targets():
    out = _model()(random_frames(tiny_cfg()))
    assert sorted(out["pred"]) == NEAR == sorted(out["target"])
    for k in NEAR:
        assert out["pred"][k].shape == out["target"][k].shape == (2, 8, 16, 16)
    assert out["logvar"] == {}  # near heads are deterministic


def test_head_outputs_match_targets_at_every_level_and_horizon():
    # SPEC §8 M4: "head output shapes match targets"
    out = _far_model()(random_frames(tiny_far_cfg()))
    assert sorted(out["pred"]) == ALL == sorted(out["target"])
    assert sorted(out["logvar"]) == FAR
    for k in ALL:
        expected = (2, *TINY_SHAPE[LEVEL[k]])
        assert out["pred"][k].shape == out["target"][k].shape == expected, k
        assert out["target"][k].shape[1:] == out["context"][LEVEL[k]].shape[1:]
    for k in FAR:
        assert out["logvar"][k].shape == out["pred"][k].shape


def test_far_head_variance_is_positive():
    # SPEC §8 M4: "far-head variance is positive"
    out = _far_model()(random_frames(tiny_far_cfg()))
    for k in FAR:
        var = out["logvar"][k].exp()
        assert (var > 0).all() and torch.isfinite(var).all(), k


def test_far_heads_read_only_their_own_level():
    # docs/decisions/2026-11-17-far-heads-read-their-level.md
    model = _far_model()
    for k in ALL:
        assert model.head_level[k] == LEVEL[k]
    feat = torch.rand(2, *TINY_SHAPE["L3"])
    mean, logvar = model.heads["20"](feat)
    assert mean.shape == feat.shape


def test_targets_are_encoded_once_per_level_group():
    # Near stacks stop at L1; only the k=10 and k=20 stacks pay for L2, only k=20 for L3.
    model = _far_model()
    seen = {0: 0, 1: 0, 2: 0}
    hooks = [
        stage.register_forward_hook(
            lambda m, i, o, s=s: seen.__setitem__(s, seen[s] + i[0].shape[0])
        )
        for s, stage in enumerate(model.target_encoder.stages)
    ]
    model(random_frames(tiny_far_cfg()))
    for h in hooks:
        h.remove()
    assert seen == {0: 2 * 7, 1: 2 * 2, 2: 2 * 1}


def test_grouped_targets_equal_one_stack_at_a_time():
    model = _far_model()
    frames = random_frames(tiny_far_cfg())
    targets = model(frames)["target"]
    for k in ALL:
        with torch.no_grad():
            alone = model.target_encoder(target_stack(frames, 4, k), upto=LEVEL[k])[LEVEL[k]]
        torch.testing.assert_close(targets[k], alone, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- stop-grad (SPEC §8 M3)


def test_no_gradient_reaches_the_target_encoder():
    model = _model()
    out = model(random_frames(tiny_cfg()))
    for k in NEAR:
        assert not out["target"][k].requires_grad
    losses = compute_losses(model, random_frames(tiny_cfg()), tiny_cfg()["loss"])
    losses["loss/total"].backward()
    target_params = list(model.target_encoder.parameters())
    assert target_params
    for p in target_params:
        assert not p.requires_grad
        assert p.grad is None
    # ...while the loss reaches every head and the encoder stage the heads read (L1 = stage 0).
    # Near heads read only L1, so the forward stops there and stages 1-2 (L2, L3) get no
    # gradient unless far heads read them.
    for name, p in model.named_parameters():
        if name.startswith(("heads.", "context_encoder.stages.0.")):
            assert p.grad is not None, name
        elif name.startswith("context_encoder."):
            assert p.grad is None, name


def test_no_gradient_reaches_the_target_encoder_with_far_heads():
    model = _far_model()
    cfg = tiny_far_cfg()
    out = model(random_frames(cfg))
    for k in ALL:
        assert not out["target"][k].requires_grad
    compute_losses(model, random_frames(cfg), cfg["loss"])["loss/total"].backward()
    for p in model.target_encoder.parameters():
        assert not p.requires_grad and p.grad is None
    for name, p in model.named_parameters():  # every trainable parameter is on a loss path now
        if not name.startswith("target_encoder."):
            assert p.grad is not None, name


def test_trainable_parameters_exclude_the_target_encoder():
    model = _model()
    trainable = {id(p) for p in model.trainable_parameters()}
    assert not trainable & {id(p) for p in model.target_encoder.parameters()}
    expected = {id(p) for p in model.context_encoder.parameters()} | {
        id(p) for p in model.heads.parameters()
    }
    assert trainable == expected


def test_latent_loss_detaches_its_target():
    pred = torch.rand(2, 4, 3, 3, requires_grad=True)
    target = torch.rand(2, 4, 3, 3, requires_grad=True)
    latent_loss(pred, target).backward()
    assert pred.grad is not None
    assert target.grad is None


# ---------------------------------------------------------------- EMA (SPEC §8 M3)


def test_target_starts_as_a_copy_of_the_context_encoder():
    model = _model()
    for pt, pc in zip(
        model.target_encoder.parameters(), model.context_encoder.parameters(), strict=True
    ):
        assert torch.equal(pt, pc)
        assert pt.data_ptr() != pc.data_ptr()


@pytest.mark.parametrize("m", [0.996, 0.9, 0.5])
def test_ema_update_matches_the_formula(m):
    model = _model()
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(torch.randn_like(p))
    before = [p.clone() for p in model.target_encoder.parameters()]
    ctx = [p.clone() for p in model.context_encoder.parameters()]
    model.ema_update(m)
    for pt, t0, c in zip(model.target_encoder.parameters(), before, ctx, strict=True):
        torch.testing.assert_close(pt, m * t0 + (1 - m) * c, rtol=1e-6, atol=1e-7)
    for p, c in zip(model.context_encoder.parameters(), ctx, strict=True):
        assert torch.equal(p, c)  # the context encoder is untouched


def test_ema_extremes_are_exact():
    model = _model()
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.ema_update(1.0)
    for pt, t0 in zip(model.target_encoder.parameters(), before, strict=True):
        assert torch.equal(pt, t0)
    model.ema_update(0.0)
    for pt, pc in zip(
        model.target_encoder.parameters(), model.context_encoder.parameters(), strict=True
    ):
        assert torch.equal(pt, pc)


def test_ema_momentum_out_of_range_raises():
    with pytest.raises(ValueError, match="momentum"):
        _model().ema_update(1.5)


# ---------------------------------------------------------------- leakage at the model boundary


def test_context_ignores_future_frames():
    model = _far_model()
    frames = random_frames(tiny_far_cfg())
    later = frames.clone()
    later[:, 4:] = torch.rand_like(later[:, 4:])  # every frame after t, up to t+20
    a, b = model(frames), model(later)
    for level in a["context"]:
        assert torch.equal(a["context"][level], b["context"][level])
    for k in ALL:  # ...so no prediction, near or far, can see them either
        assert torch.equal(a["pred"][k], b["pred"][k])
    for k in FAR:
        assert torch.equal(a["logvar"][k], b["logvar"][k])
    at_t = frames.clone()
    at_t[:, 3] = torch.rand_like(at_t[:, 3])  # frame t itself IS context
    assert not torch.equal(model(at_t)["context"]["L1"], a["context"]["L1"])


@pytest.mark.parametrize("k", ALL)
def test_target_k_ignores_frames_after_t_plus_k(k):
    model = _far_model()
    frames = random_frames(tiny_far_cfg())
    base = model(frames)["target"]
    t_plus_k = 3 + k
    after = frames.clone()
    if t_plus_k + 1 < frames.shape[1]:
        after[:, t_plus_k + 1 :] = torch.rand_like(after[:, t_plus_k + 1 :])
    assert torch.equal(model(after)["target"][k], base[k])
    at = frames.clone()
    at[:, t_plus_k] = torch.rand_like(at[:, t_plus_k])  # frame t+k IS the target's last frame
    assert not torch.equal(model(at)["target"][k], base[k])


# ---------------------------------------------------------------- losses


def test_latent_loss_is_zero_at_equality():
    x = torch.rand(2, 4, 3, 3)
    assert latent_loss(x, x.clone()).item() == 0.0


def test_latent_loss_matches_hand_value():
    pred = torch.zeros(1, 2, 1, 2)
    target = torch.tensor([[[[1.0, 0.0]], [[2.0, 3.0]]]])  # (1, 2 channels, 1, 2 positions)
    # position 0: 1^2 + 2^2 = 5; position 1: 0^2 + 3^2 = 9; mean over positions = 7
    assert latent_loss(pred, target).item() == pytest.approx(7.0)


def test_latent_loss_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        latent_loss(torch.rand(2, 4, 3, 3), torch.rand(2, 4, 3, 2))


def test_variance_guard_zero_when_std_above_gamma():
    z = torch.tensor([[-2.0], [2.0]]).view(2, 1, 1, 1).expand(2, 3, 2, 2)  # std 2 everywhere
    assert variance_guard(z, gamma=1.0, eps=1e-4).item() == 0.0


def test_variance_guard_penalises_a_collapsed_batch():
    z = torch.ones(4, 3, 2, 2, requires_grad=True)  # identical samples, std 0
    v = variance_guard(z, gamma=1.0, eps=1e-4)
    assert v.item() == pytest.approx(1.0 - 1e-2)
    v.backward()
    assert torch.isfinite(z.grad).all()


def test_variance_guard_uses_per_position_batch_std():
    # Varies across space but not across the batch: input-independent, so it IS collapsed.
    z = torch.arange(4.0).view(1, 1, 2, 2).expand(3, 1, 2, 2) * 10
    assert variance_guard(z, gamma=1.0, eps=1e-4).item() > 0.9


def test_variance_guard_single_sample_is_finite():
    v = variance_guard(torch.rand(1, 3, 2, 2), gamma=1.0, eps=1e-4)
    assert torch.isfinite(v)


def test_every_loss_term_is_present_and_finite():
    model = _model()
    losses = compute_losses(model, random_frames(tiny_cfg()), tiny_cfg()["loss"])
    expected = {f"loss/k{k}" for k in NEAR} | {"loss/var", "loss/total", "emb_std/L1"}
    assert set(losses) == expected
    for name, v in losses.items():
        assert v.ndim == 0 and torch.isfinite(v), name
    manual = sum(losses[f"loss/k{k}"] for k in NEAR) + losses["loss/var"]
    torch.testing.assert_close(losses["loss/total"], manual)


def test_collapse_guard_off_drops_the_term():
    cfg = tiny_cfg("loss.collapse_guard=false")
    losses = compute_losses(_model(cfg), random_frames(cfg), cfg["loss"])
    assert "loss/var" not in losses
    assert "emb_std/L1" in losses  # collapse stays visible either way
    manual = sum(losses[f"loss/k{k}"] for k in NEAR)
    torch.testing.assert_close(losses["loss/total"], manual)


def test_loss_weights_come_from_config():
    cfg = tiny_cfg("loss.w_k=2.0", "loss.w_var=3.0")
    losses = compute_losses(_model(cfg), random_frames(cfg), cfg["loss"])
    manual = 2.0 * sum(losses[f"loss/k{k}"] for k in NEAR) + 3.0 * losses["loss/var"]
    torch.testing.assert_close(losses["loss/total"], manual)


def test_unknown_latent_distance_raises():
    cfg = tiny_cfg("loss.latent_distance=cosine")
    with pytest.raises(ValueError, match="latent_distance"):
        compute_losses(_model(cfg), random_frames(cfg), cfg["loss"])


# ---------------------------------------------------------------- Gaussian NLL (SPEC §5.1, M4)


def test_gaussian_nll_matches_hand_value():
    mean = torch.zeros(1, 2, 1, 1)
    target = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1)
    logvar = torch.tensor([0.0, math.log(4.0)]).view(1, 2, 1, 1)
    # channel 0: 0.5*(1/1 + 0 + log2pi); channel 1: 0.5*(4/4 + log4 + log2pi); summed over channels
    expected = 0.5 * (1 + math.log(2 * math.pi)) + 0.5 * (1 + math.log(4) + math.log(2 * math.pi))
    assert gaussian_nll(mean, logvar, target).item() == pytest.approx(expected)


def test_gaussian_nll_with_unit_variance_is_half_the_near_loss_plus_a_constant():
    mean, target = torch.rand(2, 4, 3, 3), torch.rand(2, 4, 3, 3)
    nll = gaussian_nll(mean, torch.zeros_like(mean), target)
    expected = 0.5 * latent_loss(mean, target) + 0.5 * 4 * math.log(2 * math.pi)
    torch.testing.assert_close(nll, expected)


def test_gaussian_nll_detaches_its_target_and_trains_mean_and_logvar():
    mean = torch.rand(2, 4, 3, 3, requires_grad=True)
    logvar = torch.zeros(2, 4, 3, 3, requires_grad=True)
    target = torch.rand(2, 4, 3, 3, requires_grad=True)
    gaussian_nll(mean, logvar, target).backward()
    assert target.grad is None
    assert mean.grad is not None and logvar.grad is not None


def test_gaussian_nll_is_minimised_by_the_true_error_variance():
    # d NLL / d logvar = 0.5 * (1 - err^2 / var): zero where var equals the squared error
    err = torch.full((1, 1, 1, 1), 0.3)
    logvar = torch.log(err**2).requires_grad_(True)
    gaussian_nll(torch.zeros(1, 1, 1, 1), logvar, err).backward()
    assert abs(logvar.grad.item()) < 1e-6


def test_gaussian_nll_is_finite_at_the_logvar_bound():
    mean, target = torch.zeros(2, 4, 3, 3), torch.full((2, 4, 3, 3), 100.0)
    for lv in (-10.0, 10.0):
        assert torch.isfinite(gaussian_nll(mean, torch.full_like(mean, lv), target))


def test_gaussian_nll_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        gaussian_nll(torch.rand(2, 4, 3, 3), torch.rand(2, 4, 3, 3), torch.rand(2, 4, 3, 2))


# ---------------------------------------------------------------- losses with far heads


def _far_losses(active=None):
    cfg = tiny_far_cfg()
    model = _far_model()
    return model, compute_losses(model, random_frames(cfg), cfg["loss"], active=active)


def test_every_term_is_logged_with_far_heads():
    _, losses = _far_losses()
    expected = (
        {f"loss/k{k}" for k in ALL}
        | {f"sqerr/k{k}" for k in FAR}
        | {f"logvar/k{k}" for k in FAR}
        | {"loss/var", "loss/total", "emb_std/L1", "emb_std/L2", "emb_std/L3"}
    )
    assert set(losses) == expected
    for name, v in losses.items():
        assert v.ndim == 0 and torch.isfinite(v), name
    manual = sum(losses[f"loss/k{k}"] for k in ALL) + losses["loss/var"]
    torch.testing.assert_close(losses["loss/total"], manual)
    for k in FAR:  # diagnostics are detached: they never steer training
        assert not losses[f"sqerr/k{k}"].requires_grad and not losses[f"logvar/k{k}"].requires_grad


def test_far_loss_is_the_gaussian_nll_of_its_head():
    model, losses = _far_losses()
    out = model(random_frames(tiny_far_cfg()))
    for k in FAR:
        nll = gaussian_nll(out["pred"][k], out["logvar"][k], out["target"][k])
        torch.testing.assert_close(losses[f"loss/k{k}"], nll)
        torch.testing.assert_close(
            losses[f"sqerr/k{k}"], latent_loss(out["pred"][k], out["target"][k])
        )
        torch.testing.assert_close(losses[f"logvar/k{k}"], out["logvar"][k].mean())


def test_inactive_horizons_are_logged_but_left_out_of_the_total():
    model, losses = _far_losses(active=NEAR)
    assert {f"loss/k{k}" for k in FAR} <= set(losses)  # still logged
    out = model(random_frames(tiny_far_cfg()))
    guard_l1 = variance_guard(out["context"]["L1"], 1.0, 1e-4)
    torch.testing.assert_close(losses["loss/var"], guard_l1)  # guard covers active levels only
    manual = sum(losses[f"loss/k{k}"] for k in NEAR) + losses["loss/var"]
    torch.testing.assert_close(losses["loss/total"], manual)


def test_inactive_heads_and_unread_levels_get_no_gradient():
    model, losses = _far_losses(active=NEAR)
    losses["loss/total"].backward()
    for name, p in model.named_parameters():
        if name.startswith(
            ("heads.10.", "heads.20.", "context_encoder.stages.1.", "context_encoder.stages.2.")
        ):
            assert p.grad is None, name
        elif name.startswith(("heads.1.", "context_encoder.stages.0.")):
            assert p.grad is not None, name


def test_no_active_horizon_raises():
    with pytest.raises(ValueError, match="active"):
        _far_losses(active=[])
    with pytest.raises(ValueError, match="active"):
        _far_losses(active=[7])  # not a built horizon
