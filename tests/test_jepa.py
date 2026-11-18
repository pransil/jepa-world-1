"""JEPA core: target encoder, EMA, stop-grad, leakage at the model boundary, losses (SPEC §4-5)."""

from __future__ import annotations

import pytest
import torch
from tiny import NEAR, random_frames, tiny_cfg

from jepa_world.models.jepa import (
    JEPA,
    compute_losses,
    context_stack,
    latent_loss,
    target_stack,
    variance_guard,
)


def _model(cfg=None) -> JEPA:
    torch.manual_seed(0)
    return JEPA(cfg or tiny_cfg(), NEAR)


# ---------------------------------------------------------------- stacking


def test_context_stack_is_frames_before_and_at_t_in_time_order():
    frames = torch.arange(9).float().view(1, 9, 1, 1, 1).expand(1, 9, 3, 2, 2)
    ctx = context_stack(frames, 4)
    assert ctx.shape == (1, 12, 2, 2)
    assert ctx[0, :, 0, 0].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


@pytest.mark.parametrize("k", NEAR)
def test_target_stack_ends_at_t_plus_k(k):
    frames = torch.arange(9).float().view(1, 9, 1, 1, 1).expand(1, 9, 3, 2, 2)
    tgt = target_stack(frames, 4, k)
    t = 3  # index of time t when C = 4
    assert tgt[0, ::3, 0, 0].tolist() == [t + k - 3, t + k - 2, t + k - 1, t + k]


def test_too_few_frames_raises():
    model = _model()
    with pytest.raises(ValueError, match="frames"):
        model(random_frames(tiny_cfg())[:, :8])  # needs C + 5 = 9


# ---------------------------------------------------------------- shapes


def test_predictions_match_targets_for_every_horizon():
    out = _model()(random_frames(tiny_cfg()))
    assert sorted(out["pred"]) == NEAR == sorted(out["target"])
    for k in NEAR:
        assert out["pred"][k].shape == out["target"][k].shape == (2, 8, 16, 16)


def test_far_horizons_are_not_built_in_m3():
    with pytest.raises(NotImplementedError, match="M4"):
        JEPA(tiny_cfg(), [1, 10])


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
    # M3 heads read only L1, so the forward stops there and stages 1-2 (L2, L3) get no gradient
    # until M4's far heads read them.
    for name, p in model.named_parameters():
        if name.startswith(("heads.", "context_encoder.stages.0.")):
            assert p.grad is not None, name
        elif name.startswith("context_encoder."):
            assert p.grad is None, name


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
    model = _model()
    frames = random_frames(tiny_cfg())
    later = frames.clone()
    later[:, 4:] = torch.rand_like(later[:, 4:])  # every frame after t
    a, b = model(frames)["context"], model(later)["context"]
    for level in a:
        assert torch.equal(a[level], b[level])
    for k in NEAR:  # ...so no prediction can see them either
        assert torch.equal(model(frames)["pred"][k], model(later)["pred"][k])
    at_t = frames.clone()
    at_t[:, 3] = torch.rand_like(at_t[:, 3])  # frame t itself IS context
    assert not torch.equal(model(at_t)["context"]["L1"], a["L1"])


@pytest.mark.parametrize("k", NEAR)
def test_target_k_ignores_frames_after_t_plus_k(k):
    model = _model()
    frames = random_frames(tiny_cfg())
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
