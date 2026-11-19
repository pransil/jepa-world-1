"""WorldModel: the JEPA core plus the auxiliary goal head and pixel decoder (SPEC §4.1, §4.6, §4.7,
§5.2-§5.5). The aux heads read only the context levels and never the ground truth. Each term is
logged every step and joins loss/total only while its stage is active."""

from __future__ import annotations

import pytest
import torch
from tiny import NEAR, random_batch, random_frames, tiny_aux_cfg, tiny_cfg

from jepa_world import train
from jepa_world.models.goal import displacement_targets, goal_nll
from jepa_world.models.jepa import JEPA
from jepa_world.models.pixel import pixel_l1
from jepa_world.models.world import WorldModel, compute_world_losses, pix_share_weight

GOAL_KS = [1, 3, 5]  # tiny_aux_cfg's model.goal_horizons


def _model(cfg) -> WorldModel:
    torch.manual_seed(0)
    return train.build_model(cfg)


def _losses(model, batch, cfg, **kw):
    return compute_world_losses(model, batch, cfg["loss"], **kw)


# ---------------------------------------------------------------- construction


def test_aux_heads_exist_only_for_their_stages():
    m = _model(tiny_aux_cfg())
    assert isinstance(m, WorldModel) and m.goal is not None and m.pixel is not None
    assert m.goal.horizons == tuple(GOAL_KS)
    m = _model(tiny_cfg())  # near only
    assert m.goal is None and m.pixel is None
    m = _model(tiny_aux_cfg("train.stages=[near, pixel]"))
    assert m.goal is None and m.pixel is not None


def test_a_jepa_checkpoint_loads_into_a_world_model_without_aux_heads():
    """WorldModel subclasses JEPA, so pre-M5 checkpoints keep their keys and load strictly."""
    cfg = tiny_cfg()
    torch.manual_seed(1)
    jepa = JEPA(cfg, NEAR)
    world = _model(cfg)
    world.load_state_dict(jepa.state_dict())  # strict=True by default
    for a, b in zip(world.parameters(), jepa.parameters(), strict=True):
        assert torch.equal(a, b)


def test_aux_parameters_add_only_goal_and_pixel_keys():
    cfg = tiny_aux_cfg()
    jepa_keys = set(JEPA(cfg, NEAR).state_dict())
    extra = set(_model(cfg).state_dict()) - jepa_keys
    assert extra and all(k.startswith(("goal.", "pixel.")) for k in extra)


def test_trainable_parameters_include_aux_heads_but_never_the_target():
    m = _model(tiny_aux_cfg())
    ids = {id(p) for p in m.trainable_parameters()}
    assert {id(p) for p in m.goal.parameters()} <= ids
    assert {id(p) for p in m.pixel.parameters()} <= ids
    assert not ids & {id(p) for p in m.target_encoder.parameters()}


def test_config_options_reach_the_heads():
    m = _model(tiny_aux_cfg("model.goal_l2=true", "model.unet_skips=false"))
    assert m.goal.l2_ch == 16  # tiny L2 channels
    assert m.pixel.skips is False
    assert _model(tiny_aux_cfg()).goal.l2_ch == 0


# ---------------------------------------------------------------- terms and total


def test_every_aux_term_is_logged_and_finite():
    cfg = tiny_aux_cfg()
    losses = _losses(_model(cfg), random_batch(cfg), cfg)
    want = {"loss/goal", "loss/pix", "loss/total"}
    want |= {f"{p}/k{k}" for k in GOAL_KS for p in ("loss/goal", "goalerr", "goalratio")}
    want |= {f"loss/k{k}" for k in NEAR}
    assert want <= set(losses), want - set(losses)
    for name, v in losses.items():
        assert v.shape == () and torch.isfinite(v), name


def test_inactive_aux_terms_are_logged_but_left_out_of_the_total():
    cfg = tiny_aux_cfg()
    model, batch = _model(cfg), random_batch(cfg)
    on = _losses(model, batch, cfg)
    off = _losses(model, batch, cfg, aux_active=())
    assert {"loss/goal", "loss/pix"} <= set(off)
    w = cfg["loss"]
    # w_goal is relative to L3's latent terms: / C(L3) under base.yaml's latent_reduce: mean
    # (docs/decisions/2026-11-19-latent-loss-normalization.md), and logged as w_goal_eff
    c3 = cfg["model"]["levels"]["L3"]["channels"]
    w_goal = w["w_goal"] / c3 if w["latent_reduce"] == "mean" else w["w_goal"]
    assert on["w_goal_eff"].item() == pytest.approx(w_goal) and off["w_goal_eff"] == 0
    assert off["w_pix_eff"] == 0
    torch.testing.assert_close(  # w_pix_eff: base.yaml's 10% share weight (or w_pix if null)
        on["loss/total"] - off["loss/total"],
        w_goal * on["loss/goal"] + on["w_pix_eff"] * on["loss/pix"],
    )
    goal_only = _losses(model, batch, cfg, aux_active=["goal"])
    torch.testing.assert_close(
        goal_only["loss/total"] - off["loss/total"], w_goal * on["loss/goal"]
    )


def test_goal_loss_is_the_per_horizon_nll_of_the_displacement():
    cfg = tiny_aux_cfg()
    model, batch = _model(cfg), random_batch(cfg)
    losses = _losses(model, batch, cfg)
    mean, logvar = model(batch["frames"])["goal"]
    target = displacement_targets(batch["ball_pos"], cfg["model"]["context_frames"], GOAL_KS)
    nll = goal_nll(mean, logvar, target)
    for j, k in enumerate(GOAL_KS):
        torch.testing.assert_close(losses[f"loss/goal/k{k}"], nll[j])
        err = (mean[:, j] - target[:, j]).norm(dim=-1).mean()
        torch.testing.assert_close(losses[f"goalerr/k{k}"], err)
        torch.testing.assert_close(
            losses[f"goalratio/k{k}"], err / target[:, j].norm(dim=-1).mean()
        )
    torch.testing.assert_close(losses["loss/goal"], nll.sum())


def test_pixel_loss_targets_the_frame_at_t_plus_1():
    cfg = tiny_aux_cfg()
    model, batch = _model(cfg), random_batch(cfg)
    c = cfg["model"]["context_frames"]
    pred = model(batch["frames"])["pixel"]
    torch.testing.assert_close(
        _losses(model, batch, cfg)["loss/pix"], pixel_l1(pred, batch["frames"][:, c])
    )


def test_a_goal_stage_without_ball_pos_raises():
    cfg = tiny_aux_cfg()
    model = _model(cfg)
    with pytest.raises(ValueError, match="ball_pos"):
        _losses(model, random_frames(cfg), cfg)
    with pytest.raises(ValueError, match="ball_pos"):
        _losses(model, {"frames": random_frames(cfg)}, cfg)


def test_frames_only_batches_work_without_a_goal_stage():
    for cfg in (tiny_cfg(), tiny_aux_cfg("train.stages=[near, pixel]")):
        model = _model(cfg)
        a = _losses(model, random_frames(cfg), cfg)
        b = _losses(model, {"frames": random_frames(cfg)}, cfg)
        assert a.keys() == b.keys()
        torch.testing.assert_close(a["loss/total"], b["loss/total"])


def test_unbuilt_aux_stage_cannot_be_activated():
    cfg = tiny_aux_cfg("train.stages=[near, pixel]")
    with pytest.raises(ValueError, match="goal"):
        _losses(_model(cfg), random_frames(cfg), cfg, aux_active=["goal"])


# ---------------------------------------------------------------- gradients


def _grads(model, losses):
    model.zero_grad(set_to_none=True)
    losses["loss/total"].backward()


def test_inactive_aux_heads_get_no_gradient():
    cfg = tiny_aux_cfg()
    model = _model(cfg)
    _grads(model, _losses(model, random_batch(cfg), cfg, aux_active=()))
    assert all(p.grad is None for p in model.goal.parameters())
    assert all(p.grad is None for p in model.pixel.parameters())


def test_inactive_aux_heads_keep_no_autograd_graph():
    """Before its stage starts, an aux head runs only to be logged: no graph is kept for it."""
    cfg = tiny_aux_cfg()
    model, frames = _model(cfg), random_frames(cfg)
    off = model(frames, aux_active=())
    assert not off["pixel"].requires_grad and not off["goal"][0].requires_grad
    on = model(frames)
    assert on["pixel"].requires_grad and on["goal"][0].requires_grad


def test_active_goal_head_trains_l3_even_with_near_heads_only():
    """Near heads read only L1, so without the goal head the L2/L3 encoder stages get nothing."""
    cfg = tiny_aux_cfg()
    model = _model(cfg)
    _grads(model, _losses(model, random_batch(cfg), cfg, aux_active=["goal"]))
    assert all(p.grad is not None for p in model.goal.parameters())
    assert all(p.grad is None for p in model.pixel.parameters())
    l3 = model.context_encoder.stages[2]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in l3.parameters())


def test_active_pixel_decoder_trains_every_encoder_level():
    cfg = tiny_aux_cfg()
    model = _model(cfg)
    _grads(model, _losses(model, random_batch(cfg), cfg, aux_active=["pixel"]))
    for stage in model.context_encoder.stages:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters())


def test_no_gradient_reaches_the_target_encoder():
    cfg = tiny_aux_cfg()
    model = _model(cfg)
    _grads(model, _losses(model, random_batch(cfg), cfg))
    assert all(p.grad is None for p in model.target_encoder.parameters())


# ---------------------------------------------------------------- leakage


def test_aux_heads_read_only_the_context_frames():
    """Changing every frame after t (index C-1) changes neither head's output."""
    cfg = tiny_aux_cfg()
    model = _model(cfg)
    c = cfg["model"]["context_frames"]
    frames = random_frames(cfg)
    future = frames.clone()
    future[:, c:] = torch.rand_like(future[:, c:])
    a, b = model(frames), model(future)
    torch.testing.assert_close(a["pixel"], b["pixel"], rtol=0, atol=0)
    for x, y in zip(a["goal"], b["goal"], strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize("share", ["base", "null"])
def test_the_model_never_receives_ground_truth(share):
    """ball_pos is a label: forward gets the frames tensor only, and changing ball_pos changes
    only the goal terms (data/DESIGN.md: GT after index C-1 is the future)."""
    cfg = tiny_aux_cfg(*(["loss.pix_share=null"] if share == "null" else []))
    model = _model(cfg)
    seen = []
    model.register_forward_pre_hook(
        lambda mod, args, kwargs: seen.append((args, kwargs)), with_kwargs=True
    )
    batch = random_batch(cfg)
    a = _losses(model, batch, cfg)
    assert len(seen) == 1
    args, kwargs = seen[0]
    tensors = [x for x in (*args, *kwargs.values()) if isinstance(x, torch.Tensor)]
    assert len(tensors) == 1 and tensors[0] is batch["frames"]

    moved = {**batch, "ball_pos": batch["ball_pos"] + 3.0 * torch.randn_like(batch["ball_pos"])}
    b = _losses(model, moved, cfg)
    goal_terms = {k for k in a if k.startswith(("loss/goal", "goalerr", "goalratio"))}
    # loss/total includes the goal loss, and so does pix_share (a ratio to the rest of
    # loss/total). Under a share (base.yaml's default) w_pix_eff = s|rest|/loss/pix does too;
    # under the fixed w_pix it must not move.
    shared = {"loss/total", "pix_share"} | ({"w_pix_eff"} if share == "base" else set())
    for k in set(a) - goal_terms - shared:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, msg=k)
    assert not torch.equal(a["loss/goal"], b["loss/goal"])


def test_a_bad_loss_config_fails_before_the_forward():
    cfg = tiny_aux_cfg()
    model = _model(cfg)
    calls = []
    model.register_forward_pre_hook(lambda *a: calls.append(1))
    bad = {**cfg["loss"], "latent_distance": "cosine"}
    with pytest.raises(ValueError, match="latent_distance"):
        compute_world_losses(model, random_batch(cfg), bad)
    assert calls == []


def test_goalratio_is_nan_when_the_ball_does_not_move():
    """Persistence error 0 (horizon 0, or a ball at rest): the ratio is undefined, not 1e12."""
    cfg = tiny_aux_cfg("model.goal_horizons=[0, 1]")
    batch = random_batch(cfg)
    still = {**batch, "ball_pos": torch.ones_like(batch["ball_pos"])}
    losses = _losses(_model(cfg), still, cfg)
    assert torch.isnan(losses["goalratio/k0"]) and torch.isnan(losses["goalratio/k1"])
    assert torch.isfinite(losses["goalerr/k1"]) and torch.isfinite(losses["loss/goal"])
    moving = _losses(_model(cfg), batch, cfg)
    assert torch.isnan(moving["goalratio/k0"]) and torch.isfinite(moving["goalratio/k1"])


def test_a_pre_m5_run_config_still_builds_and_loads_its_checkpoint():
    """Saved run configs from before M5 lack the aux keys: building without aux stages must not
    read them (experiments/probe_embeddings.py and M6 eval rebuild old runs this way)."""
    cfg = tiny_cfg()
    for key in (
        "goal_horizons",
        "goal_l2",
        "goal_hidden",
        "goal_logvar_bound",
        "pix_widths",
        "unet_skips",
    ):
        del cfg["model"][key]
    cfg["train"]["stage_start"] = {"near": 0, "far": 0}  # what pre-M5 run configs carry
    torch.manual_seed(1)
    old = JEPA(cfg, NEAR)
    train.check_config(cfg)
    world = train.build_model(cfg)
    world.load_state_dict(old.state_dict())
    assert world.goal is None and world.pixel is None


# ---------------------------------------------------------------- pixel target share (M7)
# docs/decisions/2026-11-18-pixel-ablation-target-share.md


def _share_cfg(share="0.01", *more):
    return tiny_aux_cfg(f"loss.pix_share={share}", *more)


def test_pix_share_null_is_bitwise_the_fixed_weight():
    """Off (null, or a pre-change config with no key): loss/total is exactly the fixed-weight
    sum, and every gradient is bitwise equal."""
    # base.yaml's defaults since 2026-11-19 are a share and per-channel latents: the pre-change
    # loss is pix_share null with channel-summed latents (the fixed weights' mean-mode scaling is
    # tested in test_latent_reduce.py)
    cfg = tiny_aux_cfg("loss.pix_share=null", "loss.latent_reduce=sum")
    assert cfg["loss"]["pix_share"] is None
    old = tiny_aux_cfg()
    del old["loss"]["pix_share"]  # saved configs from before these keys existed
    del old["loss"]["latent_reduce"]
    batch = random_batch(cfg)
    grads, totals = [], []
    for c in (cfg, old):
        model = _model(c)
        losses = _losses(model, batch, c)
        rest = _losses(model, batch, c, aux_active=["goal"])["loss/total"]
        assert torch.equal(losses["loss/total"], rest + c["loss"]["w_pix"] * losses["loss/pix"])
        model.zero_grad(set_to_none=True)
        losses["loss/total"].backward()
        totals.append(losses["loss/total"])
        grads.append([p.grad for p in model.parameters()])
    assert torch.equal(totals[0], totals[1])
    for a, b in zip(*grads, strict=True):
        assert (a is None and b is None) or torch.equal(a, b)
    # The decoder's gradients against a hand-built w_pix * L_pix: a change to the pixel term too
    # small to move loss/total (absorbed by rounding) still scales every decoder gradient. (Only
    # the decoder: the encoder's gradients would sum two separate forwards in another order.)
    model = _model(cfg)
    decoder = {id(p) for p in model.pixel.parameters()}
    want = [g for p, g in zip(model.parameters(), grads[0], strict=True) if id(p) in decoder]
    _grads(model, {"loss/total": cfg["loss"]["w_pix"] * _losses(model, batch, cfg)["loss/pix"]})
    for p, g in zip(model.pixel.parameters(), want, strict=True):
        assert torch.equal(p.grad, g)


def test_fixed_weight_logs_its_weight_and_share():
    cfg = tiny_aux_cfg("loss.pix_share=null", "loss.latent_reduce=sum")  # w_pix as written
    model, batch = _model(cfg), random_batch(cfg)
    on = _losses(model, batch, cfg)
    rest = _losses(model, batch, cfg, aux_active=["goal"])["loss/total"]
    torch.testing.assert_close(on["w_pix_eff"], torch.tensor(cfg["loss"]["w_pix"]))
    torch.testing.assert_close(on["pix_share"], cfg["loss"]["w_pix"] * on["loss/pix"] / rest.abs())


def test_pix_share_holds_the_pixel_term_at_the_target_share():
    cfg = _share_cfg("0.01")
    model, batch = _model(cfg), random_batch(cfg)
    on = _losses(model, batch, cfg)
    rest = _losses(model, batch, cfg, aux_active=["goal"])["loss/total"]
    torch.testing.assert_close(on["w_pix_eff"] * on["loss/pix"], 0.01 * rest.abs())
    torch.testing.assert_close(on["loss/total"], rest + 0.01 * rest.abs())
    torch.testing.assert_close(on["pix_share"], torch.tensor(0.01))
    assert on["w_pix_eff"] != cfg["loss"]["w_pix"]  # w_pix is ignored when a share is set


def test_pix_share_weight_carries_no_gradient():
    """w_eff is detached on both sides. If it carried L_pix's gradient, w_eff * L_pix = s|L_rest|
    would not depend on the decoder at all, and the decoder would get no gradient. If it carried
    L_rest's, every other parameter's gradient would be scaled by (1 + s)."""
    cfg = _share_cfg("0.10")
    model, batch = _model(cfg), random_batch(cfg)
    _grads(model, _losses(model, batch, cfg))
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.pixel.parameters())
    with_pix = {n: p.grad.clone() for n, p in model.named_parameters() if n.startswith("heads.")}
    _grads(model, _losses(model, batch, cfg, aux_active=["goal"]))
    for n, g in with_pix.items():
        torch.testing.assert_close(model.get_parameter(n).grad, g, rtol=0, atol=0, msg=n)


def test_pix_share_adds_nothing_before_the_pixel_stage_starts():
    cfg = _share_cfg("0.10")
    model, batch = _model(cfg), random_batch(cfg)
    off = _losses(model, batch, cfg, aux_active=["goal"])
    goal_only = _losses(model, batch, tiny_aux_cfg(), aux_active=["goal"])
    assert torch.equal(off["loss/total"], goal_only["loss/total"])
    assert off["w_pix_eff"] == 0 and off["pix_share"] == 0
    assert torch.isfinite(off["loss/pix"])  # still logged


@pytest.mark.parametrize(
    ("rest", "pix"),
    [(500.0, 0.03), (-40.0, 0.03), (0.0, 0.03), (500.0, 0.0), (500.0, 1e-30), (0.0, 0.0)],
)
def test_share_weight_is_finite_and_never_overshoots(rest, pix):
    """|L_rest| (NLL terms can make it negative), and a floor on L_pix: the share can only fall
    short of the target, never exceed it."""
    rest_t, pix_t = torch.tensor(rest), torch.tensor(pix)
    w = pix_share_weight(0.1, rest_t, pix_t)  # check_pix_share refuses s = 0 upstream
    assert torch.isfinite(w) and w >= 0
    assert w * pix_t <= 0.1 * abs(rest) * (1 + 1e-6)
    if pix >= 1e-6:
        torch.testing.assert_close(w * pix_t, torch.tensor(0.1 * abs(rest)))


def test_share_weight_is_detached():
    rest = torch.tensor(300.0, requires_grad=True)
    pix = torch.tensor(0.02, requires_grad=True)
    assert not pix_share_weight(0.01, rest, pix).requires_grad


@pytest.mark.parametrize("bad", ["-0.1", "0", "1.0", "1.5", "true", "'0.01'"])
def test_bad_pix_share_is_refused(bad):
    """(0, 1) only. 0 would silently drop the pixel term and ignore w_pix: use w_pix: 0 for that."""
    cfg = _share_cfg(bad)
    with pytest.raises(ValueError, match="pix_share"):
        _losses(_model(cfg), random_batch(cfg), cfg)
