"""loss.latent_reduce (SPEC §5.1, §5.4; docs/decisions/2026-11-19-latent-loss-normalization.md).

'sum' (the 2026-11-17 decisions, and what a config without the key means): every latent term is
summed over its level's channels and averaged over (batch, positions). 'mean': the same terms
divided by their level's channel count, the variance guard included. var_reduce is relative to that
scale, so w_var and var_reduce alone set every level's guard-to-latent ratio, under either setting.
"""

from __future__ import annotations

import pytest
import torch
from tiny import (
    ALL,
    CONFIGS,
    FAR,
    LEVEL,
    TINY_SHAPE,
    random_batch,
    random_frames,
    tiny_aux_cfg,
    tiny_cfg,
    tiny_far_cfg,
)

from jepa_world import train
from jepa_world.models.jepa import compute_losses, gaussian_nll, latent_loss
from jepa_world.models.losses import gaussian_nll_terms
from jepa_world.models.world import compute_world_losses
from jepa_world.utils.config import load_config

CH = {lv: shape[0] for lv, shape in TINY_SHAPE.items()}  # L1 8, L2 16, L3 32 channels
SUM = ("loss.latent_reduce=sum",)
MEAN = ("loss.latent_reduce=mean",)
TD = {20: "L2", 10: "L1"}  # top-down: horizon -> the finer level its branch predicts
REL = {"rtol": 1e-5, "atol": 1e-7}  # ratios and totals: a different rounding order


def _model(cfg):
    torch.manual_seed(0)
    return train.build_model(cfg)


def _both(make=tiny_far_cfg, *more):
    """One model and one batch, scored under sum and under mean."""
    s, m = make(*SUM, *more), make(*MEAN, *more)
    model = _model(s)
    batch = random_batch(s) if "goal" in s["train"]["stages"] else random_frames(s)
    return compute_world_losses(model, batch, s["loss"]), compute_world_losses(
        model, batch, m["loss"]
    )


def _grads(cfg, frames):
    model = _model(cfg)
    compute_world_losses(model, frames, cfg["loss"])["loss/total"].backward()
    return {n: p.grad for n, p in model.named_parameters() if p.grad is not None}


# ---------------------------------------------------------------- sum: exactly as before


def test_base_defaults_to_mean_with_spec_weights():
    """Flipped 2026-11-19 after experiments/2026-11-19-latent-mean-collapse-check.md passed. The
    guard stays sum x25 relative to the latent terms, and the aux weights stay SPEC's, read
    relative to L3's latent terms (test_fixed_aux_weights_move_with_l3)."""
    loss = load_config(CONFIGS / "base.yaml")["loss"]
    assert loss["latent_reduce"] == "mean"
    assert (loss["var_reduce"], loss["w_var"]) == ("sum", 25.0)
    assert (loss["w_goal"], loss["w_pix"], loss["pix_share"]) == (0.5, 0.05, 0.10)


@pytest.mark.parametrize(
    "name",
    [
        "exp_first_real_run",
        "exp_guard_mean",
        "m3_near_dev",
        "m3_near_synthetic",
        "m4_far_synthetic",
        "m4_far_dev",
        "m4_collapse_baseline",
        "m4_collapse_both",
        "m4_collapse_guardsum",
        "m4_collapse_layernorm",
        "m7_pixel_share01",
        "m7_pixel_share10",
    ],
)
def test_configs_with_recorded_sum_runs_keep_sum(name):
    """Their runs trained with channel-summed latents, at w_goal 0.5 where there is a goal head."""
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg["loss"]["latent_reduce"] == "sum"
    assert cfg["loss"]["w_goal"] == 0.5


@pytest.mark.parametrize("name", ["exp_headline", "exp_headline_48k", "m7_obstacles"])
def test_the_m7_reference_chain_inherits_the_new_default(name):
    """docs/decisions/2026-11-19-m7-ablations-at-48k.md: re-run at base.yaml's defaults. The whole
    loss block, so no key of it can be pinned away from them; test_base_defaults_to_mean_with_spec_
    weights pins what those defaults are."""
    base = load_config(CONFIGS / "base.yaml")["loss"]
    assert load_config(CONFIGS / f"{name}.yaml")["loss"] == base


def test_sum_is_bitwise_the_channel_sum_formulas():
    """The pre-2026-11-19 expressions, verbatim: no rounding change on the default path."""
    cfg = tiny_far_cfg(*SUM)
    model = _model(cfg)
    frames = random_frames(cfg)
    losses = compute_world_losses(model, frames, cfg["loss"])
    out = model(frames)
    for k in ALL:
        err2 = (out["target"][k] - out["pred"][k]).square()
        expected = (
            gaussian_nll_terms(err2, out["logvar"][k]).sum(dim=1).mean()
            if k in FAR
            else err2.sum(dim=1).mean()
        )
        torch.testing.assert_close(losses[f"loss/k{k}"], expected, rtol=0, atol=0, msg=f"k{k}")
        if k in FAR:
            sq = err2.sum(dim=1).mean()
            torch.testing.assert_close(losses[f"sqerr/k{k}"], sq, rtol=0, atol=0)


def test_a_config_without_the_key_is_bitwise_sum():
    """Run configs saved before 2026-11-19 (checkpoints, eval, latent_relerr.py) have no key."""
    cfg = tiny_far_cfg(*SUM)
    old = tiny_far_cfg(*SUM)
    del old["loss"]["latent_reduce"]
    frames = random_frames(cfg)
    a = compute_world_losses(_model(cfg), frames, cfg["loss"])
    b = compute_world_losses(_model(old), frames, old["loss"])
    assert set(a) == set(b)
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0, equal_nan=True, msg=k)
    ga, gb = _grads(cfg, frames), _grads(old, frames)
    assert ga.keys() == gb.keys()
    for n in ga:
        torch.testing.assert_close(ga[n], gb[n], rtol=0, atol=0, msg=n)


# ---------------------------------------------------------------- mean: per channel, per level


def test_mean_divides_each_level_term_by_its_channel_count():
    s, m = _both()
    for k in ALL:
        c = CH[LEVEL[k]]
        torch.testing.assert_close(m[f"loss/k{k}"], s[f"loss/k{k}"] / c, rtol=0, atol=0)
    for k in FAR:
        c = CH[LEVEL[k]]
        torch.testing.assert_close(m[f"sqerr/k{k}"], s[f"sqerr/k{k}"] / c, rtol=0, atol=0)
    for lv, c in CH.items():  # each level's guard moves with its latent terms
        torch.testing.assert_close(m[f"var/{lv}"], s[f"var/{lv}"] / c, rtol=0, atol=0)
    torch.testing.assert_close(m["loss/var"], m["var/L1"] + m["var/L2"] + m["var/L3"])


@pytest.mark.parametrize("var_reduce", ["sum", "mean"])
def test_every_level_keeps_its_guard_to_latent_ratio(var_reduce):
    """The M4 far-collapse fix is a per-level balance (guard x w_var against that level's latent
    terms: docs/decisions/2026-11-17-far-collapse-guard-sum.md). Both sides move by 1/C, so the
    ratio is var_reduce's (and w_var's) alone: sum x25 stays sum x25, the mean x1 ablation stays
    mean x1."""
    s, m = _both(tiny_far_cfg, f"loss.var_reduce={var_reduce}")
    reads = {lv: [k for k in ALL if LEVEL[k] == lv] for lv in CH}
    for lv, ks in reads.items():
        ratio_s = s[f"var/{lv}"] / sum(s[f"loss/k{k}"] for k in ks)
        ratio_m = m[f"var/{lv}"] / sum(m[f"loss/k{k}"] for k in ks)
        torch.testing.assert_close(ratio_m, ratio_s, **REL, msg=lv)


def test_scale_free_diagnostics_are_bitwise_unchanged():
    """relerr is a ratio of two channel sums, logvar and emb_std are per-element means: none
    depends on the reduction, so runs under either stay comparable on them."""
    s, m = _both()
    keys = [k for k in s if k.startswith(("relerr/", "logvar/", "emb_std/"))]
    assert len(keys) == len(ALL) + len(FAR) + len(CH)  # relerr/k, logvar/k (far), emb_std/L
    for k in keys:
        torch.testing.assert_close(m[k], s[k], rtol=0, atol=0, equal_nan=True, msg=k)


def test_on_a_single_level_mean_is_a_uniform_rescale_of_sum():
    """Near only (L1): the whole total and every gradient are the sum run's divided by C(L1). With
    Adam that is (up to its eps) no change at all, so what mean changes is the balance between
    levels, and between the latent terms and the aux terms."""
    s_cfg, m_cfg = tiny_cfg(*SUM), tiny_cfg(*MEAN)
    frames = random_frames(s_cfg)
    s = compute_world_losses(_model(s_cfg), frames, s_cfg["loss"])
    m = compute_world_losses(_model(m_cfg), frames, m_cfg["loss"])
    torch.testing.assert_close(m["loss/total"], s["loss/total"] / CH["L1"], **REL)
    gs, gm = _grads(s_cfg, frames), _grads(m_cfg, frames)
    assert gs.keys() == gm.keys() and gs
    for n in gs:
        torch.testing.assert_close(gm[n], gs[n] / CH["L1"], rtol=1e-4, atol=1e-9, msg=n)


def test_top_down_terms_use_their_target_levels_channels():
    s, m = _both(tiny_far_cfg, "model.top_down=true")
    for k, lv in TD.items():
        torch.testing.assert_close(m[f"loss/td/k{k}"], s[f"loss/td/k{k}"] / CH[lv], rtol=0, atol=0)
        torch.testing.assert_close(
            m[f"sqerr/td/k{k}"], s[f"sqerr/td/k{k}"] / CH[lv], rtol=0, atol=0
        )
        torch.testing.assert_close(m[f"relerr/td/k{k}"], s[f"relerr/td/k{k}"], rtol=0, atol=0)


def test_goal_and_pixel_terms_do_not_depend_on_it():
    """latent_reduce is a JEPA-core switch: the aux losses are their own reductions. The pixel
    share is a share of whatever the rest is, so it still hits its target."""
    s, m = _both(tiny_aux_cfg)
    aux = [k for k in s if k.startswith(("loss/goal", "goalerr/", "goalratio/", "loss/pix"))]
    assert "loss/goal" in aux and "loss/pix" in aux
    for k in aux:
        torch.testing.assert_close(m[k], s[k], rtol=0, atol=0, equal_nan=True, msg=k)
    for x in (s, m):
        assert x["pix_share"].item() == pytest.approx(0.10, rel=1e-5)
    assert m["w_pix_eff"] < s["w_pix_eff"]  # |rest| shrank, so the share's weight did too


@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_fixed_aux_weights_move_with_l3(reduce):
    """Both aux heads read L3, so w_goal and a fixed w_pix are relative to L3's latent terms: under
    mean they are divided by C(L3), as those terms are. SPEC's 0.5 / 0.05 keep their meaning, and
    the M7 reference can't drift by --set loss.latent_reduce alone."""
    cfg = tiny_aux_cfg(f"loss.latent_reduce={reduce}", "loss.pix_share=null")
    model = _model(cfg)
    batch = random_batch(cfg)
    on = compute_world_losses(model, batch, cfg["loss"])
    off = compute_world_losses(model, batch, cfg["loss"], aux_active=())
    scale = 1 / CH["L3"] if reduce == "mean" else 1.0
    w = cfg["loss"]
    want = scale * (w["w_goal"] * on["loss/goal"] + w["w_pix"] * on["loss/pix"])
    torch.testing.assert_close(on["loss/total"] - off["loss/total"], want, **REL)
    torch.testing.assert_close(on["w_pix_eff"], torch.tensor(scale * w["w_pix"]))


def test_public_helpers_take_the_reduction():
    g = torch.Generator().manual_seed(0)
    p, t, lv = (torch.randn(2, 6, 3, 3, generator=g) for _ in range(3))
    assert latent_loss(p, t) == latent_loss(p, t, reduce="sum")
    torch.testing.assert_close(latent_loss(p, t, reduce="mean"), (t - p).square().mean())
    nll = gaussian_nll_terms((t - p).square(), lv)
    torch.testing.assert_close(gaussian_nll(p, lv, t, reduce="mean"), nll.mean())
    assert gaussian_nll(p, lv, t) == gaussian_nll(p, lv, t, reduce="sum")
    with pytest.raises(ValueError, match="latent_reduce"):
        latent_loss(p, t, reduce="max")


def test_mean_trains_and_stays_finite():
    cfg = tiny_far_cfg(*MEAN)
    model = _model(cfg)
    opt = train.build_optimizer(model, cfg)
    frames = random_frames(cfg)
    first = None
    for step in range(5):
        losses = train.train_step(model, opt, frames, cfg, step=step)
        assert all(torch.isfinite(v) for k, v in losses.items() if not k.startswith("relerr")), step
        first = losses["loss/total"] if first is None else first
    assert losses["loss/total"] < first  # the same fixed batch: the loss goes down


# ---------------------------------------------------------------- config checks


def test_an_unknown_reduction_is_refused_before_the_forward():
    cfg = tiny_far_cfg("loss.latent_reduce=max")
    with pytest.raises(ValueError, match="latent_reduce"):
        train.check_config(cfg)
    model = _model(tiny_far_cfg())
    calls = []
    model.register_forward_pre_hook(lambda *a: calls.append(1))
    with pytest.raises(ValueError, match="latent_reduce"):
        compute_losses(model, random_frames(cfg), cfg["loss"])
    assert calls == []


def test_collapse_check_config_is_the_sum_x25_run_with_only_the_reduction_changed():
    """configs/exp_latent_mean.yaml against configs/exp_first_real_run.yaml (the sum x25 arm of
    experiments/2026-11-18-guard-comparison.md): near + far, 8000 steps, far after 2000."""
    ref = load_config(CONFIGS / "exp_first_real_run.yaml")
    cfg = load_config(CONFIGS / "exp_latent_mean.yaml")
    assert cfg["loss"]["latent_reduce"] == "mean" and ref["loss"]["latent_reduce"] == "sum"
    for c in (cfg, ref):  # the guard: sum x25 relative to the latent terms, in both arms
        assert (c["loss"]["var_reduce"], c["loss"]["w_var"]) == ("sum", 25.0)
    changed = {"latent_reduce"}
    assert {k: v for k, v in cfg["loss"].items() if k not in changed} == {
        k: v for k, v in ref["loss"].items() if k not in changed
    }
    assert {k: v for k, v in cfg.items() if k != "loss"} == {
        k: v for k, v in ref.items() if k != "loss"
    }
    assert cfg["train"]["stages"] == ["near", "far"] and cfg["train"]["steps"] == 8000
    train.check_config(load_config(CONFIGS / "exp_latent_mean.yaml", ["data.source=synthetic"]))
