"""Top-down prediction (M7 §4; SPEC §9.2; docs/decisions/2026-11-17-top-down-targets.md).

Each far head also predicts the next-finer target at its own horizon, from the coarse context level
only: k = 20 (L3) -> target L2(t+20), k = 10 (L2) -> target L1(t+10). It uses the far heads'
Gaussian NLL, weighted by loss.w_td, and is active exactly when its far horizon is. Off by default,
and then nothing about the model changes (state-dict keys, outputs, losses).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
from tiny import CONFIGS, FAR, TINY_SHAPE, random_batch, random_frames, tiny_cfg, tiny_far_cfg

from jepa_world import train
from jepa_world.models.heads import TopDownHead
from jepa_world.models.jepa import gaussian_nll, normalize_target, target_stack
from jepa_world.models.world import compute_world_losses
from jepa_world.utils.config import load_config

TD = {20: "L2", 10: "L1"}  # horizon -> the finer level its branch predicts
TD_KEYS = [f"{p}/td/k{k}" for k in FAR for p in ("loss", "sqerr", "logvar", "relerr")]


def td_cfg(*overrides: str) -> dict:
    return tiny_far_cfg("model.top_down=true", *overrides)


def _model(cfg):
    torch.manual_seed(0)
    return train.build_model(cfg)


def _losses(model, cfg, active=None, seed=0):
    return compute_world_losses(model, random_frames(cfg, seed=seed), cfg["loss"], active=active)


# ---------------------------------------------------------------- off by default: nothing changes


def test_base_config_has_top_down_off_and_w_td_half():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg["model"]["top_down"] is False and cfg["loss"]["w_td"] == 0.5


def test_off_adds_no_parameters_outputs_or_loss_terms():
    model = _model(tiny_far_cfg())
    assert not any(k.startswith("td_heads") for k in model.state_dict())
    out = model(random_frames(tiny_far_cfg()))
    assert not any(k.startswith("td_") for k in out)
    assert not any("/td/" in k for k in _losses(model, tiny_far_cfg()))


def test_a_config_from_before_m7_builds_the_same_model():
    """Run configs saved before this change have no model.top_down (e.g. the probes reload them)."""
    old = tiny_far_cfg()
    del old["model"]["top_down"], old["loss"]["w_td"]
    a, b = _model(old), _model(tiny_far_cfg())
    b.load_state_dict(a.state_dict())  # strict: identical keys
    torch.manual_seed(1)
    frames = random_frames(old)
    la = compute_world_losses(a, frames, old["loss"])
    lb = compute_world_losses(b, frames, tiny_far_cfg()["loss"])
    assert la.keys() == lb.keys()
    for k in la:
        torch.testing.assert_close(la[k], lb[k], rtol=0, atol=0, equal_nan=True, msg=k)


def test_on_adds_only_td_heads_keys_and_off_checkpoints_load_into_it():
    off, on = _model(tiny_far_cfg()), _model(td_cfg())
    extra = set(on.state_dict()) - set(off.state_dict())
    assert extra and all(k.startswith(("td_heads.20.", "td_heads.10.")) for k in extra)
    assert set(off.state_dict()) <= set(on.state_dict())
    missing, unexpected = on.load_state_dict(off.state_dict(), strict=False)
    assert unexpected == [] and set(missing) == extra


def test_top_down_without_far_heads_is_a_config_error():
    with pytest.raises(ValueError, match="top_down"):
        _model(tiny_cfg("model.top_down=true"))  # near only: there is no far head to extend


# ---------------------------------------------------------------- shapes and targets


def test_branch_outputs_match_the_finer_target_shapes():
    cfg = td_cfg()
    model = _model(cfg)
    out = model(random_frames(cfg))
    bound = cfg["model"]["far_logvar_bound"]
    for k, level in TD.items():
        shape = (2, *TINY_SHAPE[level])
        assert out["td_pred"][k].shape == out["td_logvar"][k].shape == shape
        assert out["td_target"][k].shape == shape
        assert out["td_logvar"][k].abs().max() < bound


def test_targets_are_the_target_encoders_finer_level_at_t_plus_k():
    cfg = td_cfg("loss.target_norm=layernorm")
    model = _model(cfg)
    frames = random_frames(cfg)
    out = model(frames)
    C = cfg["model"]["context_frames"]
    for k, level in TD.items():
        assert not out["td_target"][k].requires_grad
        with torch.no_grad():
            z = model.target_encoder(target_stack(frames, C, k), upto=level)[level]
        want = normalize_target(z, "layernorm")
        torch.testing.assert_close(out["td_target"][k], want, rtol=1e-5, atol=1e-6)


def test_the_branch_reads_only_its_far_heads_coarse_level():
    cfg = td_cfg()
    model = _model(cfg)
    out = model(random_frames(cfg))
    coarse = {20: "L3", 10: "L2"}
    for k, level in coarse.items():
        mean, logvar = model.td_heads[str(k)](out["context"][level])
        torch.testing.assert_close(mean, out["td_pred"][k], rtol=0, atol=0)
        torch.testing.assert_close(logvar, out["td_logvar"][k], rtol=0, atol=0)


def test_top_down_head_upsamples_by_its_scale():
    head = TopDownHead(
        16, 8, scale=4, logvar_bound=10.0, hidden=4, kernel=3, embed_dim=2, action_dim=0,
        norm_groups=2,
    )  # fmt: skip
    mean, logvar = head(torch.randn(2, 16, 2, 2))
    assert mean.shape == logvar.shape == (2, 8, 8, 8)


# ---------------------------------------------------------------- losses and gradients


def test_td_terms_are_logged_and_use_the_far_heads_nll():
    cfg = td_cfg()
    model = _model(cfg)
    frames = random_frames(cfg)
    losses = compute_world_losses(model, frames, cfg["loss"])
    for key in TD_KEYS:
        assert key in losses and torch.isfinite(losses[key]), key
    with torch.no_grad():
        out = model(frames)
    for k in FAR:
        want = gaussian_nll(out["td_pred"][k], out["td_logvar"][k], out["td_target"][k])
        torch.testing.assert_close(losses[f"loss/td/k{k}"].detach(), want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("far_active", [True, False])
def test_w_td_weights_the_terms_only_while_the_far_stage_is_active(far_active):
    active = [1, 2, 3, 4, 5, 10, 20] if far_active else [1, 2, 3, 4, 5]
    totals, terms = [], None
    for w in (0.0, 0.5):
        cfg = td_cfg(f"loss.w_td={w}")
        losses = _losses(_model(cfg), cfg, active=active)
        totals.append(losses["loss/total"].detach())
        terms = losses["loss/td/k10"].detach() + losses["loss/td/k20"].detach()
    expected = 0.5 * terms if far_active else torch.zeros(())
    torch.testing.assert_close(totals[1] - totals[0], expected, rtol=1e-5, atol=1e-5)


def test_errors_propagate_up_into_the_coarse_context_levels_and_never_the_target():
    cfg = td_cfg()
    model = _model(cfg)
    enc = model.context_encoder
    for k, coarse in {20: "L3", 10: "L2"}.items():
        model.zero_grad(set_to_none=True)
        _losses(model, cfg)[f"loss/td/k{k}"].backward()
        stage = enc.stages[enc.names.index(coarse)]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stage.parameters()), k
        assert all(p.grad is None for p in model.target_encoder.parameters())
        assert all(p.grad is None for p in model.td_heads[str(10 if k == 20 else 20)].parameters())


def test_an_inactive_far_stage_gives_the_branches_no_gradient():
    cfg = td_cfg()
    model = _model(cfg)
    _losses(model, cfg, active=[1, 2, 3, 4, 5])["loss/total"].backward()
    assert all(p.grad is None for p in model.td_heads.parameters())


def test_trainable_parameters_include_the_branches():
    model = _model(td_cfg())
    ids = {id(p) for p in model.trainable_parameters()}
    assert all(id(p) in ids for p in model.td_heads.parameters())


def test_a_few_steps_on_a_fixed_batch_lower_the_top_down_loss():
    cfg = td_cfg()
    model = _model(cfg)
    opt = train.build_optimizer(model, cfg)
    batch = random_frames(cfg)
    first = train.train_step(model, opt, batch, cfg)
    for _ in range(5):
        last = train.train_step(model, opt, batch, cfg)
    for k in FAR:
        assert last[f"loss/td/k{k}"] < first[f"loss/td/k{k}"], k


def test_a_run_logs_every_td_term_every_step(tmp_path):
    cfg = td_cfg()
    train.run(cfg, tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == cfg["train"]["steps"]
    for row in rows:
        assert all(key in row for key in TD_KEYS)


def test_world_model_with_aux_heads_still_gets_the_branches():
    cfg = td_cfg("train.stages=[near, far, goal, pixel]", "model.goal_horizons=[1, 3]",
                 "model.goal_hidden=4", "model.pix_widths=[4, 4]")  # fmt: skip
    model = _model(cfg)
    losses = compute_world_losses(model, random_batch(cfg), cfg["loss"])
    assert all(key in losses for key in TD_KEYS) and "loss/goal" in losses


# ---------------------------------------------------------------- MPS (local Mac only)

MPS_TD_STEP = """
import json, sys
sys.path.insert(0, "tests")
import jepa_world  # sets PYTORCH_ENABLE_MPS_FALLBACK before torch
import torch
from tiny import spec_cfg
from jepa_world import train
from jepa_world.bench import record_mps_fallbacks
from jepa_world.data.synthetic import moving_disc_batch

cfg = spec_cfg("train.batch_size=2", "train.stages=[near, far]", "model.top_down=true")
with record_mps_fallbacks() as fallbacks:
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = train.build_model(cfg).to(dev)
    opt = train.build_optimizer(model, cfg)
    frames = moving_disc_batch(2, 24, 64, generator=torch.Generator().manual_seed(0)).to(dev)
    losses = train.train_step(model, opt, frames, cfg)
    torch.mps.synchronize()
td = {k: bool(torch.isfinite(v).item()) for k, v in losses.items() if "/td/" in k}
print(json.dumps({"td_finite": td, "fallbacks": fallbacks}))
"""


@pytest.mark.mps
def test_top_down_step_on_mps_at_spec_shapes_is_finite_with_no_cpu_fallback():
    r = subprocess.run(
        [sys.executable, "-c", MPS_TD_STEP],
        cwd=CONFIGS.parent,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["fallbacks"] == []
    assert len(got["td_finite"]) == 8 and all(got["td_finite"].values())
