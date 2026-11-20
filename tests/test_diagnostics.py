"""A: the latent heads' "copy the current embedding" baseline (relerr/copy/k) and per-head encoder
gradient norms (gradnorm/enc/<term>/<level>). Both are diagnostics: neither may change training
(jepa_world/models/DESIGN.md, "Diagnostics")."""

from __future__ import annotations

import itertools
import json
import math

import pytest
import torch
from tiny import random_batch, tiny_aux_cfg, tiny_cfg

from jepa_world import train
from jepa_world.models.jepa import target_stack
from jepa_world.models.world import compute_world_losses

FULL = [
    "train.stages=[near, far, goal, pixel]",
    "train.stage_start={near: 0, far: 0, goal: 0, pixel: 0}",
]
TOP_DOWN = ["model.top_down=true"]


def full_cfg(*overrides: str) -> dict:
    """Every term there is: near, far, guard, goal, pixel (and top-down with TOP_DOWN)."""
    return tiny_aux_cfg(*FULL, *overrides)


def model_of(cfg: dict, seed: int = 0) -> train.WorldModel:
    torch.manual_seed(seed)
    return train.build_model(cfg)


# ---- the weighted terms of loss/total ----


@pytest.mark.parametrize("extra", [[], TOP_DOWN], ids=["plain", "top_down"])
@pytest.mark.parametrize(
    ("active", "aux"), [(None, None), ([1, 2, 3, 4, 5], ()), ([1, 2, 3, 4, 5, 10], ["goal"])]
)
def test_weighted_terms_sum_to_loss_total_bitwise(extra, active, aux):
    """terms[name] is each weighted term exactly as it enters loss/total, and only the active ones:
    their sum, in order, is loss/total bit for bit."""
    cfg = full_cfg(*extra)
    model = model_of(cfg)
    terms: dict[str, torch.Tensor] = {}
    losses = compute_world_losses(
        model, random_batch(cfg), cfg["loss"], active=active, aux_active=aux, terms=terms
    )
    total = losses["loss/total"].new_zeros(())
    for t in terms.values():
        total = total + t
    assert torch.equal(total, losses["loss/total"])
    act = set(model.horizons if active is None else active)
    want = {f"k{k}" for k in act} | {"var"}
    want |= {f"td/k{k}" for k in model.td_level if k in act}
    want |= {train.GRAD_TERMS_AUX[s] for s in (model.aux_stages if aux is None else aux)}
    assert set(terms) == want


def test_asking_for_terms_or_the_copy_baseline_changes_no_other_value():
    cfg = full_cfg()
    batch = random_batch(cfg)
    plain = compute_world_losses(model_of(cfg), batch, cfg["loss"])
    both = compute_world_losses(model_of(cfg), batch, cfg["loss"], terms={}, copy_baseline=True)
    assert set(both) - set(plain) == {f"relerr/copy/k{k}" for k in model_of(cfg).horizons}
    for k, v in plain.items():
        assert torch.equal(v, both[k]) or (v.isnan() and both[k].isnan()), k


# ---- the copy baseline ----


def test_copy_baseline_is_zero_when_nothing_moves():
    """A static clip: the stack ending at t is the stack ending at t+k, so copying is exact."""
    cfg = full_cfg()
    batch = random_batch(cfg)
    batch["frames"] = batch["frames"][:, :1].expand_as(batch["frames"]).contiguous()
    losses = compute_world_losses(model_of(cfg), batch, cfg["loss"], copy_baseline=True)
    for k in model_of(cfg).horizons:
        assert losses[f"relerr/copy/k{k}"].item() == 0.0, k


def test_copy_baseline_is_the_relerr_of_the_target_embedding_at_t():
    """relerr/copy/k = mean_{b,pos} ||z_t - z_{t+k}||^2 over channels / the target's batch spread,
    the same denominator as relerr/k, with z from the target encoder at the head's level."""
    cfg = full_cfg()
    model = model_of(cfg)
    batch = random_batch(cfg, batch=3)
    losses = compute_world_losses(model, batch, cfg["loss"], copy_baseline=True)
    frames, c = batch["frames"], model.context_frames
    for k in model.horizons:
        lv = model.head_level[k]
        with torch.no_grad():
            z0 = model.target_encoder(target_stack(frames, c, 0), upto=lv)[lv]
            zk = model.target_encoder(target_stack(frames, c, k), upto=lv)[lv]
        sq = (z0 - zk).pow(2).sum(1).mean()
        spread = zk.var(dim=0, unbiased=False).sum(0).mean()
        torch.testing.assert_close(losses[f"relerr/copy/k{k}"], sq / spread, rtol=1e-5, atol=0)


def test_copy_baseline_takes_no_gradient():
    cfg = full_cfg()
    model = model_of(cfg)
    losses = compute_world_losses(model, random_batch(cfg), cfg["loss"], copy_baseline=True)
    assert all(not losses[f"relerr/copy/k{k}"].requires_grad for k in model.horizons)


# ---- encoder gradient norms ----


def test_grad_norms_match_autograd_per_term_and_level():
    cfg = full_cfg(*TOP_DOWN)
    model = model_of(cfg)
    terms: dict[str, torch.Tensor] = {}
    compute_world_losses(model, random_batch(cfg), cfg["loss"], terms=terms)
    norms = train.encoder_grad_norms(model, terms)
    enc = model.context_encoder
    for name, t in terms.items():
        for lv, stage in zip(enc.names, enc.stages, strict=True):
            ps = list(stage.parameters())
            gs = torch.autograd.grad(t, ps, retain_graph=True, allow_unused=True)
            want = math.sqrt(sum(float(g.pow(2).sum()) for g in gs if g is not None))
            got = norms[f"gradnorm/enc/{name}/{lv}"]
            assert math.isclose(float(got), want, rel_tol=1e-5, abs_tol=1e-12), (name, lv)


def test_heads_pull_only_on_the_levels_below_what_they_read():
    """Near heads read L1: no gradient reaches the L2 and L3 stages. The k = 20 head and the goal
    head read L3, so they reach every stage."""
    cfg = full_cfg()
    model = model_of(cfg)
    terms: dict[str, torch.Tensor] = {}
    compute_world_losses(model, random_batch(cfg), cfg["loss"], terms=terms)
    n = {k: float(v) for k, v in train.encoder_grad_norms(model, terms).items()}
    for k in (1, 5):
        assert n[f"gradnorm/enc/k{k}/L1"] > 0
        assert n[f"gradnorm/enc/k{k}/L2"] == 0 and n[f"gradnorm/enc/k{k}/L3"] == 0
    for name in ("k20", "goal"):
        assert all(n[f"gradnorm/enc/{name}/{lv}"] > 0 for lv in ("L1", "L2", "L3")), name


# ---- training is unchanged ----


def _params_after(cfg: dict, tmp_path, name: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    batches = (random_batch(cfg, seed=i) for i in itertools.count())
    train.run(cfg, tmp_path / name, batches=batches, device=torch.device("cpu"))
    (ckpt,) = (tmp_path / name / "checkpoints").iterdir()
    return torch.load(ckpt, weights_only=False)["model"]


@pytest.mark.parametrize("extra", [[], TOP_DOWN], ids=["plain", "top_down"])
def test_diagnostics_on_or_off_train_bitwise_the_same(tmp_path, extra):
    """The M7 ablations reuse the 48k reference only while training is bitwise unchanged: params
    after a few steps are equal with both diagnostics every step and with both off."""
    base = ["train.steps=4", *extra]
    off = full_cfg(*base, "train.grad_norm_every=0", "train.copy_every=0")
    on = full_cfg(*base, "train.grad_norm_every=1", "train.copy_every=1")
    a, b = _params_after(off, tmp_path, "off"), _params_after(on, tmp_path, "on")
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def _rows(run_dir) -> list[dict]:
    return [json.loads(s) for s in (run_dir / "metrics.jsonl").read_text().splitlines()]


def test_diagnostics_are_logged_every_nth_step_and_nan_otherwise(tmp_path):
    cfg = full_cfg("train.steps=5", "train.grad_norm_every=2", "train.copy_every=3")
    torch.manual_seed(0)
    batches = (random_batch(cfg, seed=i) for i in itertools.count())
    train.run(cfg, tmp_path, batches=batches, device=torch.device("cpu"))
    rows = _rows(tmp_path)
    keys = train.diagnostic_keys(train.build_model(cfg), cfg)
    assert all(set(keys) <= set(r) for r in rows)  # the same columns every step
    for r in rows:
        done = r["step"] - 1  # steps completed before this one, as the stage schedule counts
        for key in keys:
            every = 3 if key.startswith("relerr/copy/") else 2
            assert math.isfinite(r[key]) == (done % every == 0), (r["step"], key)


def test_a_term_not_yet_in_the_loss_has_no_grad_norm(tmp_path):
    """Before its stage starts a head isn't trained, so it pulls on nothing: NaN, not a norm."""
    cfg = full_cfg(
        "train.steps=2",
        "train.stage_start={near: 0, far: 1, goal: 1, pixel: 1}",
        "train.grad_norm_every=1",
    )
    torch.manual_seed(0)
    batches = (random_batch(cfg, seed=i) for i in itertools.count())
    train.run(cfg, tmp_path, batches=batches, device=torch.device("cpu"))
    first, second = _rows(tmp_path)
    for name in ("k10", "k20", "goal", "pix"):
        assert math.isnan(first[f"gradnorm/enc/{name}/L1"]), name
        assert math.isfinite(second[f"gradnorm/enc/{name}/L1"]), name
    assert math.isfinite(first["gradnorm/enc/k1/L1"])


def test_total_grad_norm_is_the_encoder_gradient_the_optimizer_gets(tmp_path):
    cfg = full_cfg("train.steps=1", "train.grad_norm_every=1")
    model = model_of(cfg)
    batch = random_batch(cfg)
    ref = model_of(cfg)
    compute_world_losses(ref, batch, cfg["loss"])["loss/total"].backward()
    out = train.train_step(model, train.build_optimizer(model, cfg), batch, cfg, step=0)
    enc = ref.context_encoder
    for lv, stage in zip(enc.names, enc.stages, strict=True):
        want = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in stage.parameters()))
        assert math.isclose(float(out[f"gradnorm/enc/total/{lv}"]), want, rel_tol=1e-5), lv


# ---- config ----


@pytest.mark.parametrize("key", ["grad_norm_every", "copy_every"])
@pytest.mark.parametrize("value", ["-1", "1.5", "true", "null"])
def test_diagnostic_intervals_must_be_ints_ge_0(key, value):
    with pytest.raises(ValueError, match=f"train.{key}"):
        train.check_config(tiny_cfg(f"train.{key}={value}"))


def test_diagnostics_are_on_by_default():
    """So the M7 ablation runs, which inherit base.yaml, log them."""
    t = tiny_cfg()["train"]
    assert t["grad_norm_every"] > 0 and t["copy_every"] > 0
