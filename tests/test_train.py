"""Training step, loop, run directory and CLI (SPEC §5.4, §6)."""

from __future__ import annotations

import itertools
import json

import pytest
import torch
from tiny import CONFIGS, NEAR, TINY, random_frames, spec_cfg, tiny_cfg

from jepa_world import train
from jepa_world.data.synthetic import moving_disc_batch
from jepa_world.utils import paths


def _built(cfg):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    return model, train.build_optimizer(model, cfg)


def test_optimizer_holds_no_target_parameter():
    model, opt = _built(tiny_cfg())
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    assert not in_opt & {id(p) for p in model.target_encoder.parameters()}
    assert in_opt == {id(p) for p in model.trainable_parameters()}


def test_optimizer_hyperparameters_come_from_config():
    _, opt = _built(tiny_cfg("train.lr=3e-4", "train.weight_decay=0.05"))
    assert opt.param_groups[0]["lr"] == 3e-4
    assert opt.param_groups[0]["weight_decay"] == 0.05


def test_train_step_applies_ema_after_the_optimizer_step():
    cfg = tiny_cfg("model.ema_momentum=0.9")
    model, opt = _built(cfg)
    t_before = [p.clone() for p in model.target_encoder.parameters()]
    c_before = [p.clone() for p in model.context_encoder.parameters()]
    train.train_step(model, opt, random_frames(cfg), cfg)
    c_after = list(model.context_encoder.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(c_after, c_before, strict=True))
    for pt, t0, c in zip(model.target_encoder.parameters(), t_before, c_after, strict=True):
        torch.testing.assert_close(pt, 0.9 * t0 + 0.1 * c, rtol=1e-6, atol=1e-7)


def test_train_step_returns_detached_terms():
    cfg = tiny_cfg()
    model, opt = _built(cfg)
    losses = train.train_step(model, opt, random_frames(cfg), cfg)
    assert {f"loss/k{k}" for k in NEAR} <= set(losses)
    assert all(not v.requires_grad for v in losses.values())


def test_training_lowers_the_latent_loss_on_a_fixed_batch():
    # Fast sibling of SPEC §8 M3 "loss decreases on the development set" (that one needs M2).
    cfg = tiny_cfg()
    model, opt = _built(cfg)
    frames = moving_disc_batch(4, 9, 16, generator=torch.Generator().manual_seed(0))
    latent = []
    for _ in range(30):
        losses = train.train_step(model, opt, frames, cfg)
        latent.append(sum(losses[f"loss/k{k}"] for k in NEAR).item())
    assert latent[-1] < 0.75 * latent[0], latent


def test_same_seed_gives_identical_losses(tmp_path):
    cfg = tiny_cfg("train.steps=3")
    a = train.run(cfg, tmp_path / "a", device=torch.device("cpu"))
    b = train.run(cfg, tmp_path / "b", device=torch.device("cpu"))
    assert a == b
    c = train.run(tiny_cfg("train.steps=3", "seed=1"), tmp_path / "c", device=torch.device("cpu"))
    assert a != c


def test_run_logs_every_term_every_step_and_checkpoints(tmp_path):
    cfg = tiny_cfg("train.steps=5", "train.log_every=2", "train.ckpt_every=4")
    train.run(cfg, tmp_path, device=torch.device("cpu"))
    lines = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [1, 2, 3, 4, 5]  # 5 is flushed even though 5 % 2 != 0
    expected = {f"loss/k{k}" for k in NEAR} | {"loss/var", "loss/total", "emb_std/L1"}
    for r in lines:
        assert set(r) == expected | {"step"}
        assert all(isinstance(r[key], float) for key in expected)
    ckpts = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert ckpts == ["step_0000004.pt", "step_0000005.pt"]
    ckpt = torch.load(tmp_path / "checkpoints" / "step_0000005.pt", weights_only=False)
    assert set(ckpt) == {"model", "optimizer", "step", "config"}
    assert ckpt["step"] == 5 and ckpt["config"] == cfg
    assert any(k.startswith("target_encoder.") for k in ckpt["model"])
    assert (tmp_path / "tensorboard").is_dir()


def test_run_accepts_an_explicit_batch_iterator(tmp_path):
    cfg = tiny_cfg("train.steps=2")
    frames = random_frames(cfg)
    final = train.run(cfg, tmp_path, batches=itertools.repeat(frames), device=torch.device("cpu"))
    assert set(final) >= {"loss/total"}


def test_cli_creates_a_complete_run_dir():
    sets = [arg for s in TINY for arg in ("--set", s)]
    rc = train.main(["--config", str(CONFIGS / "m3_near_synthetic.yaml"), *sets, "--name", "cli"])
    assert rc == 0
    (run,) = (paths.artifacts_root() / "runs").iterdir()
    assert "cli" in run.name
    for name in ("config.yaml", "meta.json", "metrics.jsonl"):
        assert (run / name).is_file(), name
    assert len((run / "metrics.jsonl").read_text().splitlines()) == 3
    meta = json.loads((run / "meta.json").read_text())
    assert meta["device"] == "cpu" and "libraries" in meta
    assert list((run / "checkpoints").iterdir())


def test_episode_source_says_it_needs_m2():
    with pytest.raises(NotImplementedError, match="M2"):
        train.check_config(tiny_cfg("data.source=episodes"))


@pytest.mark.parametrize("stage, milestone", [("far", "M4"), ("goal", "M5"), ("pixel", "M5")])
def test_later_stages_say_which_milestone_builds_them(stage, milestone):
    with pytest.raises(NotImplementedError, match=milestone):
        train.check_config(tiny_cfg(f"train.stages=[near, {stage}]"))


def test_cli_rejects_bad_config_before_creating_a_run_dir():
    sets = [arg for s in TINY for arg in ("--set", s)]
    with pytest.raises(NotImplementedError):
        train.main(["--config", str(CONFIGS / "base.yaml"), *sets, "--set", "data.source=episodes"])
    assert not (paths.artifacts_root() / "runs").exists()


@pytest.mark.mps
def test_train_step_on_mps_at_spec_shapes_is_finite_with_no_cpu_fallback():
    from jepa_world.bench import record_mps_fallbacks

    cfg = spec_cfg("train.batch_size=2")
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = train.build_model(cfg).to(dev)
    opt = train.build_optimizer(model, cfg)
    frames = moving_disc_batch(2, 9, 64, generator=torch.Generator().manual_seed(0)).to(dev)
    with record_mps_fallbacks() as fallbacks:
        losses = train.train_step(model, opt, frames, cfg)
        torch.mps.synchronize()
    assert all(torch.isfinite(v).item() for v in losses.values())
    assert fallbacks == [], fallbacks
