"""Training step, loop, run directory and CLI (SPEC §5.4, §6)."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys

import pytest
import torch
from tiny import CONFIGS, NEAR, TINY, random_batch, random_frames, tiny_aux_cfg, tiny_cfg

from jepa_world import train
from jepa_world.data.synthetic import moving_disc_batch
from jepa_world.utils import paths
from jepa_world.utils.config import load_config


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
    expected = {f"{p}/k{k}" for p in ("loss", "relerr") for k in NEAR} | {
        "loss/var",
        "var/L1",
        "loss/total",
        "emb_std/L1",
    }
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


def test_episode_source_needs_its_split_built():
    # The episode loader itself is tested in test_train_episodes.py.
    with pytest.raises(FileNotFoundError, match=r"jepa_world\.data\.build"):
        train.check_config(tiny_cfg("data.source=episodes"))


def test_goal_and_pixel_stages_build_and_check():
    """M5 built the last two SPEC §5.5 stages."""
    cfg = tiny_aux_cfg()
    assert train.check_config(cfg) == NEAR  # aux stages own no horizon
    model = train.build_model(cfg)
    assert model.goal is not None and model.pixel is not None


def test_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="bogus"):
        train.check_config(tiny_cfg("train.stages=[near, bogus]"))


@pytest.mark.parametrize("stage", ["goal", "pixel"])
def test_aux_stage_needs_an_int_start(stage):
    with pytest.raises(ValueError, match=f"stage_start.{stage}"):
        train.check_config(tiny_aux_cfg(f"train.stage_start.{stage}=null"))


@pytest.mark.parametrize("stage", ["goal", "pixel"])
def test_aux_stage_that_would_never_train_is_refused(stage):
    with pytest.raises(ValueError, match="never train"):
        train.check_config(tiny_aux_cfg("train.steps=3", f"train.stage_start.{stage}=3"))


def test_active_aux_follows_stage_start():
    cfg = tiny_aux_cfg("train.stage_start={near: 0, goal: 2, pixel: 4}")
    assert [train.active_aux(cfg, s) for s in range(6)] == [
        [],
        [],
        ["goal"],
        ["goal"],
        ["goal", "pixel"],
        ["goal", "pixel"],
    ]
    assert train.active_aux(cfg, None) == ["goal", "pixel"]
    assert train.active_aux(tiny_cfg(), 100) == []


def test_batches_carry_ball_pos_only_with_a_goal_stage():
    c = 4
    cfg = tiny_aux_cfg()
    batch = next(train.make_batches(cfg, NEAR))
    assert set(batch) == {"frames", "ball_pos"}
    assert batch["frames"].shape == (2, c + 5, 3, 16, 16)
    assert batch["ball_pos"].shape == (2, c + 5, 2) and batch["ball_pos"].dtype == torch.float32
    frames_only = next(train.make_batches(tiny_aux_cfg("train.stages=[near, pixel]"), NEAR))
    assert isinstance(frames_only, torch.Tensor) and frames_only.shape == (2, c + 5, 3, 16, 16)


def test_batch_length_covers_the_goal_horizons():
    cfg = tiny_aux_cfg("model.goal_horizons=[1, 20]")
    assert train.batch_k_max(cfg, NEAR) == 20
    assert next(train.make_batches(cfg, NEAR))["frames"].shape[1] == 4 + 20
    assert train.batch_k_max(tiny_cfg(), NEAR) == 5


def test_train_step_takes_a_dict_or_a_frames_tensor():
    cfg = tiny_aux_cfg()
    model, opt = _built(cfg)
    losses = train.train_step(model, opt, random_batch(cfg), cfg)
    assert {"loss/goal", "loss/pix"} <= set(losses)
    assert all(not v.requires_grad for v in losses.values())
    cfg = tiny_cfg()
    model, opt = _built(cfg)
    assert "loss/total" in train.train_step(model, opt, random_frames(cfg), cfg)


def test_train_step_keeps_unstarted_aux_stages_out_of_the_total():
    cfg = tiny_aux_cfg("train.stage_start={near: 0, goal: 5, pixel: 5}")
    model, opt = _built(cfg)
    train.train_step(model, opt, random_batch(cfg), cfg, step=0)
    assert all(p.grad is None for p in model.goal.parameters())
    assert all(p.grad is None for p in model.pixel.parameters())
    train.train_step(model, opt, random_batch(cfg), cfg, step=5)
    assert all(p.grad is not None for p in model.goal.parameters())


def test_same_seed_gives_identical_losses_with_aux_stages(tmp_path):
    cfg = tiny_aux_cfg("train.steps=2")
    a = train.run(cfg, tmp_path / "a", device=torch.device("cpu"))
    b = train.run(cfg, tmp_path / "b", device=torch.device("cpu"))
    assert a == b and {"loss/goal", "loss/pix"} <= set(a)
    c = train.run(
        tiny_aux_cfg("train.steps=2", "seed=1"), tmp_path / "c", device=torch.device("cpu")
    )
    assert a != c


def test_run_logs_the_aux_terms_every_step(tmp_path):
    cfg = tiny_aux_cfg("train.steps=3")
    train.check_config(cfg)
    train.run(cfg, tmp_path, device=torch.device("cpu"))
    lines = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [1, 2, 3]
    for r in lines:
        assert {"loss/goal", "loss/pix", "loss/goal/k3", "goalerr/k3", "goalratio/k3"} <= set(r)
    ckpt = torch.load(tmp_path / "checkpoints" / "step_0000003.pt", weights_only=False)
    assert any(k.startswith("goal.") for k in ckpt["model"])
    assert any(k.startswith("pixel.") for k in ckpt["model"])


def test_pixel_term_does_not_dominate_the_logged_total():
    """SPEC §8 M5. At base.yaml's default (pix_share 0.10, decision 2026-11-19) the weighted pixel
    term is 0.10/1.10 of loss/total, under 10%, at every step of a short run."""
    cfg = tiny_aux_cfg("train.stages=[near, pixel]")
    assert cfg["loss"]["pix_share"] == 0.10  # the base default, not one picked for the test
    model, opt = _built(cfg)
    g = torch.Generator().manual_seed(0)
    for step in range(10):
        frames = moving_disc_batch(2, 9, 16, generator=g)
        losses = train.train_step(model, opt, frames, cfg, step=step)
        pix = losses["w_pix_eff"] * losses["loss/pix"]
        assert pix < 0.1 * losses["loss/total"], (step, pix.item(), losses["loss/total"].item())


def test_fixed_pixel_weight_term_is_bounded_by_w_pix():
    """pix_share null (e.g. the headline configs): pixel_l1 <= 1 bounds the term by w_pix
    outright, and at w_pix 0.05 it stays under 10% of loss/total. A sum reduction (x3 channels
    x every pixel) would break both."""
    cfg = tiny_aux_cfg("train.stages=[near, pixel]", "loss.pix_share=null")
    assert cfg["loss"]["w_pix"] == 0.05  # the base weight, not one picked for the test
    model, opt = _built(cfg)
    g = torch.Generator().manual_seed(0)
    for step in range(10):
        frames = moving_disc_batch(2, 9, 16, generator=g)
        losses = train.train_step(model, opt, frames, cfg, step=step)
        pix = cfg["loss"]["w_pix"] * losses["loss/pix"]
        assert losses["w_pix_eff"] == cfg["loss"]["w_pix"]
        assert pix <= cfg["loss"]["w_pix"]
        assert pix < 0.1 * losses["loss/total"], (step, pix.item(), losses["loss/total"].item())


def test_cli_rejects_bad_config_before_creating_a_run_dir():
    sets = [arg for s in TINY for arg in ("--set", s)]
    with pytest.raises(FileNotFoundError):  # episodes, but no split built in this artifact root
        train.main(["--config", str(CONFIGS / "base.yaml"), *sets, "--set", "data.source=episodes"])
    assert not (paths.artifacts_root() / "runs").exists()


def test_precision_other_than_fp32_is_refused():
    with pytest.raises(NotImplementedError, match="fp32"):
        train.check_config(tiny_cfg("train.precision=bf16"))


def test_config_without_a_near_horizon_is_refused():
    with pytest.raises(ValueError, match="near horizon"):
        train.check_config(tiny_cfg("model.horizons=[10, 20]"))


def test_model_builds_for_any_data_source():
    # Data-source checks belong to the run entry points, not to model construction: M2's
    # dev-set test must be able to build the model from an episodes config.
    model = train.build_model(tiny_cfg("data.source=episodes"))
    assert model.horizons == NEAR


def test_a_crash_still_writes_the_buffered_steps(tmp_path):
    cfg = tiny_cfg("train.steps=10", "train.log_every=50")
    frames = random_frames(cfg)

    def three_then_crash():
        for _ in range(3):
            yield frames
        raise RuntimeError("simulated failure at step 4")

    with pytest.raises(RuntimeError, match="step 4"):
        train.run(cfg, tmp_path, batches=three_then_crash(), device=torch.device("cpu"))
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(s)["step"] for s in lines] == [1, 2, 3]


MPS_STEP = """
import json, sys
sys.path.insert(0, "tests")
import jepa_world  # sets PYTORCH_ENABLE_MPS_FALLBACK before torch
import torch
from tiny import spec_cfg
from jepa_world import train
from jepa_world.bench import record_mps_fallbacks
from jepa_world.data.synthetic import moving_disc_batch

cfg = spec_cfg("train.batch_size=2")
with record_mps_fallbacks() as fallbacks:
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = train.build_model(cfg).to(dev)
    opt = train.build_optimizer(model, cfg)
    frames = moving_disc_batch(2, 9, 64, generator=torch.Generator().manual_seed(0)).to(dev)
    losses = train.train_step(model, opt, frames, cfg)
    torch.mps.synchronize()
finite = {k: bool(torch.isfinite(v).item()) for k, v in losses.items()}
print(json.dumps({"finite": finite, "fallbacks": fallbacks}))
"""


@pytest.mark.mps
def test_train_step_on_mps_at_spec_shapes_is_finite_with_no_cpu_fallback():
    # torch warns about each fallback op once per process, so an earlier MPS test in this
    # process could hide one. A fresh interpreter sees every op's first use.
    root = CONFIGS.parent
    r = subprocess.run(
        [sys.executable, "-c", MPS_STEP], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(result["finite"].values()), result
    assert {f"loss/k{k}" for k in NEAR} <= set(result["finite"])
    assert result["fallbacks"] == [], result["fallbacks"]


MPS_ALL_STAGES = """
import json, sys
sys.path.insert(0, "tests")
import jepa_world  # sets PYTORCH_ENABLE_MPS_FALLBACK before torch
import torch
from tiny import spec_cfg
from jepa_world import train
from jepa_world.bench import record_mps_fallbacks
from jepa_world.data.synthetic import moving_disc_batch

cfg = spec_cfg(
    "train.batch_size=2",
    "train.stages=[near, far, goal, pixel]",
    "train.stage_start={near: 0, far: 0, goal: 0, pixel: 0}",
)
with record_mps_fallbacks() as fallbacks:
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = train.build_model(cfg).to(dev)
    opt = train.build_optimizer(model, cfg)
    batch = moving_disc_batch(2, 24, 64, generator=torch.Generator().manual_seed(0), with_gt=True)
    batch = {k: v.to(dev) for k, v in batch.items()}
    losses = train.train_step(model, opt, batch, cfg)
    torch.mps.synchronize()
finite = {k: bool(torch.isfinite(v).item()) for k, v in losses.items()}
print(json.dumps({"finite": finite, "fallbacks": fallbacks}))
"""


@pytest.mark.mps
def test_all_four_stages_on_mps_at_spec_shapes_are_finite_with_no_cpu_fallback():
    root = CONFIGS.parent
    r = subprocess.run(
        [sys.executable, "-c", MPS_ALL_STAGES],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(result["finite"].values()), result
    assert {"loss/goal", "loss/pix", "loss/k20"} <= set(result["finite"])
    assert result["fallbacks"] == [], result["fallbacks"]


def test_pix_share_is_logged_every_step_and_hits_its_target(tmp_path):
    cfg = tiny_aux_cfg(
        "train.steps=4", "train.stage_start={near: 0, goal: 0, pixel: 2}", "loss.pix_share=0.10"
    )
    train.check_config(cfg)
    train.run(cfg, tmp_path, device=torch.device("cpu"))
    lines = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [1, 2, 3, 4]
    for r in lines:  # pixel joins after 2 completed steps: metrics steps 3 and 4
        if r["step"] <= 2:
            assert r["w_pix_eff"] == 0 and r["pix_share"] == 0
        else:
            assert r["w_pix_eff"] > 0 and r["pix_share"] == pytest.approx(0.10, rel=1e-4)


def test_check_config_refuses_a_bad_pix_share():
    with pytest.raises(ValueError, match="pix_share"):
        train.check_config(tiny_aux_cfg("loss.pix_share=2.0"))


def test_pix_share_is_a_no_op_without_a_pixel_stage():
    """base.yaml's share (0.10 since 2026-11-19) must not refuse near/far-only runs: with no pixel
    stage there is no decoder and no pixel term, exactly as w_pix is then unused."""
    cfg = tiny_cfg()  # near only, base.yaml's loss block
    assert cfg["loss"]["pix_share"] == 0.10
    train.check_config(cfg)
    model, opt = _built(cfg)
    losses = train.train_step(model, opt, random_frames(cfg), cfg, step=0)
    assert model.pixel is None
    assert not {"loss/pix", "w_pix_eff", "pix_share"} & set(losses)


def test_base_default_holds_the_pixel_term_at_ten_percent():
    """docs/decisions/2026-11-19-pixel-share-default.md. w_pix stays as the null fallback."""
    loss = load_config(CONFIGS / "base.yaml")["loss"]
    assert loss["pix_share"] == 0.10
    assert loss["w_pix"] == 0.05


@pytest.mark.parametrize("name", ["exp_headline", "exp_headline_48k"])
def test_configs_with_finished_runs_keep_the_fixed_pixel_weight(name):
    """Their runs trained at w_pix 0.05 (pixel share <= 3e-5); re-running them must reproduce
    that, whatever base.yaml defaults to."""
    loss = load_config(CONFIGS / f"{name}.yaml")["loss"]
    assert loss["pix_share"] is None
    assert loss["w_pix"] == 0.05


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.stem)
def test_every_committed_config_passes_check_config(path):
    """A base.yaml default must never make an existing experiment config unrunnable. Synthetic
    data, because the episode datasets aren't built in the test artifact root."""
    train.check_config(load_config(path, ["data.source=synthetic"]))


@pytest.mark.parametrize(
    ("name", "share"), [("m7_pixel_share01", 0.01), ("m7_pixel_share10", 0.10)]
)
def test_m7_pixel_share_configs(name, share):
    """M7's rescaled pixel ablation: the headline run (24k steps, all four stages, base starts)
    with the pixel term held at a share of the rest of the loss."""
    cfg = load_config(CONFIGS / f"{name}.yaml")
    headline = load_config(CONFIGS / "exp_headline.yaml")
    assert cfg["loss"]["pix_share"] == share
    assert cfg["train"]["steps"] == 24000
    assert cfg["train"]["stages"] == ["near", "far", "goal", "pixel"]
    rest = {k: v for k, v in cfg["loss"].items() if k != "pix_share"}
    assert rest == {k: v for k, v in headline["loss"].items() if k != "pix_share"}
    assert {k: v for k, v in cfg.items() if k != "loss"} == {
        k: v for k, v in headline.items() if k != "loss"
    }
