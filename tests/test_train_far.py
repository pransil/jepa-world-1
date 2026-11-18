"""Training with the far stage (M4) and the §5.5 stage schedule.

A separate file from test_train.py, so this lane and the M2 lane (which edits the episode-source
tests in test_train.py) don't contend for the same file.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
from fake_episodes import SMALL_SPLITS, Recorder, decode
from tiny import ALL, CONFIGS, FAR, NEAR, random_frames, tiny_cfg, tiny_far_cfg

from jepa_world import train
from jepa_world.data import build
from jepa_world.data.synthetic import moving_disc_batch


def _built(cfg):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    return model, train.build_optimizer(model, cfg)


def test_far_stage_builds_every_horizon():
    assert train.build_model(tiny_far_cfg()).horizons == ALL
    assert train.build_model(tiny_cfg()).horizons == NEAR


@pytest.mark.parametrize("step, expected", [(None, ALL), (0, NEAR), (2, NEAR), (3, ALL), (99, ALL)])
def test_active_horizons_follow_stage_start(step, expected):
    cfg = tiny_far_cfg("train.stage_start={near: 0, far: 3}")
    assert train.active_horizons(cfg, ALL, step) == expected


@pytest.mark.parametrize(
    "overrides, match",
    [
        (["train.stage_start={near: 0}"], "far"),  # a stage without a start step
        (["train.stage_start={near: 5, far: 10}", "train.steps=20"], "start at 0"),  # none at 0
        (["train.stage_start={near: 0, far: -1}"], "far"),  # negative
        (["train.stage_start={near: 0, far: true}"], "far"),  # a YAML bool is not a step
        (["train.stage_start={near: 0, far: 3}", "train.steps=3"], "never"),  # starts after the end
        # far owns no horizon here, so its 0 must not count as "something trains at step 0"
        (["train.stage_start={near: 1, far: 0}", "model.horizons=[1, 2]"], "start at 0"),
    ],
)
def test_bad_stage_start_is_refused(overrides, match):
    with pytest.raises(ValueError, match=match):
        train.check_config(tiny_far_cfg(*overrides))


def test_stage_start_counts_completed_steps(tmp_path):
    # far: 2 -> steps 1 and 2 (as numbered in metrics.jsonl) are near-only, far joins at step 3
    cfg = tiny_far_cfg("train.steps=4", "train.stage_start={near: 0, far: 2}")
    train.run(cfg, tmp_path, device=torch.device("cpu"))
    lines = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    for r in lines:
        near_only = sum(r[f"loss/k{k}"] for k in NEAR) + cfg["loss"]["w_var"] * r["var/L1"]
        if r["step"] <= 2:
            assert r["loss/total"] == pytest.approx(near_only, rel=1e-5), r["step"]
        else:
            assert r["loss/total"] != pytest.approx(near_only, rel=1e-3), r["step"]


def test_far_heads_do_not_move_before_their_stage_starts():
    cfg = tiny_far_cfg("train.stage_start={near: 0, far: 2}")
    model, opt = _built(cfg)
    far = {
        n: p.clone()
        for n, p in model.named_parameters()
        if n.startswith(("heads.10.", "heads.20."))
    }
    frames = random_frames(cfg)
    for step in (0, 1):
        train.train_step(model, opt, frames, cfg, step=step)
    params = dict(model.named_parameters())
    for n, p0 in far.items():
        assert torch.equal(params[n], p0), n  # no gradient, so no Adam step and no weight decay
    train.train_step(model, opt, frames, cfg, step=2)
    assert any(not torch.equal(params[n], p0) for n, p0 in far.items())


def test_training_lowers_near_and_far_losses_on_a_fixed_batch():
    cfg = tiny_far_cfg()
    model, opt = _built(cfg)
    frames = moving_disc_batch(4, 24, 16, generator=torch.Generator().manual_seed(0))
    near, far = [], []
    for _ in range(20):
        losses = train.train_step(model, opt, frames, cfg)
        near.append(sum(losses[f"loss/k{k}"] for k in NEAR).item())
        far.append(sum(losses[f"loss/k{k}"] for k in FAR).item())
    assert near[-1] < 0.75 * near[0], near
    assert far[-1] < far[0] - 1.0, far  # NLL can be negative, so compare by difference


def test_run_with_far_stage_logs_every_term_every_step(tmp_path):
    cfg = tiny_far_cfg("train.steps=4", "train.stage_start={near: 0, far: 2}")
    train.run(cfg, tmp_path, device=torch.device("cpu"))
    lines = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [1, 2, 3, 4]
    expected = (
        {f"loss/k{k}" for k in ALL}
        | {f"relerr/k{k}" for k in ALL}
        | {f"sqerr/k{k}" for k in FAR}
        | {f"logvar/k{k}" for k in FAR}
        | {"loss/var", "var/L1", "var/L2", "var/L3", "loss/total", "step"}
        | {"emb_std/L1", "emb_std/L2", "emb_std/L3"}
    )
    for r in lines:
        assert set(r) == expected  # far terms are logged before their stage starts, too


def test_same_seed_gives_identical_losses_with_far_stage(tmp_path):
    cfg = tiny_far_cfg("train.steps=3")
    a = train.run(cfg, tmp_path / "a", device=torch.device("cpu"))
    b = train.run(cfg, tmp_path / "b", device=torch.device("cpu"))
    assert a == b


MPS_FAR_STEP = """
import json, sys
sys.path.insert(0, "tests")
import jepa_world  # sets PYTORCH_ENABLE_MPS_FALLBACK before torch
import torch
from tiny import spec_cfg
from jepa_world import train
from jepa_world.bench import record_mps_fallbacks
from jepa_world.data.synthetic import moving_disc_batch

cfg = spec_cfg("train.batch_size=2", "train.stages=[near, far]")
with record_mps_fallbacks() as fallbacks:
    torch.manual_seed(0)
    dev = torch.device("mps")
    model = train.build_model(cfg).to(dev)
    opt = train.build_optimizer(model, cfg)
    frames = moving_disc_batch(2, 24, 64, generator=torch.Generator().manual_seed(0)).to(dev)
    losses = train.train_step(model, opt, frames, cfg)  # step=None: every stage active
    torch.mps.synchronize()
finite = {k: bool(torch.isfinite(v).item()) for k, v in losses.items()}
print(json.dumps({"finite": finite, "fallbacks": fallbacks}))
"""


@pytest.mark.mps
def test_far_train_step_on_mps_at_spec_shapes_is_finite_with_no_cpu_fallback():
    # Near AND far heads at SPEC shapes, in a fresh interpreter: torch warns about each fallback
    # op once per process, so an earlier MPS test in this process could hide one.
    r = subprocess.run(
        [sys.executable, "-c", MPS_FAR_STEP],
        cwd=CONFIGS.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert all(result["finite"].values()), result
    assert {f"loss/k{k}" for k in ALL} <= set(result["finite"])
    assert result["fallbacks"] == [], result["fallbacks"]


# ---------------------------------------------------------------- far stage on M2's episode loader


def test_far_stage_reads_24_frame_windows_from_the_episode_loader(tmp_path):
    # k_max = 20, so each window is C + 20 = 24 consecutive frames, ending inside the episode.
    cfg = tiny_far_cfg("data.source=episodes", "data.split=dev", *SMALL_SPLITS, "train.steps=2")
    build.build_split(cfg, "dev", simulate=Recorder())
    horizons = train.check_config(cfg)
    assert horizons == ALL
    batch = next(train.make_batches(cfg, horizons))
    c = cfg["model"]["context_frames"]
    assert batch.shape == (2, c + 20, 3, 16, 16)
    idx, _, _ = decode(batch)
    assert (idx[:, 0] >= 0).all() and (idx[:, -1] <= cfg["sim"]["episode_frames"] - 1).all()
    assert (idx[:, 1:] - idx[:, :-1] == 1).all()  # consecutive, in order
    final = train.run(cfg, tmp_path, device=torch.device("cpu"))
    assert {f"loss/k{k}" for k in ALL} <= set(final)
