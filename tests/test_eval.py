"""The model eval command, `python -m jepa_world.eval --run` (SPEC §7.2, §7.3, §8 M6).

Tiny CPU models on tiny fake datasets: the numbers are meaningless, the wiring and the guarantees
are what is tested (same windows as the baselines, frames-only input, no leakage across configs,
test as the headline and dev kept separate)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
from fake_episodes import SHORT, SMALL_SPLITS, Recorder
from tiny import AUX, BASE, TINY

from jepa_world import eval as eval_cli
from jepa_world import train
from jepa_world.data import build
from jepa_world.data.dataset import load_split, valid_starts
from jepa_world.data.splits import LeakageError
from jepa_world.utils import paths
from jepa_world.utils.config import load_config

EPISODES = ["data.source=episodes", *SHORT, "sim.frame_size=16"]  # T=30; 16 px for TINY's L1
GOAL_KS = [1, 3, 5]  # AUX's model.goal_horizons
EVAL_SETS = [
    *SMALL_SPLITS,
    "eval.device=cpu",
    "eval.batch_size=8",
    "eval.sim_params=[]",
    "eval.probes.train_windows=16",
]


def run_cfg(*extra: str) -> dict:
    return load_config(BASE, [*TINY, *AUX, *EPISODES, *extra])


def eval_cfg(*extra: str) -> dict:
    return load_config(BASE, [*EVAL_SETS, *extra])


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """One artifact root with dev, train and test built (fake episodes), shared by this module."""
    root = tmp_path_factory.mktemp("artifacts")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JEPA_ARTIFACTS", str(root))
        cfg = run_cfg()
        for split in ("dev", "train", "test"):
            build.build_split(cfg, split, simulate=Recorder())
    return root


@pytest.fixture
def artifacts(root, monkeypatch):
    monkeypatch.setenv("JEPA_ARTIFACTS", str(root))
    return root


def make_run(cfg: dict, steps=(3,), name="tiny-eval"):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    opt = train.build_optimizer(model, cfg)
    run = train.start_run(cfg, name, torch.device("cpu"))
    for s in steps:
        train.save_checkpoint(run / "checkpoints" / f"step_{s:07d}.pt", model, opt, s, cfg)
    return run, model


# --- pure pieces ------------------------------------------------------------------------------


def test_goal_positions_are_ball_pos_at_t_plus_the_mean_displacement():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(2, 12, 2))
    ep, t = np.array([0, 1, 1]), np.array([2, 2, 4])
    mean = rng.normal(size=(3, 2, 2))
    logvar = rng.normal(size=(3, 2, 2))
    g = eval_cli.goal_positions(pos, ep, t, mean, logvar, [1, 5])
    for j, k in enumerate([1, 5]):
        np.testing.assert_array_equal(g["persistence"][k], pos[ep, t])
        np.testing.assert_array_equal(g["truth"][k], pos[ep, t + k])
        np.testing.assert_allclose(g["pred"][k], pos[ep, t] + mean[:, j])
        np.testing.assert_allclose(g["var"][k], np.exp(logvar[:, j]))
    with pytest.raises(ValueError, match="horizons"):
        eval_cli.goal_positions(pos, ep, t, mean, logvar, [1, 3, 5])


def test_eval_split_check_refuses_leakage_across_configs():
    cfg = run_cfg()  # trained on train
    eval_cli.check_eval_split(cfg, eval_cfg(), "test")
    eval_cli.check_eval_split(cfg, eval_cfg(), "dev")
    with pytest.raises(LeakageError, match="trained on"):
        eval_cli.check_eval_split(cfg, eval_cfg(), "train")
    with pytest.raises(LeakageError, match="trained on"):
        eval_cli.check_eval_split(run_cfg("data.split=dev"), eval_cfg(), "dev")
    # An eval config whose test terrains are the run's training terrains.
    overlap = "data.splits.test={terrain_seeds: [1000, 2], ball_seeds: [1000, 2]}"
    with pytest.raises(LeakageError, match="terrain"):
        eval_cli.check_eval_split(cfg, eval_cfg(overlap), "test")
    # Disjoint, but not the test set the run's config defines: not comparable.
    other = "data.splits.test={terrain_seeds: [200000, 2], ball_seeds: [1000, 2]}"
    with pytest.raises(ValueError, match="differ"):
        eval_cli.check_eval_split(cfg, eval_cfg(other), "test")


# --- loading a run ------------------------------------------------------------------------------


def test_load_run_takes_the_latest_checkpoint_strictly(artifacts):
    cfg = run_cfg()
    run, model = make_run(cfg, steps=(2, 10))
    loaded = eval_cli.load_run(run, device=torch.device("cpu"))
    assert loaded.step == 10 and loaded.checkpoint.name == "step_0000010.pt"
    for (n, a), (_, b) in zip(
        model.state_dict().items(), loaded.model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), n
    assert not loaded.model.training
    bad = torch.load(run / "checkpoints" / "step_0000010.pt", weights_only=True)
    bad["model"]["stray.weight"] = torch.zeros(1)
    torch.save(bad, run / "checkpoints" / "step_0000011.pt")
    with pytest.raises(RuntimeError, match="stray"):
        eval_cli.load_run(run, device=torch.device("cpu"))
    empty = paths.new_run_dir("no-ckpt")
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        eval_cli.load_run(empty, device=torch.device("cpu"))


def test_eval_outputs_ignore_frames_after_t(artifacts):
    cfg = run_cfg()
    _, model = make_run(cfg)
    model.eval()
    es = load_split(cfg, "test")
    t0 = 5
    ep, t = np.arange(es.n), np.full(es.n, t0)
    a = eval_cli.model_outputs(model, es, ep, t, batch_size=3, device=torch.device("cpu"))
    frames = es.frames.clone()
    frames[:, t0 + 1 :] = torch.randint(0, 256, frames[:, t0 + 1 :].shape, dtype=torch.uint8)
    b = eval_cli.model_outputs(
        model, dataclasses.replace(es, frames=frames), ep, t, batch_size=3,
        device=torch.device("cpu"),
    )  # fmt: skip
    assert a["goal_mean"].shape == (es.n, len(GOAL_KS), 2)
    np.testing.assert_array_equal(a["goal_mean"], b["goal_mean"])
    np.testing.assert_array_equal(a["goal_logvar"], b["goal_logvar"])
    assert set(a["features"]) == {"L1", "L2", "L3"}
    for lv in a["features"]:
        assert a["features"][lv].shape == (es.n, cfg["model"]["levels"][lv]["channels"])
        np.testing.assert_array_equal(a["features"][lv], b["features"][lv])


# --- the command, end to end ----------------------------------------------------------------------


def test_eval_run_reports_test_as_headline_and_dev_separately(artifacts, monkeypatch):
    cfg = run_cfg()
    run, _ = make_run(cfg)
    locks = []
    real_lock = eval_cli.gpu_lock

    def spy_lock(label, *, timeout=None, enabled=True):
        locks.append(enabled)
        return real_lock(label, timeout=timeout, enabled=False)

    monkeypatch.setattr(eval_cli, "gpu_lock", spy_lock)
    args = ["--run", str(run), "--split", "test", "--split", "dev"]
    for s in EVAL_SETS:
        args += ["--set", s]
    assert eval_cli.main(args) == 0
    assert locks == [False]  # CPU: no GPU lock
    (path,) = (run / "eval").glob("*.json")
    out = json.loads(path.read_text())
    assert out["meta"]["step"] == 3 and out["meta"]["checkpoint"].endswith("step_0000003.pt")
    assert out["report"]["headline"]["split"] == "test"
    assert set(out["report"]["development"]) == {"dev"}
    es = load_split(cfg, "test")
    starts = valid_starts(es.T, 4, 20)  # max of model.horizons and goal_horizons
    for metrics in (out["report"]["headline"]["metrics"], out["report"]["development"]["dev"]):
        assert set(metrics) == {"goal_head", "visibility", "events", "probes", "baselines"}
        goal, base = metrics["goal_head"], metrics["baselines"]
        assert goal["horizons"] == GOAL_KS and base["horizons"] == GOAL_KS
        assert goal["windows"] == base["windows"]
        assert base["starts"] == [int(starts[0]), int(starts[-1])]
        assert set(goal["calibration"]["1"]) == {"0.5", "0.9"}
        assert goal["ratio_to_baseline"]["persistence"] == goal["ratio_to_persistence"]
        assert set(goal["ratio_to_baseline"]) == set(base["error_m"])
        assert set(metrics["visibility"]) == {"1", "3", "5"}
        assert set(metrics["visibility"]["1"]) == {"out_of_view", "occluded", "hidden"}
        assert set(metrics["events"]["5"]) == {"out_of_view", "occluded"}
        probes = metrics["probes"]
        assert probes["train_split"] == "train" and probes["n_train"] == 16
        assert set(probes["by_level"]) == {"L1", "L2", "L3", "pooled"}
        assert set(probes["by_level"]["L3"]) == {"pos", "friction", "restitution", "radius"}


def test_eval_run_without_a_goal_head_reports_probes_and_baselines(artifacts):
    cfg = load_config(BASE, [*TINY, *EPISODES])  # near only: a pre-M5 run
    run, _ = make_run(cfg, name="near-only")
    report, path = eval_cli.evaluate_run(run, eval_cfg(), ["test"])
    metrics = report["report"]["headline"]["metrics"]
    assert set(metrics) == {"probes", "baselines"}
    assert "goal" in report["meta"]["note"]
    assert path.parent == run / "eval"


def test_eval_refuses_a_run_trained_on_synthetic_frames(artifacts):
    run, _ = make_run(run_cfg("data.source=synthetic"), name="synthetic")
    with pytest.raises(ValueError, match="episodes"):
        eval_cli.evaluate_run(run, eval_cfg(), ["test"])


def test_baselines_mode_takes_one_split():
    with pytest.raises(SystemExit):
        eval_cli.main(["--baselines", "--config", str(BASE), "--split", "test", "--split", "dev"])
