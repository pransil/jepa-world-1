"""experiments/m7_matched_windows.py: a run's goal head scored on one dataset's frames, with the
hidden windows flagged by another dataset's ground truth (M7 §3 and §5).

Tiny CPU models on tiny fake datasets, as in test_eval.py: the numbers are meaningless. What is
tested: on its own flags it is exactly eval's visibility and events; flags come only from the flags
set and frames only from the frames set; and it refuses two sets with different physics."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from fake_episodes import SHORT, SMALL_SPLITS, Recorder
from tiny import AUX, BASE, TINY

from jepa_world import baselines, train
from jepa_world import eval as eval_cli
from jepa_world.data import build
from jepa_world.data.dataset import load_split, valid_starts
from jepa_world.utils.config import load_config

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "m7_matched_windows.py"
_spec = importlib.util.spec_from_file_location("m7_matched_windows", _PATH)
mw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mw)

EPISODES = ["data.source=episodes", *SHORT, "sim.frame_size=16"]
EVAL_SETS = [*SMALL_SPLITS, "eval.device=cpu", "eval.batch_size=8", "eval.sim_params=[]"]
CPU = torch.device("cpu")


def run_cfg(*extra: str) -> dict:
    return load_config(BASE, [*TINY, *AUX, *EPISODES, *EVAL_SETS, *extra])


@pytest.fixture(scope="module")
def root(tmp_path_factory):
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


def make_run(cfg: dict, step: int = 3):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    opt = train.build_optimizer(model, cfg)
    run = train.start_run(cfg, "tiny-matched", CPU)
    train.save_checkpoint(run / "checkpoints" / f"step_{step:07d}.pt", model, opt, step, cfg)
    return run, model.eval()


def with_flags(es, seed: int = 0):
    """`es` with a random out_of_view mask (fake episodes have none), so hidden windows exist."""
    rng = np.random.default_rng(seed)
    return dataclasses.replace(es, out_of_view=rng.random(es.out_of_view.shape) < 0.3)


def eval_reference(cfg, model, es):
    """eval's own visibility and events for `es`, on eval's windows."""
    c = cfg["model"]["context_frames"]
    k_eval = max([*cfg["model"]["horizons"], *cfg["model"]["goal_horizons"]])
    ep, t = baselines.window_index(es.n, valid_starts(es.T, c, k_eval))
    out = eval_cli.model_outputs(model, es, ep, t, 8, CPU, features=False)
    fam = eval_cli.score_split(cfg, cfg, "test", es, ep, t, out)
    return fam["visibility"], fam["events"]


def test_on_its_own_flags_it_is_evals_visibility_and_events(artifacts):
    cfg = run_cfg()
    _, model = make_run(cfg)
    es = with_flags(load_split(cfg, "test"))
    vis, events = eval_reference(cfg, model, es)
    ours = mw.score_windows(model, cfg, es, es, "out_of_view", CPU, 8)
    ks = [str(k) for k in cfg["model"]["goal_horizons"]]
    assert sorted(ours["visibility"]) == sorted(ks) == sorted(ours["events"])
    for k in ks:
        assert vis[k]["out_of_view"]["has_hidden"]
        assert ours["visibility"][k] == vis[k]["out_of_view"]
        assert ours["events"][k] == events[k]["out_of_view"]


def test_flags_come_from_the_flags_set_and_frames_from_the_frames_set(artifacts):
    cfg = run_cfg()
    _, model = make_run(cfg)
    base = load_split(cfg, "test")
    flagged = with_flags(base)
    want = mw.score_windows(model, cfg, flagged, flagged, "out_of_view", CPU, 8)
    # The frames set carries no flags; the flags set has the flags and blank frames.
    blank = dataclasses.replace(flagged, frames=torch.zeros_like(flagged.frames))
    got = mw.score_windows(model, cfg, base, blank, "out_of_view", CPU, 8)
    assert got == want
    # And the frames set's frames are the ones scored: blank frames change the output.
    other = mw.score_windows(model, cfg, blank, flagged, "out_of_view", CPU, 8)
    assert other != want


def test_two_sets_must_hold_the_same_episodes(artifacts):
    cfg = run_cfg()
    es = load_split(cfg, "test")
    mw.check_same_episodes(es, es)
    pos = es.ball_pos.copy()
    pos[0, 0, 0] += 1e-9
    with pytest.raises(ValueError, match="ball_pos"):
        mw.check_same_episodes(es, dataclasses.replace(es, ball_pos=pos))
    vel = es.ball_vel.copy()
    vel[-1, -1, 1] = -vel[-1, -1, 1] - 1.0
    with pytest.raises(ValueError, match="ball_vel"):
        mw.check_same_episodes(es, dataclasses.replace(es, ball_vel=vel))
    seeds = es.ball_seed.copy()
    seeds[0] += 1
    with pytest.raises(ValueError, match="ball_seed"):
        mw.check_same_episodes(es, dataclasses.replace(es, ball_seed=seeds))
    _, model = make_run(cfg)
    with pytest.raises(ValueError, match="ball_pos"):
        mw.score_windows(
            model, cfg, es, dataclasses.replace(es, ball_pos=pos), "out_of_view", CPU, 8
        )


def test_only_a_known_flag(artifacts):
    cfg = run_cfg()
    _, model = make_run(cfg)
    es = load_split(cfg, "test")
    with pytest.raises(ValueError, match="flag"):
        mw.score_windows(model, cfg, es, es, "hidden", CPU, 8)


def test_score_run_writes_one_file_under_matched_with_its_metadata(artifacts):
    cfg = run_cfg()
    run, model = make_run(cfg, step=7)
    doc, path = mw.score_run(run, cfg, "out_of_view", device=CPU)
    assert path.parent == run / "matched" and path.suffix == ".json"
    assert not (run / "eval").exists()  # never next to eval's reports
    assert json.loads(path.read_text()) == json.loads(json.dumps(doc))
    meta = doc["meta"]
    assert meta["step"] == 7 and meta["split"] == "test" and meta["flag"] == "out_of_view"
    assert {"sha", "dirty"} <= set(meta["git"])
    es = load_split(cfg, "test")
    assert (
        doc["visibility"]
        == mw.score_windows(model, cfg, es, es, "out_of_view", CPU, 8)["visibility"]
    )
    with pytest.raises(Exception, match="trained on"):  # the run's own training split
        mw.score_run(run, cfg, "out_of_view", split="train", device=CPU)
