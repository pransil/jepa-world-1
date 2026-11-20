"""data.curriculum: train on another world (sim overrides) for the first N steps, then on the run's
own world (jepa_world/data/DESIGN.md, "Curriculum"). The H experiment's configs are checked here
too: configs/world_gentle.yaml and configs/exp_gentle_curriculum.yaml."""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json

import pytest
import torch
from fake_episodes import SMALL_SPLITS, Recorder, decode, fake_episode
from tiny import BASE, CONFIGS, NEAR, TINY, tiny_cfg

from jepa_world import train
from jepa_world.data import build
from jepa_world.data.dataset import batches as episode_batches
from jepa_world.data.dataset import load_split
from jepa_world.data.splits import LeakageError
from jepa_world.utils.config import load_config

GENTLE = "{terrain: {ledges: [0, 0]}, ball: {drop: [0.0, 0.0]}}"
MARK = 99  # ball code (blue channel) of every curriculum-world frame; real ball seeds here are 0, 1


def curriculum(steps, sim: str = GENTLE) -> str:
    return f"data.curriculum={{steps: {steps}, sim: {sim}}}"


def episodes_cfg(*overrides: str) -> dict:
    return tiny_cfg("data.source=episodes", "data.split=dev", *SMALL_SPLITS, *overrides)


def marked_episode(sim_cfg, terrain_seed, ball_seed):
    ep = fake_episode(sim_cfg, terrain_seed, ball_seed)
    frames = ep.frames.copy()
    frames[..., 2] = MARK
    return dataclasses.replace(ep, frames=frames)


def build_both(cfg: dict) -> None:
    """The run's world (plain fake episodes) and the curriculum's world (marked ones)."""
    build.build_split(cfg, "dev", simulate=Recorder())
    build.build_split(train.curriculum_cfg(cfg), "dev", simulate=Recorder(marked_episode))


def ball_codes(batch) -> set[int]:
    return set(decode(batch)[2].ravel().tolist())


# ---- no curriculum: the loader is exactly what it was ----


def test_no_curriculum_by_default_and_batches_are_the_plain_loader():
    cfg = episodes_cfg()
    assert cfg["data"]["curriculum"] is None and train.curriculum_cfg(cfg) is None
    build.build_split(cfg, "dev", simulate=Recorder())
    C, B = cfg["model"]["context_frames"], cfg["train"]["batch_size"]
    plain = episode_batches(load_split(cfg, "dev"), B, C, max(NEAR), cfg["seed"])
    got = itertools.islice(train.make_batches(cfg, NEAR), 6)
    for got_b, want in zip(got, itertools.islice(plain, 6), strict=True):
        assert torch.equal(got_b, want)


def _trained_params(cfg: dict, run_dir, batches=None) -> dict[str, torch.Tensor]:
    train.run(cfg, run_dir, batches=batches, device=torch.device("cpu"))
    (ckpt,) = (run_dir / "checkpoints").iterdir()
    return torch.load(ckpt, weights_only=False)["model"]


def test_no_curriculum_trains_bitwise_as_before_whether_the_key_is_null_or_absent(tmp_path):
    """The 48k reference is reused as the M7 baseline only while training at main is bitwise
    what it was. With data.curriculum null, or absent (a run config.yaml from before the key
    existed), the parameters after a few steps equal those from feeding the plain M2 loader, which
    is what make_batches returned before the curriculum existed."""
    null = episodes_cfg("train.steps=4")
    absent = episodes_cfg("train.steps=4")
    del absent["data"]["curriculum"]
    build.build_split(null, "dev", simulate=Recorder())
    C, B = null["model"]["context_frames"], null["train"]["batch_size"]
    plain = episode_batches(load_split(null, "dev"), B, C, max(NEAR), null["seed"])
    want = _trained_params(null, tmp_path / "plain", batches=plain)
    for name, cfg in (("null", null), ("absent", absent)):
        got = _trained_params(cfg, tmp_path / name)
        assert got.keys() == want.keys()
        for k in want:
            assert torch.equal(got[k], want[k]), (name, k)


# ---- the switch ----


def test_curriculum_world_feeds_the_first_steps_then_the_run_world_from_its_first_batch():
    cfg = episodes_cfg("train.steps=10", curriculum(3))
    build_both(cfg)
    got = list(itertools.islice(train.make_batches(cfg, NEAR), 7))
    assert [ball_codes(b) == {MARK} for b in got] == [True] * 3 + [False] * 4
    assert all(MARK not in ball_codes(b) for b in got[3:])
    # After the switch the run's world starts at its own first batch, as a run without a
    # curriculum would: the curriculum doesn't consume the run world's sampler.
    plain = train.make_batches(episodes_cfg("train.steps=10"), NEAR)
    for a, b in zip(got[3:], itertools.islice(plain, 4), strict=True):
        assert torch.equal(a, b)


def test_curriculum_cfg_merges_the_overrides_onto_sim_and_nothing_else():
    cfg = episodes_cfg(curriculum(2))
    phase = train.curriculum_cfg(cfg)
    assert phase["sim"]["terrain"]["ledges"] == [0, 0] and phase["sim"]["ball"]["drop"] == [0, 0]
    assert phase["data"]["curriculum"] is None
    want = {k: v for k, v in cfg.items() if k not in ("sim", "data")}
    assert {k: v for k, v in phase.items() if k not in ("sim", "data")} == want
    assert {**phase["data"], "curriculum": cfg["data"]["curriculum"]} == cfg["data"]
    # every sim key the overrides don't name is the run's
    for section in ("camera", "contact"):
        assert phase["sim"][section] == cfg["sim"][section]
    assert phase["sim"]["ball"]["radius"] == cfg["sim"]["ball"]["radius"]
    assert cfg["sim"]["terrain"]["ledges"] != [0, 0]  # the run's own config is untouched


def test_train_runs_through_the_switch(tmp_path):
    cfg = episodes_cfg("train.steps=3", curriculum(2))
    build_both(cfg)
    train.check_config(cfg)
    final = train.run(cfg, tmp_path, device=torch.device("cpu"))
    assert torch.isfinite(torch.tensor(final["loss/total"]))
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(s)["step"] for s in lines] == [1, 2, 3]


def test_cli_loads_both_worlds_before_taking_the_gpu_lock(monkeypatch):
    cfg_args = ["data.source=episodes", "data.split=dev", *SMALL_SPLITS, "train.steps=3"]
    build_both(episodes_cfg(curriculum(2)))
    events = []
    real_load = train.load_split
    monkeypatch.setattr(train, "load_split", lambda *a: events.append("load") or real_load(*a))

    @contextlib.contextmanager
    def fake_lock(label, **kw):
        events.append("lock")
        yield

    monkeypatch.setattr(train, "gpu_lock", fake_lock)
    sets = [arg for s in [*TINY, *cfg_args, curriculum(2)] for arg in ("--set", s)]
    assert train.main(["--config", str(BASE), *sets]) == 0
    assert events == ["load", "load", "lock"]


# ---- refusals, all before any run dir ----


@pytest.mark.parametrize("steps", [0, -1, 3, 5, 2.5, "true", '"2"'])
def test_curriculum_steps_must_leave_steps_in_the_run_world(steps):
    cfg = episodes_cfg("train.steps=3", curriculum(steps))
    with pytest.raises(ValueError, match=r"data\.curriculum\.steps"):
        train.check_curriculum(cfg)


@pytest.mark.parametrize(
    ("sim", "named"),
    [
        ("{terrain: {ledge: [0, 0]}}", "ledge"),  # a typo
        ("{wind: 1.0}", "wind"),  # not a sim key
        ("{terrain: 1}", "terrain"),  # a section replaced by a scalar
        ("{}", "sim"),  # nothing to change
        ("[1, 2]", "sim"),
        ("{terrain: {relief: 1.0}}", "relief"),  # a range given as a number
        ("{terrain: {ledges: [0, 0, 0]}}", "ledges"),  # a range of the wrong length
        ("{terrain: {max_slope: true}}", "max_slope"),  # a bool for a number
        ("{camera_mode: 3}", "camera_mode"),  # a number for a name
        # the batch contract and the generator's identity are the run's: a curriculum world
        # with other frames would fail only inside run(), after the run dir and the GPU lock
        ("{frame_size: 32}", "frame_size"),
        ("{episode_frames: 20}", "episode_frames"),
        ("{fps: 30}", "fps"),
        ("{generator_version: 2}", "generator_version"),
    ],
)
def test_curriculum_sim_may_only_override_existing_sim_keys(sim, named):
    cfg = episodes_cfg(curriculum(2, sim))
    with pytest.raises((KeyError, ValueError), match=named):
        train.check_curriculum(cfg)


@pytest.mark.parametrize(
    "value",
    [
        "{steps: 2}",
        f"{{sim: {GENTLE}}}",
        f"{{steps: 2, sim: {GENTLE}, split: train}}",
        "12000",
    ],
)
def test_curriculum_needs_exactly_steps_and_sim(value):
    cfg = episodes_cfg(f"data.curriculum={value}")
    with pytest.raises(ValueError, match=r"data\.curriculum"):
        train.check_curriculum(cfg)


def test_curriculum_needs_the_episode_source():
    cfg = tiny_cfg(curriculum(2))  # data.source: synthetic
    with pytest.raises(ValueError, match="episodes"):
        train.check_config(cfg)


def test_unbuilt_curriculum_world_fails_check_config_with_the_build_command():
    cfg = episodes_cfg(curriculum(2))
    build.build_split(cfg, "dev", simulate=Recorder())  # only the run's own world
    with pytest.raises(
        FileNotFoundError, match=r"(?s)data\.curriculum.*jepa_world\.data\.build"
    ) as e:
        train.check_config(cfg)
    # it names this curriculum's overrides, not some other experiment's files
    assert "ledges" in str(e.value) and "world_gentle" not in str(e.value)


def test_curriculum_never_reads_test():
    cfg = episodes_cfg("data.split=test", curriculum(2))
    with pytest.raises(LeakageError, match="test"):
        train.check_config(cfg)


# ---- the H experiment's configs ----


def test_gentle_world_keeps_the_ball_camera_and_walls_and_cannot_slip():
    """configs/world_gentle.yaml: no ledges, the ball starts on the surface, a lower slope cap.
    A solid sphere rolls without slipping while tan(slope) <= 3.5 mu, so the cap stays below
    3.5 x the smallest friction. Everything else is the full world's."""
    full = load_config(BASE)["sim"]
    gentle = load_config(CONFIGS / "world_gentle.yaml")["sim"]
    tg, tf = gentle["terrain"], full["terrain"]
    assert tg["ledges"] == [0, 0] and gentle["ball"]["drop"] == [0.0, 0.0]
    assert tg["max_slope"] < 3.5 * gentle["ball"]["friction"][0]
    assert tg["max_slope"] < tf["max_slope"] and tg["relief"][1] < tf["relief"][1]
    assert {k: v for k, v in gentle["ball"].items() if k != "drop"} == {
        k: v for k, v in full["ball"].items() if k != "drop"
    }
    for key in ("camera", "camera_mode", "contact", "fps", "episode_frames", "frame_size"):
        assert gentle[key] == full[key], key
    for key in ("length", "columns", "wall_width", "wall_height", "base"):
        assert tg[key] == tf[key], key


def test_gentle_curriculum_trains_on_the_gentle_world_then_the_m7_reference_world():
    """configs/exp_gentle_curriculum.yaml is the 48k reference plus data.curriculum, and its
    curriculum world is configs/world_gentle.yaml's, dataset for dataset."""
    exp = load_config(CONFIGS / "exp_gentle_curriculum.yaml")
    ref = load_config(CONFIGS / "exp_headline_48k.yaml")
    cur = exp["data"]["curriculum"]
    assert cur["steps"] == 12000
    assert {**exp, "data": {**exp["data"], "curriculum": None}} == ref
    train.check_curriculum(exp)
    gentle = load_config(CONFIGS / "world_gentle.yaml")
    phase = train.curriculum_cfg(exp)
    assert phase["sim"] == gentle["sim"]
    assert build.split_key(phase, "train") == build.split_key(gentle, "train")
    assert build.split_key(exp, "train") == build.split_key(load_config(BASE), "train")
