"""Building a split into the artifact root: write-once, recorded, regenerable (SPEC §3.5)."""

from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest
from fake_episodes import SHORT, Recorder, fake_episode, physics_episode
from tiny import BASE

from jepa_world.data import build
from jepa_world.data.splits import LeakageError, split_episodes
from jepa_world.sim.episode import GT_FIELDS, simulate_episode
from jepa_world.sim.io import load_episode
from jepa_world.utils import paths
from jepa_world.utils.config import load_config


def short_cfg(*overrides: str) -> dict:
    return load_config(BASE, [*SHORT, *overrides])


def test_build_writes_every_episode_and_a_manifest():
    cfg = short_cfg()
    sim = Recorder()
    path = build.build_split(cfg, "train", simulate=sim)
    expected = split_episodes(cfg["data"], "train")
    assert sim.calls == expected
    assert path == paths.dataset_dir(build.split_key(cfg, "train"), "episodes-train")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["split"] == "train"
    assert manifest["format_version"] == cfg["data"]["format_version"]
    rows = manifest["episodes"]
    assert [(r["terrain_seed"], r["ball_seed"]) for r in rows] == expected
    for r in rows:
        ep = load_episode(path / r["file"])
        assert (ep.meta["terrain_seed"], ep.meta["ball_seed"]) == (
            r["terrain_seed"],
            r["ball_seed"],
        )
        assert r["terrain_sha"] == build.terrain_sha(ep.terrain)
        assert ep.frames.shape == (30, 8, 8, 3)
    assert json.loads((path / "config.json").read_text()) == json.loads(
        json.dumps(build.split_key(cfg, "train"), default=str)
    )


def test_build_is_write_once():
    cfg = short_cfg()
    first = build.build_split(cfg, "dev", simulate=Recorder())
    again = Recorder()
    assert build.build_split(cfg, "dev", simulate=again) == first
    assert again.calls == []


def test_split_key_changes_with_seeds_split_sim_and_format():
    cfg = short_cfg()
    h = paths.config_hash
    base = h(build.split_key(cfg, "train"))
    assert h(build.split_key(cfg, "dev")) != base
    for override in (
        "data.splits.train.ball_seeds=[0, 3]",
        "data.splits.train.terrain_seeds=[1001, 3]",
        "sim.camera.lerp_z=0.05",
        "sim.generator_version=99",
        "data.format_version=99",
    ):
        assert h(build.split_key(short_cfg(override), "train")) != base, override
    # keys that don't change what is on disk don't change the key
    for override in ("data.split=dev", "seed=7", "train.batch_size=8"):
        assert h(build.split_key(short_cfg(override), "train")) == base, override


def test_build_records_run_metadata():
    path = build.build_split(short_cfg(), "dev", simulate=Recorder())
    meta = json.loads((path / "meta.json").read_text())
    for key in ("git", "platform", "libraries", "build_seconds", "episodes"):
        assert key in meta, key
    assert meta["episodes"] == 4


def test_build_refuses_episodes_without_frames_and_leaves_nothing():
    cfg = short_cfg()

    def no_frames(sim_cfg, ts, bs, **_):
        return dataclasses.replace(fake_episode(sim_cfg, ts, bs), frames=None)

    with pytest.raises(ValueError, match="frames"):
        build.build_split(cfg, "dev", simulate=no_frames)
    assert not paths.dataset_dir(build.split_key(cfg, "dev"), "episodes-dev").exists()


def test_build_refuses_an_episode_from_the_wrong_seeds():
    def wrong_seed(sim_cfg, ts, bs, **_):
        return fake_episode(sim_cfg, ts + 1, bs)

    with pytest.raises(ValueError, match="seed"):
        build.build_split(short_cfg(), "dev", simulate=wrong_seed)


def test_build_refuses_a_leaky_split_config():
    cfg = short_cfg("data.splits.train.terrain_seeds=[1, 3]")  # 1 is a dev seed
    with pytest.raises(LeakageError):
        build.build_split(cfg, "train", simulate=Recorder())


def test_episode_regenerates_bitwise_from_manifest_seeds():
    """SPEC §3.5: store the generator config and seeds so episodes can be regenerated exactly."""
    cfg = short_cfg("data.splits.dev.terrain_seeds=[3, 1]", "data.splits.dev.ball_seeds=[1, 1]")
    path = build.build_split(cfg, "dev", simulate=physics_episode)
    stored = json.loads((path / "config.json").read_text())
    (row,) = json.loads((path / "manifest.json").read_text())["episodes"]
    ep = load_episode(path / row["file"])
    again = simulate_episode(stored["sim"], row["terrain_seed"], row["ball_seed"], render=False)
    for name in GT_FIELDS:
        np.testing.assert_array_equal(getattr(ep, name), getattr(again, name), err_msg=name)


@pytest.mark.render
def test_real_build_frames_match_a_fresh_render():
    cfg = short_cfg(
        "sim.frame_size=16",
        "data.splits.dev.terrain_seeds=[2, 1]",
        "data.splits.dev.ball_seeds=[0, 1]",
    )
    path = build.build_split(cfg, "dev")  # the default simulate: real MuJoCo rendering
    (row,) = json.loads((path / "manifest.json").read_text())["episodes"]
    ep = load_episode(path / row["file"])
    fresh = simulate_episode(copy.deepcopy(cfg["sim"]), 2, 0, render=True)
    assert ep.frames.shape == (30, 16, 16, 3) and ep.frames.dtype == np.uint8
    np.testing.assert_array_equal(ep.frames, fresh.frames)
    assert ep.frames.max() > 0
