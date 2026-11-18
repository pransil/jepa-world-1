"""Episode export (SPEC §3.4 ground truth on disk) and the preview helpers."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from jepa_world.sim.episode import GT_FIELDS, simulate_episode
from jepa_world.sim.io import load_episode, save_episode

from jepa_world.sim import preview
from jepa_world.utils.config import load_config


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "base.yaml")["sim"]


@pytest.fixture(scope="module")
def episode(sim_cfg):
    return simulate_episode(sim_cfg, 2, 1, render=False)


def _assert_same(a, b):
    for name in GT_FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype, name
        np.testing.assert_array_equal(x, y, err_msg=name)
    assert a.ball == b.ball
    assert a.meta == b.meta
    assert a.terrain.params == b.terrain.params
    np.testing.assert_array_equal(a.terrain.x, b.terrain.x)
    np.testing.assert_array_equal(a.terrain.heights, b.terrain.heights)


def test_roundtrip_without_frames(tmp_path, episode):
    path = save_episode(tmp_path / "ep.npz", episode)
    back = load_episode(path)
    _assert_same(episode, back)
    assert back.frames is None and back.seg is None


def test_roundtrip_with_frames_and_segmentation(tmp_path, episode, sim_cfg):
    T, S = sim_cfg["episode_frames"], sim_cfg["frame_size"]
    rng = np.random.default_rng(0)
    ep = dataclasses.replace(
        episode,
        frames=rng.integers(0, 256, size=(T, S, S, 3), dtype=np.uint8),
        seg=rng.integers(-1, 3, size=(T, S, S)).astype(np.int16),
    )
    back = load_episode(save_episode(tmp_path / "ep.npz", ep))
    _assert_same(ep, back)
    np.testing.assert_array_equal(back.frames, ep.frames)
    assert back.frames.dtype == np.uint8
    np.testing.assert_array_equal(back.seg, ep.seg)


def test_file_loads_without_pickle(tmp_path, episode):
    path = save_episode(tmp_path / "ep.npz", episode)
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
    assert meta["meta"]["terrain_seed"] == 2 and meta["meta"]["ball_seed"] == 1
    assert meta["ball"]["radius"] == episode.ball.radius


def test_preview_stats_and_contact_sheet(sim_cfg, episode):
    stats = preview.episode_stats([episode, episode])
    for key in (
        "episodes",
        "contact_rate",
        "bounces_per_episode",
        "out_of_view_rate",
        "episodes_with_out_of_view",
    ):
        assert key in stats
    assert stats["episodes"] == 2
    assert 0.0 <= stats["contact_rate"] <= 1.0
    T, S = sim_cfg["episode_frames"], sim_cfg["frame_size"]
    frames = np.zeros((T, S, S, 3), dtype=np.uint8)
    sheet = preview.contact_sheet(frames, cols=10)
    assert sheet.dtype == np.uint8 and sheet.shape[2] == 3
    assert sheet.shape[0] >= (T // 10) * S and sheet.shape[1] >= 10 * S
