"""The episode loader: batch layout, leakage, on-disk split isolation (SPEC §3.5, §8 M2)."""

from __future__ import annotations

import itertools
import json
import shutil

import numpy as np
import pytest
import torch
from fake_episodes import SHORT, Recorder, decode
from tiny import BASE

from jepa_world.data import build
from jepa_world.data.dataset import (
    EpisodeSet,
    batches,
    load_dir,
    load_split,
    valid_starts,
    window_frames,
)
from jepa_world.data.splits import SPLITS, LeakageError
from jepa_world.models.jepa import context_stack, target_stack
from jepa_world.utils.config import load_config

C = 4
HORIZONS = [1, 2, 3, 4, 5, 10, 20]
K_MAX = max(HORIZONS)


def short_cfg(*overrides: str) -> dict:
    return load_config(BASE, [*SHORT, *overrides])


@pytest.fixture(scope="module")
def train_set(tmp_path_factory) -> EpisodeSet:
    """Train split of SHORT (3 terrains x 2 balls, T=30, S=8), built once for the module."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JEPA_ARTIFACTS", str(tmp_path_factory.mktemp("artifacts")))
        return load_dir(build.build_split(short_cfg(), "train", simulate=Recorder()))


def _unstack(stack: torch.Tensor, n: int) -> torch.Tensor:
    """(B, 3n, H, W) -> (B, n, 3, H, W), undoing the model's channel stacking."""
    b, _, h, w = stack.shape
    return stack.reshape(b, n, 3, h, w)


def test_episode_set_holds_the_split(train_set):
    es = train_set
    assert es.split == "train" and es.n == 6 and es.T == 30
    assert es.frames.shape == (6, 30, 3, 8, 8) and es.frames.dtype == torch.uint8
    assert es.ball_pos.shape == es.ball_vel.shape == (6, 30, 2)
    assert es.out_of_view.shape == (6, 30) and es.out_of_view.dtype == bool
    assert sorted(set(es.terrain_seed.tolist())) == [1000, 1001, 1002]
    assert es.fps == 15
    # frames are stored channels-first but are the same pixels the file holds
    idx, tcode, bcode = decode(es.frames.float() / 255)
    assert (idx == np.arange(30)).all()
    assert (tcode == (es.terrain_seed % 251)[:, None]).all()


def test_batch_layout_matches_m3_handshake(train_set):
    """LANES.md handshake M3 -> M2: (B, C + k_max, 3, H, W) float32 in [0, 1], index C-1 = t."""
    batch = next(batches(train_set, batch_size=5, context_frames=C, k_max=K_MAX, seed=0))
    assert batch.shape == (5, C + K_MAX, 3, 8, 8)
    assert batch.dtype == torch.float32 and batch.device.type == "cpu"
    assert 0.0 <= batch.min() and batch.max() <= 1.0
    idx, _, _ = decode(batch)
    t = idx[:, C - 1]
    assert (idx == t[:, None] + np.arange(-(C - 1), K_MAX + 1)).all()


def test_context_never_includes_frames_at_or_after_t_plus_k(train_set):
    """SPEC §8 M2. Decoded through the model's own context_stack / target_stack."""
    it = batches(train_set, batch_size=16, context_frames=C, k_max=K_MAX, seed=1)
    for batch in itertools.islice(it, 20):
        ctx, _, _ = decode(_unstack(context_stack(batch, C), C))
        t = ctx[:, -1]
        assert (ctx == t[:, None] + np.arange(-(C - 1), 1)).all()  # exactly t-C+1 .. t
        for k in HORIZONS:
            assert (ctx < (t + k)[:, None]).all(), k
            tgt, _, _ = decode(_unstack(target_stack(batch, C, k), C))
            assert (tgt == (t + k)[:, None] + np.arange(-(C - 1), 1)).all(), k  # ends at t+k


def test_window_stays_within_one_episode(train_set):
    batch = next(batches(train_set, batch_size=64, context_frames=C, k_max=K_MAX, seed=2))
    _, tcode, bcode = decode(batch)
    assert (tcode == tcode[:, :1]).all() and (bcode == bcode[:, :1]).all()


def test_valid_starts_bounds():
    assert valid_starts(60, 4, 20).tolist() == list(range(3, 40))
    assert valid_starts(60, 4, 5).tolist() == list(range(3, 55))
    assert valid_starts(24, 4, 20).tolist() == [3]
    with pytest.raises(ValueError, match="no valid"):
        valid_starts(23, 4, 20)
    with pytest.raises(ValueError):
        valid_starts(60, 0, 5)


def test_sampling_reaches_every_window(train_set):
    starts = valid_starts(train_set.T, C, K_MAX)
    seen = set()
    for batch in itertools.islice(
        batches(train_set, batch_size=16, context_frames=C, k_max=K_MAX, seed=3), 50
    ):
        idx, tcode, bcode = decode(batch)
        seen |= set(zip(tcode[:, 0], bcode[:, 0], idx[:, C - 1], strict=True))
    assert len(seen) == train_set.n * len(starts)


@pytest.mark.parametrize("t", [C - 2, 30 - K_MAX, -1])
def test_window_frames_refuses_starts_outside_the_episode(train_set, t):
    """A negative index would silently wrap to the episode's END: future frames in the context."""
    with pytest.raises(IndexError):
        window_frames(train_set.frames, np.array([0]), np.array([t]), C, K_MAX)


def test_same_seed_same_batches(train_set):
    def take(seed):
        it = batches(train_set, batch_size=4, context_frames=C, k_max=K_MAX, seed=seed)
        return list(itertools.islice(it, 3))

    a, b, c = take(7), take(7), take(8)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(a, c, strict=True))


def test_loaded_splits_share_no_terrain():
    """SPEC §8 M2: splits never share terrain seeds, checked on the data actually on disk."""
    cfg = short_cfg()
    for s in SPLITS:
        build.build_split(cfg, s, simulate=Recorder())
    sets = {s: load_split(cfg, s) for s in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        assert not set(sets[a].terrain_seed.tolist()) & set(sets[b].terrain_seed.tolist())
        assert not set(sets[a].terrain_sha) & set(sets[b].terrain_sha)


def _smuggle_dev_episode_into_train(cfg) -> None:
    dev = build.build_split(cfg, "dev", simulate=Recorder())
    train = build.build_split(cfg, "train", simulate=Recorder())
    row = json.loads((dev / "manifest.json").read_text())["episodes"][0]
    shutil.copy(dev / row["file"], train / row["file"])
    manifest = json.loads((train / "manifest.json").read_text())
    manifest["episodes"].append(row)
    (train / "manifest.json").write_text(json.dumps(manifest))


def test_loader_rejects_an_episode_from_another_split():
    cfg = short_cfg()
    _smuggle_dev_episode_into_train(cfg)
    with pytest.raises(LeakageError, match="train"):
        load_split(cfg, "train")


def test_loader_rejects_a_train_ball_seed_on_disk_in_test():
    cfg = short_cfg()
    path = build.build_split(cfg, "test", simulate=Recorder())
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["episodes"][0]["ball_seed"] = 0  # a train ball seed
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(LeakageError, match="ball_seed"):
        load_split(cfg, "test")


def test_loader_rejects_a_file_that_disagrees_with_its_manifest():
    path = build.build_split(short_cfg(), "dev", simulate=Recorder())
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["episodes"][0]["terrain_seed"] = 1
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(LeakageError, match="manifest"):
        load_dir(path)


def test_missing_split_error_names_the_build_command():
    with pytest.raises(FileNotFoundError, match=r"jepa_world\.data\.build.*--split test"):
        load_split(short_cfg(), "test")
