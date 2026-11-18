"""Ground-truth fields in loader batches (with_gt=True), for M5's goal head and M6's metrics."""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest
import torch
from fake_episodes import SHORT, Recorder, decode, fake_episode
from tiny import BASE

from jepa_world import metrics
from jepa_world.data import build
from jepa_world.data.dataset import GT_KEYS, batches, load_dir, window_frames, window_gt
from jepa_world.models.goal import displacement_targets
from jepa_world.sim.io import load_episode
from jepa_world.utils.config import load_config

C = 4
HORIZONS = [1, 2, 3, 4, 5, 10, 20]
K_MAX = max(HORIZONS)


def patterned(sim_cfg, ts, bs, **_):
    """A fake episode whose camera and flags differ by frame and by episode, so a misaligned
    or mixed-up window can't match by accident."""
    ep = fake_episode(sim_cfg, ts, bs)
    t = np.arange(len(ep.t))
    return dataclasses.replace(
        ep,
        cam_pos=np.stack([ts + 0.001 * t, bs - 0.01 * t], axis=1),
        out_of_view=(t + ts + bs) % 3 == 0,
        occluded=(t + 2 * bs) % 5 == 0,
    )


def _build(tmp_path_factory, *overrides, simulate=patterned):
    cfg = load_config(BASE, [*SHORT, *overrides])
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JEPA_ARTIFACTS", str(tmp_path_factory.mktemp("artifacts")))
        path = build.build_split(cfg, "train", simulate=Recorder(simulate))
    episodes = {}
    for f in path.glob("*.npz"):
        ep = load_episode(f)
        episodes[(ep.meta["terrain_seed"] % 251, ep.meta["ball_seed"] % 251)] = ep
    return load_dir(path), episodes


@pytest.fixture(scope="module")
def short(tmp_path_factory):
    """SHORT train split: 3 terrains x 2 balls, T=30, S=8."""
    return _build(tmp_path_factory)


@pytest.fixture(scope="module")
def full_length(tmp_path_factory):
    """Full 60-frame episodes (tiny 4x4 frames), 1 terrain x 2 balls."""
    return _build(
        tmp_path_factory,
        "sim.episode_frames=60",
        "sim.frame_size=4",
        "data.splits.train.terrain_seeds=[1000, 1]",
    )


def gt_batches(es, batch_size=16, seed=0):
    return batches(es, batch_size, context_frames=C, k_max=K_MAX, seed=seed, with_gt=True)


def test_gt_batch_has_every_field_with_its_shape_and_dtype(short):
    es, _ = short
    batch = next(gt_batches(es, batch_size=5))
    assert set(batch) == {"frames", *GT_KEYS}
    F = C + K_MAX
    assert batch["frames"].shape == (5, F, 3, 8, 8) and batch["frames"].dtype == torch.float32
    for key in ("ball_pos", "ball_vel", "cam_pos"):
        assert batch[key].shape == (5, F, 2) and batch[key].dtype == torch.float32, key
    for key in ("out_of_view", "occluded"):
        assert batch[key].shape == (5, F) and batch[key].dtype == torch.bool, key
    assert all(v.device.type == "cpu" for v in batch.values())


def test_gt_fields_are_on_the_frames_time_axis(short):
    """Each GT value is the episode's own array at the frame index the pixels decode to."""
    es, episodes = short
    for batch in itertools.islice(gt_batches(es), 10):
        idx, tcode, bcode = decode(batch["frames"])
        for b in range(len(idx)):
            ep = episodes[(tcode[b, 0], bcode[b, 0])]
            for key in ("ball_pos", "ball_vel", "cam_pos"):
                want = getattr(ep, key)[idx[b]].astype(np.float32)
                np.testing.assert_array_equal(batch[key][b].numpy(), want, err_msg=key)
            for key in ("out_of_view", "occluded"):
                np.testing.assert_array_equal(batch[key][b].numpy(), getattr(ep, key)[idx[b]])


def test_default_output_is_unchanged(short):
    """with_gt=False (the default) still yields the bare frames tensor, and with_gt=True draws
    the same windows from the same seed."""
    es, _ = short
    plain = batches(es, 4, context_frames=C, k_max=K_MAX, seed=5)
    rich = gt_batches(es, batch_size=4, seed=5)
    for a, b in itertools.islice(zip(plain, rich, strict=True), 3):
        assert isinstance(a, torch.Tensor)
        assert torch.equal(a, b["frames"])


def test_gt_window_never_runs_past_the_episode_at_k_max_20(full_length):
    es, episodes = full_length
    assert es.T == 60
    first, t_seen = set(), set()
    for batch in itertools.islice(gt_batches(es), 40):
        idx, _, _ = decode(batch["frames"])
        assert idx.min() >= 0 and idx.max() <= 59
        first |= set(idx[:, 0].tolist())
        t_seen |= set(idx[:, C - 1].tolist())
    assert first == set(range(0, 37))  # window starts 0..36: every one reachable
    assert t_seen == set(range(3, 40))


@pytest.mark.parametrize("t", [C - 2, 30 - K_MAX, -1])
def test_window_gt_refuses_starts_outside_the_episode(short, t):
    es, _ = short
    with pytest.raises(IndexError):
        window_gt(es, np.array([0]), np.array([t]), C, K_MAX)


@pytest.mark.parametrize("ep", [-1, 6])
def test_windows_refuse_an_episode_index_outside_the_set(short, ep):
    """numpy and torch wrap -1 to the last episode: GT and frames from the wrong episode."""
    es, _ = short
    for fn in (window_gt, window_frames):
        with pytest.raises(IndexError, match="episode index"):
            fn(es, np.array([ep]), np.array([C - 1]), C, K_MAX)


def test_loader_refuses_gt_of_the_wrong_kind(tmp_path_factory):
    """Flags must be bool and positions floating, in every file, not just the first."""

    def int_flags(sim_cfg, ts, bs, **_):
        ep = patterned(sim_cfg, ts, bs)
        return (
            ep
            if (ts, bs) == (1000, 0)
            else dataclasses.replace(ep, out_of_view=ep.out_of_view.astype(np.uint8))
        )

    with pytest.raises(ValueError, match="out_of_view is uint8"):
        _build(tmp_path_factory, simulate=int_flags)


def test_m5_and_m6_consumers_accept_gt_batches(short):
    """The goal head's displacement targets match float64 truth to well under a millimetre, and
    M6's visibility metric takes the flags at t."""
    es, episodes = short
    batch = next(gt_batches(es, batch_size=8))
    disp = displacement_targets(batch["ball_pos"], C, HORIZONS)
    assert disp.shape == (8, len(HORIZONS), 2) and disp.dtype == torch.float32
    idx, tcode, bcode = decode(batch["frames"])
    for b in range(8):
        pos = episodes[(tcode[b, 0], bcode[b, 0])].ball_pos
        t = idx[b, C - 1]
        want = np.stack([pos[t + k] - pos[t] for k in HORIZONS])
        np.testing.assert_allclose(disp[b].numpy(), want, atol=1e-5)
    m = np.zeros((8, 2))
    out = metrics.visibility(
        m,
        m + 1.0,
        m,
        {key: batch[key][:, C - 1].numpy() for key in ("out_of_view", "occluded")},
    )
    assert {"out_of_view", "occluded", "hidden"} <= set(out)
