"""train.py on the M2 episode loader (data.source: episodes)."""

from __future__ import annotations

import contextlib
import itertools
import json

import pytest
import torch
from fake_episodes import SMALL_SPLITS, Recorder, decode
from tiny import BASE, NEAR, TINY, tiny_cfg

from jepa_world import train
from jepa_world.data import build
from jepa_world.data.splits import LeakageError


def episodes_cfg(*overrides: str) -> dict:
    return tiny_cfg("data.source=episodes", "data.split=dev", *SMALL_SPLITS, *overrides)


def test_make_batches_reads_the_configured_split():
    cfg = episodes_cfg()
    build.build_split(cfg, "dev", simulate=Recorder())
    it = train.make_batches(cfg, NEAR)
    C = cfg["model"]["context_frames"]
    for batch in itertools.islice(it, 5):
        assert batch.shape == (2, C + max(NEAR), 3, 16, 16) and batch.dtype == torch.float32
        idx, tcode, _ = decode(batch)
        assert set(tcode[:, 0]) <= {0, 1}  # dev terrains 0 and 1
        assert (idx[:, 0] >= 0).all() and (idx[:, -1] <= cfg["sim"]["episode_frames"] - 1).all()


def test_train_runs_on_episode_source(tmp_path):
    cfg = episodes_cfg("train.steps=3")
    build.build_split(cfg, "dev", simulate=Recorder())
    train.check_config(cfg)
    final = train.run(cfg, tmp_path, device=torch.device("cpu"))
    assert {f"loss/k{k}" for k in NEAR} | {"loss/total"} <= set(final)
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(s)["step"] for s in lines] == [1, 2, 3]


def test_training_on_the_test_split_is_refused():
    cfg = episodes_cfg("data.split=test")
    build.build_split(cfg, "test", simulate=Recorder())  # even when it is built
    with pytest.raises(LeakageError, match="test"):
        train.check_config(cfg)
    with pytest.raises(LeakageError):
        train.make_batches(cfg, NEAR)


def test_cli_loads_the_split_before_taking_the_gpu_lock(monkeypatch):
    """Decompressing a split is CPU work: other lanes shouldn't wait on the MPS lock for it."""
    cfg_args = [
        "data.source=episodes",
        "data.split=dev",
        *SMALL_SPLITS,
        "train.steps=1",
    ]
    build.build_split(episodes_cfg(), "dev", simulate=Recorder())
    events = []
    real_load = train.load_split
    monkeypatch.setattr(train, "load_split", lambda *a: events.append("load") or real_load(*a))

    @contextlib.contextmanager
    def fake_lock(label, **kw):
        events.append("lock")
        yield

    monkeypatch.setattr(train, "gpu_lock", fake_lock)
    sets = [arg for s in [*TINY, *cfg_args] for arg in ("--set", s)]
    assert train.main(["--config", str(BASE), *sets]) == 0
    assert events == ["load", "lock"]


def test_unbuilt_split_fails_check_config_with_the_build_command():
    with pytest.raises(FileNotFoundError, match=r"jepa_world\.data\.build.*--split dev"):
        train.check_config(episodes_cfg())


def test_unknown_source_is_refused():
    with pytest.raises(ValueError, match="nonsense"):
        train.check_config(tiny_cfg("data.source=nonsense"))
