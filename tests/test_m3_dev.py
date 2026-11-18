"""SPEC §8 M3: "loss decreases on the development set", on real rendered dev episodes.

Slow (a few hundred SPEC-shape training steps on CPU) and render (it builds the dev split with
MuJoCo's renderer into the test's own artifact root, never the shared one). Its fast siblings
are test_train.py::test_training_lowers_the_latent_loss_on_a_fixed_batch (the loss mechanics)
and test_train_episodes.py::test_train_runs_on_episode_source (the loader → train.run path).
"""

from __future__ import annotations

import json
import math

import pytest
import torch
from tiny import CONFIGS

from jepa_world import train
from jepa_world.data import build
from jepa_world.utils.config import load_config

pytestmark = [pytest.mark.slow, pytest.mark.render]

# configs/m3_near_dev.yaml, shrunk to fit a CPU test: same model, data and loss; fewer, smaller steps.
CPU_RUN = [
    "train.device=cpu",
    "train.batch_size=4",
    "train.steps=250",
    "train.log_every=50",
    "train.ckpt_every=100000",
]
WINDOW = 25  # steps averaged at each end, so one noisy batch can't decide the test
MARGIN = 0.5  # the last window's mean must be below (1 - MARGIN) x the first window's


def test_near_latent_loss_decreases_on_the_dev_set(tmp_path):
    cfg = load_config(CONFIGS / "m3_near_dev.yaml", CPU_RUN)
    build.build_split(cfg, "dev")  # real rendering, into this test's tmp artifact root
    horizons = train.check_config(cfg)
    assert horizons == [1, 2, 3, 4, 5]
    torch.manual_seed(cfg["seed"])
    train.run(cfg, tmp_path, batches=train.make_batches(cfg, horizons), device=torch.device("cpu"))

    rows = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, cfg["train"]["steps"] + 1))
    latent = [sum(r[f"loss/k{k}"] for k in horizons) for r in rows]
    assert all(math.isfinite(v) for v in latent)
    first = sum(latent[:WINDOW]) / WINDOW
    last = sum(latent[-WINDOW:]) / WINDOW
    assert last < (1 - MARGIN) * first, (first, last)
    # Not by collapse: the guarded embedding keeps a batch spread at the end.
    std = sum(r["emb_std/L1"] for r in rows[-WINDOW:]) / WINDOW
    assert std > 0.1, std
