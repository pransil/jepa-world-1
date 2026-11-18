"""SPEC §8 M3: "loss decreases on the development set", on real rendered dev episodes.

Slow (a few hundred SPEC-shape training steps on CPU) and render (it builds the dev split with
MuJoCo's renderer into the test's own artifact root, never the shared one). Its fast siblings
are test_train.py::test_training_lowers_the_latent_loss_on_a_fixed_batch (the loss mechanics)
and test_train_episodes.py::test_train_runs_on_episode_source (the loader → train.run path).

The main criterion is the scale-free relerr/k (error relative to predicting the batch-mean
target, models/jepa.py::relative_error). The raw latent loss scales with the embedding spread,
which the variance guard grows (about 2x over this run under the guard settings of
docs/decisions/2026-11-17-far-collapse-guard-sum.md). So a raw-loss margin measures the guard as
much as the predictor. The raw loss is still required to fall, which is SPEC's literal claim.
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
# Measured on this setup (docs/milestones/M3.md): relerr 2.42 -> 0.38, raw 110 -> 31.
RELERR_RATIO = 0.5  # last-window mean relerr below half the first window's
RAW_RATIO = 0.8  # last-window mean raw latent loss below 0.8x the first window's


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def test_near_latent_loss_decreases_on_the_dev_set(tmp_path):
    cfg = load_config(CONFIGS / "m3_near_dev.yaml", CPU_RUN)
    build.build_split(cfg, "dev")  # real rendering, into this test's tmp artifact root
    horizons = train.check_config(cfg)
    assert horizons == [1, 2, 3, 4, 5]
    # train.run seeds itself from cfg["seed"], and the loader's sampler is seeded from it too.
    batches = train.make_batches(cfg, horizons)
    train.run(cfg, tmp_path, batches=batches, device=torch.device("cpu"))

    rows = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == list(range(1, cfg["train"]["steps"] + 1))
    raw = [sum(r[f"loss/k{k}"] for k in horizons) for r in rows]
    rel = [_mean(r[f"relerr/k{k}"] for k in horizons) for r in rows]
    assert all(math.isfinite(v) for v in raw + rel)
    first, last = _mean(rel[:WINDOW]), _mean(rel[-WINDOW:])
    assert last < RELERR_RATIO * first, ("relerr", first, last)
    assert last < 1.0, ("relerr: no better than predicting the batch-mean target", last)
    first, last = _mean(raw[:WINDOW]), _mean(raw[-WINDOW:])
    assert last < RAW_RATIO * first, ("raw latent", first, last)
    # Not by collapse: the guarded embedding keeps a batch spread at the end.
    std = _mean(r["emb_std/L1"] for r in rows[-WINDOW:])
    assert std > 0.1, std
