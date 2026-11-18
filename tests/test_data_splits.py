"""Terrain-seed splits (SPEC §3.5) and the config-level leakage check (SPEC §8 M2)."""

from __future__ import annotations

import copy
import itertools

import pytest
from tiny import BASE

from jepa_world.data.splits import (
    SPLITS,
    LeakageError,
    ball_seeds,
    check_disjoint,
    check_split_config,
    split_episodes,
    terrain_seeds,
)
from jepa_world.utils.config import load_config


@pytest.fixture(scope="module")
def data_cfg():
    return load_config(BASE)["data"]


def test_split_sizes_match_config(data_cfg):
    sizes = {s: (len(terrain_seeds(data_cfg, s)), len(ball_seeds(data_cfg, s))) for s in SPLITS}
    assert sizes == {"dev": (10, 4), "train": (500, 4), "test": (50, 4)}
    eps = split_episodes(data_cfg, "train")
    assert len(eps) == len(set(eps)) == 2000
    assert set(eps) == set(itertools.product(terrain_seeds(data_cfg, "train"), range(4)))


def test_splits_never_share_terrain_seeds(data_cfg):
    """SPEC §8 M2: splits never share terrain seeds (config level; the on-disk check is in
    test_data_dataset.py::test_loaded_splits_share_no_terrain)."""
    check_split_config(data_cfg)
    seeds = {s: set(terrain_seeds(data_cfg, s)) for s in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        assert not seeds[a] & seeds[b], (a, b)


def test_overlapping_split_config_is_refused(data_cfg):
    bad = copy.deepcopy(data_cfg)
    bad["splits"]["train"]["terrain_seeds"] = [5, 10]  # 5..14 overlaps dev's 0..9
    with pytest.raises(LeakageError, match=r"dev.*train|train.*dev") as e:
        check_split_config(bad)
    assert "5" in str(e.value)


def test_check_disjoint_names_the_seed_and_both_splits():
    with pytest.raises(LeakageError, match=r"\b7\b") as e:
        check_disjoint({"a": [1, 7], "b": [7, 9], "c": [2]})
    assert "'a'" in str(e.value) and "'b'" in str(e.value)
    check_disjoint({"a": [1], "b": [2]})  # no overlap: no error


def test_test_ball_seeds_are_disjoint_from_train(data_cfg):
    assert not set(ball_seeds(data_cfg, "test")) & set(ball_seeds(data_cfg, "train"))
    bad = copy.deepcopy(data_cfg)
    bad["splits"]["test"]["ball_seeds"] = [3, 4]  # 3 is a train ball seed
    with pytest.raises(LeakageError, match="ball seed"):
        check_split_config(bad)


def test_train_and_test_terrains_avoid_m1_tuning_range(data_cfg):
    """M1 tuned its event rates on terrain seeds 0-199 (docs/milestones/M1.md)."""
    tuned = set(range(200))
    for s in ("train", "test"):
        assert not set(terrain_seeds(data_cfg, s)) & tuned, s


@pytest.mark.parametrize("spec", [[0, 0], [-1, 3], [0.5, 2], [1, 2, 3], "0-9"])
def test_malformed_seed_range_is_refused(data_cfg, spec):
    bad = copy.deepcopy(data_cfg)
    bad["splits"]["dev"]["terrain_seeds"] = spec
    with pytest.raises(ValueError, match="dev"):
        terrain_seeds(bad, "dev")


def test_unknown_or_missing_split_is_refused(data_cfg):
    with pytest.raises(ValueError, match="valid"):
        terrain_seeds(data_cfg, "validation")
    bad = copy.deepcopy(data_cfg)
    del bad["splits"]["test"]
    with pytest.raises(ValueError, match="test"):
        check_split_config(bad)
