"""Terrain: seeded heightfield profile h(x), constant along y (SPEC §3.1). Pure numpy."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from jepa_world.sim.terrain import Terrain, sample_terrain
from jepa_world.utils.config import load_config


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(Path(__file__).parents[1] / "configs" / "base.yaml")["sim"]


def test_same_seed_gives_identical_terrain(sim_cfg):
    a, b = sample_terrain(sim_cfg, 7), sample_terrain(sim_cfg, 7)
    np.testing.assert_array_equal(a.heights, b.heights)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.params == b.params


def test_different_seeds_give_different_terrain(sim_cfg):
    assert not np.array_equal(
        sample_terrain(sim_cfg, 1).heights, sample_terrain(sim_cfg, 2).heights
    )


def test_columns_span_the_configured_length(sim_cfg):
    t = sample_terrain(sim_cfg, 0)
    n, half = sim_cfg["terrain"]["columns"], sim_cfg["terrain"]["length"] / 2
    assert t.x.shape == t.heights.shape == (n,)
    assert t.x[0] == -half and t.x[-1] == half
    np.testing.assert_allclose(np.diff(t.x), 2 * half / (n - 1))
    assert np.isfinite(t.heights).all()


def test_height_at_hits_nodes_and_is_linear_between(sim_cfg):
    t = sample_terrain(sim_cfg, 3)
    np.testing.assert_allclose(t.height_at(t.x), t.heights, atol=1e-12)
    mid = (t.x[:-1] + t.x[1:]) / 2
    np.testing.assert_allclose(t.height_at(mid), (t.heights[:-1] + t.heights[1:]) / 2, atol=1e-12)


def test_heights_are_exactly_what_the_float32_hfield_encodes(sim_cfg):
    """MuJoCo stores hfield data as float32. The terrain's heights must be the decoded values, so
    the geometry oracle in the tests is MuJoCo's surface exactly, not an approximation of it."""
    t = sample_terrain(sim_cfg, 4)
    data, zmin, elevation = t.hfield()
    assert data.dtype == np.float32 and data.min() >= 0.0 and data.max() <= 1.0
    np.testing.assert_array_equal(zmin + data.astype(np.float64) * elevation, t.heights)


def _away_from_walls_and_ledges(t, tc):
    """Segment mask: both ends inside the walls and clear of every ledge transition."""
    ok = np.abs(t.x) <= tc["length"] / 2 - tc["wall_width"]
    for lx in t.params["ledge_x"]:
        ok &= np.abs(t.x - lx) > tc["ledge_width"]
    return ok[:-1] & ok[1:]


def test_slope_is_capped_away_from_walls_and_ledges(sim_cfg):
    tc = sim_cfg["terrain"]
    for seed in range(5):
        t = sample_terrain(sim_cfg, seed)
        slopes = np.diff(t.heights) / np.diff(t.x)
        assert np.abs(slopes[_away_from_walls_and_ledges(t, tc)]).max() <= tc["max_slope"] + 1e-6


def test_ledges_are_steps_of_the_configured_size_away_from_the_walls(sim_cfg):
    tc = sim_cfg["terrain"]
    half, margin = tc["length"] / 2, tc["wall_width"] + tc["ledge_gap"]
    for seed in range(10):
        t = sample_terrain(sim_cfg, seed)
        lx, lh = np.array(t.params["ledge_x"]), np.array(t.params["ledge_height"])
        assert tc["ledges"][0] <= len(lx) <= tc["ledges"][1]
        assert (np.abs(lx) <= half - margin).all()
        assert ((np.abs(lh) >= tc["ledge_height"][0]) & (np.abs(lh) <= tc["ledge_height"][1])).all()
        assert (np.diff(lx) >= tc["ledge_gap"]).all(), "ledges must not merge into one cliff"
        # The step is sampled on the column grid, so measure one column beyond each end of it.
        reach = tc["ledge_width"] / 2 + (t.x[1] - t.x[0])
        for x_s, h_s in zip(lx, lh, strict=True):
            # Across one ledge the height changes by its step, give or take the waves' slope.
            dh = t.height_at(x_s + reach) - t.height_at(x_s - reach)
            assert abs(dh - h_s) <= tc["max_slope"] * 2 * reach + 1e-6, (seed, x_s, h_s, dh)


def test_relief_is_drawn_from_the_configured_range_then_slope_capped(sim_cfg):
    lo, hi = sim_cfg["terrain"]["relief"]
    for seed in range(5):
        p = sample_terrain(sim_cfg, seed).params
        assert lo <= p["target_relief"] <= hi
        assert p["slope_scale"] <= 1.0


def test_too_many_ledges_for_the_length_raises(sim_cfg):
    cfg = copy.deepcopy(sim_cfg)
    cfg["terrain"]["length"] = 12.0  # 12 - 2 * (2 + 1) = 6 m for up to 5 ledges: 1.2 m bins, OK
    sample_terrain(cfg, 0)
    cfg["terrain"]["length"] = 10.0  # 4 m for 5 ledges: 0.8 m bins < 1 m gap
    with pytest.raises(ValueError, match="too short"):
        sample_terrain(cfg, 0)


def test_end_walls_rise_wall_height_above_the_terrain_inside_them(sim_cfg):
    """Each wall adds wall_height on top of the profile. Across the wall zone the waves change by
    at most max_slope * wall_width, and there are no ledges there."""
    tc = sim_cfg["terrain"]
    half, ww = tc["length"] / 2, tc["wall_width"]
    for seed in range(5):
        t = sample_terrain(sim_cfg, seed)
        lower = tc["wall_height"] - tc["max_slope"] * ww
        assert t.heights[0] - t.height_at(-half + ww) >= lower - 1e-6
        assert t.heights[-1] - t.height_at(half - ww) >= lower - 1e-6


def test_params_record_everything_needed_to_regenerate(sim_cfg):
    p = sample_terrain(sim_cfg, 5).params
    assert p["seed"] == 5
    for key in (
        "amplitude",
        "wavelength",
        "phase",
        "slope_scale",
        "relief",
        "target_relief",
        "ledge_x",
        "ledge_height",
    ):
        assert key in p
    assert len(p["amplitude"]) == sim_cfg["terrain"]["waves"]


def test_gap_on_flat_ground():
    t = Terrain(x=np.linspace(-5, 5, 11), heights=np.zeros(11), params={})
    np.testing.assert_allclose(t.gap(np.array([[0.0, 1.0], [2.5, 0.25]]), 0.25), [0.75, 0.0])
    assert t.gap(np.array([0.0, 0.1]), 0.25) < 0  # penetrating


def test_gap_is_distance_to_the_polyline_not_vertical():
    t = Terrain(x=np.array([-1.0, 0.0, 1.0]), heights=np.array([1.0, 0.0, 1.0]), params={})
    # From (0, 1) the nearest surface points are on the two 45-degree sides, at distance 1/sqrt(2).
    np.testing.assert_allclose(t.gap(np.array([0.0, 1.0]), 0.0), 1 / np.sqrt(2))
    # A point below the surface has a negative gap.
    assert t.gap(np.array([0.5, 0.0]), 0.0) < 0


def test_slope_at_matches_segments():
    t = Terrain(x=np.array([0.0, 1.0, 2.0]), heights=np.array([0.0, 2.0, 1.0]), params={})
    np.testing.assert_allclose(t.slope_at(np.array([0.5, 1.5])), [2.0, -1.0])
