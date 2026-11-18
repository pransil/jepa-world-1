"""Evaluation metrics (SPEC §7.2, §7.3) on synthetic arrays with known answers. Pure numpy."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from jepa_world import baselines, metrics

HORIZONS = (1, 2, 5)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _by_k(fn, horizons=HORIZONS):
    return {k: fn(k) for k in horizons}


# --- position error, stats, ratio ---------------------------------------------------------------


def test_position_error_is_euclidean_in_world_units():
    pred = np.array([[0.0, 0.0], [1.0, 1.0]])
    truth = np.array([[3.0, 4.0], [1.0, 1.0]])
    np.testing.assert_allclose(metrics.position_error(pred, truth), [5.0, 0.0])


def test_position_error_rejects_mismatched_or_non_2d_positions():
    with pytest.raises(ValueError):
        metrics.position_error(np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        metrics.position_error(np.zeros((3, 3)), np.zeros((3, 3)))


def test_error_stats_match_the_baselines_keys():
    s = metrics.error_stats(np.arange(11.0))
    assert set(s) == {"mean", "median", "p90"}
    assert s["mean"] == 5.0 and s["median"] == 5.0 and s["p90"] == pytest.approx(9.0)
    assert all(type(v) is float for v in s.values())


def test_errors_by_horizon_uses_str_keys_and_per_k_errors():
    truth = _by_k(lambda k: np.zeros((4, 2)))
    pred = _by_k(lambda k: np.tile([0.0, float(k)], (4, 1)))
    out = metrics.errors_by_horizon(pred, truth)
    assert list(out) == ["1", "2", "5"]
    assert [out[str(k)]["mean"] for k in HORIZONS] == [1.0, 2.0, 5.0]


def test_ratio_is_one_when_the_model_is_persistence():
    rng = _rng()
    truth = _by_k(lambda k: rng.normal(size=(50, 2)))
    pers = _by_k(lambda k: rng.normal(size=(50, 2)))
    model = {k: v.copy() for k, v in pers.items()}
    ratio = metrics.ratio_to_persistence(model, pers, truth)
    assert ratio == {"1": 1.0, "2": 1.0, "5": 1.0}


def test_ratio_is_zero_for_a_perfect_model_and_none_for_a_perfect_persistence():
    rng = _rng(1)
    truth = _by_k(lambda k: rng.normal(size=(20, 2)))
    pers = _by_k(lambda k: truth[k] + 1.0)
    pers[2] = truth[2].copy()  # persistence exact at k=2: the ratio is undefined there
    ratio = metrics.ratio_to_persistence(truth, pers, truth)
    assert ratio["1"] == 0.0 and ratio["5"] == 0.0
    assert ratio["2"] is None


def test_ratio_is_the_ratio_of_mean_errors():
    truth = _by_k(lambda k: np.zeros((2, 2)))
    pers = _by_k(lambda k: np.array([[1.0, 0.0], [3.0, 0.0]]))  # mean error 2
    model = _by_k(lambda k: np.array([[0.0, 1.0], [0.0, 0.0]]))  # mean error 0.5
    assert metrics.ratio_to_persistence(model, pers, truth)["1"] == pytest.approx(0.25)


def test_every_horizon_must_be_scored_on_the_same_windows():
    truth = {1: np.zeros((10, 2)), 5: np.zeros((8, 2))}  # k=5 lost two windows
    with pytest.raises(ValueError, match="windows"):
        metrics.errors_by_horizon(truth, truth)
    with pytest.raises(ValueError, match="windows"):
        metrics.ratio_to_persistence(truth, truth, truth)


def test_horizon_sets_must_agree():
    a = _by_k(lambda k: np.zeros((3, 2)))
    b = _by_k(lambda k: np.zeros((3, 2)), horizons=(1, 2))
    with pytest.raises(ValueError, match="horizons"):
        metrics.ratio_to_persistence(a, b, a)


# --- calibration --------------------------------------------------------------------------------


def _gaussian(m=20_000, seed=0):
    rng = _rng(seed)
    mean = rng.normal(size=(m, 2))
    var = rng.uniform(0.01, 1.0, size=(m, 2))
    truth = mean + np.sqrt(var) * rng.standard_normal((m, 2))
    return mean, var, truth


def test_a_calibrated_gaussian_covers_the_nominal_fractions():
    mean, var, truth = _gaussian()
    cov = metrics.coverage(mean, var, truth)
    assert set(cov) == {"0.5", "0.9"}
    for p in (0.5, 0.9):
        for axis in ("x", "z", "joint"):
            assert cov[str(p)][axis] == pytest.approx(p, abs=0.015), (p, axis)


def test_overconfident_undercovers_and_underconfident_overcovers():
    mean, var, truth = _gaussian(seed=1)
    tight = metrics.coverage(mean, var * 0.25, truth)  # sigma halved
    loose = metrics.coverage(mean, var * 4.0, truth)  # sigma doubled
    for p in (0.5, 0.9):
        for axis in ("x", "z", "joint"):
            assert tight[str(p)][axis] < p - 0.05
            assert loose[str(p)][axis] > p + 0.05


def test_coverage_levels_are_configurable():
    mean, var, truth = _gaussian(m=2000, seed=2)
    assert set(metrics.coverage(mean, var, truth, levels=(0.68,))) == {"0.68"}


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_coverage_rejects_non_positive_or_non_finite_variance(bad):
    mean, var, truth = _gaussian(m=10)
    var[3, 1] = bad
    with pytest.raises(ValueError, match="var"):
        metrics.coverage(mean, var, truth)


def test_coverage_rejects_levels_outside_the_open_unit_interval():
    mean, var, truth = _gaussian(m=10)
    with pytest.raises(ValueError, match="level"):
        metrics.coverage(mean, var, truth, levels=(1.0,))


# --- rollout drift ------------------------------------------------------------------------------


def test_linear_error_growth():
    ks = (1, 2, 3, 4, 5, 10, 20)
    d = metrics.rollout_drift({k: 0.1 * k for k in ks})
    assert d["horizons"] == list(ks)
    assert d["slope_m_per_frame"] == pytest.approx(0.1)
    assert d["exponent"] == pytest.approx(1.0)
    assert d["growth"] == {str(k): pytest.approx(float(k)) for k in ks}


def test_quadratic_error_growth_has_exponent_two():
    d = metrics.rollout_drift({k: 0.01 * k**2 for k in (1, 2, 5, 10, 20)})
    assert d["exponent"] == pytest.approx(2.0)


def test_flat_error_has_zero_slope():
    d = metrics.rollout_drift({1: 0.3, 5: 0.3, 20: 0.3})
    assert d["slope_m_per_frame"] == pytest.approx(0.0, abs=1e-12)
    assert d["exponent"] == pytest.approx(0.0, abs=1e-12)


def test_drift_is_undefined_where_it_cannot_be_measured():
    one = metrics.rollout_drift({5: 0.2})
    assert one["slope_m_per_frame"] is None and one["exponent"] is None
    zero = metrics.rollout_drift({1: 0.0, 5: 0.2})
    assert zero["exponent"] is None and zero["growth"] == {"1": None, "5": None}
    assert zero["slope_m_per_frame"] == pytest.approx(0.05)


# --- series and events --------------------------------------------------------------------------


def test_to_series_places_values_and_fills_nan():
    s = metrics.to_series(np.array([1.0, 2.0, 3.0]), ep=[0, 1, 1], t=[3, 3, 5], n=2, T=8)
    assert s.shape == (2, 8)
    assert s[0, 3] == 1.0 and s[1, 3] == 2.0 and s[1, 5] == 3.0
    assert np.isnan(s).sum() == 16 - 3


def test_to_series_rejects_duplicate_or_out_of_range_windows():
    with pytest.raises(ValueError, match="duplicate"):
        metrics.to_series(np.ones(2), ep=[0, 0], t=[3, 3], n=1, T=8)
    with pytest.raises(ValueError, match="range"):
        metrics.to_series(np.ones(1), ep=[0], t=[8], n=1, T=8)


def test_predictive_std_is_the_rms_sigma():
    np.testing.assert_allclose(
        metrics.predictive_std(np.array([[1.0, 9.0], [4.0, 4.0]])), [np.sqrt(5.0), 2.0]
    )
    with pytest.raises(ValueError):
        metrics.predictive_std(np.array([[1.0, -1.0]]))


def _blackout(T=20, start=8, stop=12, n=1):
    flag = np.zeros((n, T), bool)
    flag[:, start:stop] = True
    return flag


def test_uncertainty_that_steps_with_the_flag_rises_then_falls():
    flag = _blackout()
    series = 1.0 + 2.0 * flag
    r = metrics.event_response(series, flag, window=3)
    on, off = r["onset"], r["offset"]
    assert on["n_events"] == 1 and off["n_events"] == 1
    assert on["curve"] == [1.0, 1.0, 1.0, 3.0, 3.0, 3.0]
    assert on["offsets"] == [-3, -2, -1, 0, 1, 2]
    assert on["delta"] == pytest.approx(2.0) and off["delta"] == pytest.approx(-2.0)
    assert on["mean_before"] == 1.0 and on["mean_after"] == 3.0
    assert off["mean_before"] == 3.0 and off["mean_after"] == 1.0


def test_no_events_is_clean():
    flag = np.zeros((3, 20), bool)  # occluded, until M7
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = metrics.event_response(np.ones((3, 20)), flag, window=4)
    for side in ("onset", "offset"):
        assert r[side]["n_events"] == 0
        assert r[side]["delta"] is None
        assert r[side]["mean_before"] is None and r[side]["mean_after"] is None
        assert r[side]["curve"] == [None] * 8
        assert r[side]["n_per_offset"] == [0] * 8
    json.dumps(r)


def test_a_flag_true_at_frame_zero_is_not_an_onset_and_one_to_the_end_has_no_offset():
    flag = np.zeros((2, 20), bool)
    flag[0, :5] = True  # starts hidden, reappears at t=5
    flag[1, 15:] = True  # disappears at t=15, never reappears
    r = metrics.event_response(np.ones((2, 20)), flag, window=3)
    assert r["onset"]["n_events"] == 1
    assert r["offset"]["n_events"] == 1


def test_events_near_the_edges_and_nan_gaps_are_counted_not_averaged_in():
    flag = _blackout(T=20, start=1, stop=4)  # onset at t=1: offsets -3, -2 fall before t=0
    series = np.ones((1, 20))
    series[0, 2] = np.nan  # an unscored window
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        on = metrics.event_response(series, flag, window=3)["onset"]
    assert on["n_per_offset"] == [0, 0, 1, 1, 0, 1]
    assert on["curve"] == [None, None, 1.0, 1.0, None, 1.0]
    assert on["delta"] == pytest.approx(0.0)


def test_delta_pairs_each_event_with_its_own_baseline():
    # Two episodes with different uncertainty levels, each rising by 1 at its onset.
    flag = _blackout(n=2)
    series = np.where(flag, 1.0, 0.0) + np.array([[0.0], [10.0]])
    on = metrics.event_response(series, flag, window=3)["onset"]
    assert on["n_events"] == 2 and on["delta"] == pytest.approx(1.0)


def test_event_response_validates_inputs():
    with pytest.raises(ValueError):
        metrics.event_response(np.ones((2, 20)), np.zeros((2, 19), bool))
    with pytest.raises(ValueError):
        metrics.event_response(np.ones((2, 20)), np.zeros((2, 20), bool), window=0)


def test_event_behaviour_reports_every_flag():
    flag = _blackout()
    out = metrics.event_behaviour(
        1.0 + flag, {"out_of_view": flag, "occluded": np.zeros_like(flag)}, window=2
    )
    assert set(out) == {"out_of_view", "occluded"}
    assert out["out_of_view"]["onset"]["delta"] == pytest.approx(1.0)
    assert out["occluded"]["onset"]["n_events"] == 0


# --- evaluate_positions, split tags, report -----------------------------------------------------


def _positions(m=30, seed=0):
    rng = _rng(seed)
    truth = _by_k(lambda k: rng.normal(size=(m, 2)))
    pers = _by_k(lambda k: truth[k] + rng.normal(size=(m, 2)))
    mean = _by_k(lambda k: truth[k] + 0.1 * rng.normal(size=(m, 2)))
    var = _by_k(lambda k: np.full((m, 2), 0.01))
    return mean, var, truth, pers


def test_evaluate_positions_assembles_every_metric_as_plain_json():
    mean, var, truth, pers = _positions()
    gone = _by_k(lambda k: np.arange(30) % 3 == 0)
    out = metrics.evaluate_positions(mean, truth, pers, var=var, out_of_view=gone)
    assert out["windows"] == 30 and out["horizons"] == list(HORIZONS)
    assert set(out["error_m"]) == set(out["ratio_to_persistence"]) == {"1", "2", "5"}
    assert all(r < 1.0 for r in out["ratio_to_persistence"].values())
    assert set(out["calibration"]["1"]) == {"0.5", "0.9"}
    assert out["drift"]["horizons"] == list(HORIZONS)
    split = out["mean_error_m_by_view_at_t_plus_k"]["5"]
    assert split["n_out_of_view"] == 10
    assert split["in_view"] is not None and split["out_of_view"] is not None
    json.loads(json.dumps(out))


def test_evaluate_positions_without_variance_skips_calibration():
    mean, _, truth, pers = _positions()
    out = metrics.evaluate_positions(mean, truth, pers)
    assert "calibration" not in out and "mean_error_m_by_view_at_t_plus_k" not in out


def test_evaluate_positions_rejects_variance_on_other_windows():
    mean, var, truth, pers = _positions()
    var[5] = var[5][:-1]
    with pytest.raises(ValueError, match="windows"):
        metrics.evaluate_positions(mean, truth, pers, var=var)


def test_only_the_test_split_is_a_headline():
    assert metrics.SplitResult("test", {}).headline
    assert not metrics.SplitResult("dev", {}).headline
    assert not metrics.SplitResult("train", {}).headline
    with pytest.raises(ValueError, match="split"):
        metrics.SplitResult("val", {})


def test_report_never_puts_dev_numbers_under_headline():
    dev = metrics.SplitResult("dev", {"x": 1.0})
    r = metrics.report([dev])
    assert r["headline"] is None
    assert r["development"] == {"dev": {"x": 1.0}}
    r = metrics.report([dev, metrics.SplitResult("test", {"x": 2.0})])
    assert r["headline"] == {"split": "test", "metrics": {"x": 2.0}}
    assert "test" not in r["development"]
    json.loads(json.dumps(r))


def test_report_merges_families_of_one_split_and_rejects_clashes():
    goal = metrics.SplitResult("test", {"goal_head": {"e": 1.0}})
    probe = metrics.SplitResult("test", {"probes": {"r2": 0.9}})
    r = metrics.report([goal, probe, metrics.SplitResult("dev", {"goal_head": {"e": 3.0}})])
    assert r["headline"]["metrics"] == {"goal_head": {"e": 1.0}, "probes": {"r2": 0.9}}
    assert r["development"] == {"dev": {"goal_head": {"e": 3.0}}}
    with pytest.raises(ValueError, match="goal_head"):
        metrics.report([goal, metrics.SplitResult("test", {"goal_head": {"e": 2.0}})])


# --- review fixes: keys, empties, shapes, levels ------------------------------------------------


def test_str_horizon_keys_as_this_module_emits_them_are_accepted():
    truth = {"1": np.zeros((3, 2)), "10": np.zeros((3, 2))}
    pred = {"1": np.ones((3, 2)), "10": 2 * np.ones((3, 2))}
    errs = metrics.errors_by_horizon(pred, truth)
    assert list(errs) == ["1", "10"]
    d = metrics.rollout_drift({k: v["mean"] for k, v in errs.items()})
    assert d["horizons"] == [1, 10]
    out = metrics.evaluate_positions(pred, truth, {1: truth["1"], 10: truth["10"]})
    assert out["ratio_to_persistence"] == {"1": None, "10": None}


def test_a_horizon_given_twice_raises():
    with pytest.raises(ValueError, match="twice"):
        metrics.rollout_drift({1: 0.1, "1": 0.2})


def test_no_windows_is_a_clear_error():
    empty = {1: np.zeros((0, 2))}
    with pytest.raises(ValueError, match="no windows"):
        metrics.errors_by_horizon(empty, empty)


def test_levels_may_be_a_generator_and_every_horizon_gets_calibration():
    mean, var, truth, pers = _positions()
    out = metrics.evaluate_positions(mean, truth, pers, var=var, levels=(p for p in (0.5, 0.9)))
    assert all(set(c) == {"0.5", "0.9"} for c in out["calibration"].values())


def test_levels_that_collide_as_keys_raise():
    mean, var, truth = _gaussian(m=10)
    with pytest.raises(ValueError, match="share the key"):
        metrics.coverage(mean, var, truth, levels=(0.5, np.float64(0.5)))
    with pytest.raises(ValueError, match="share the key"):
        metrics.coverage(mean, var, truth, levels=(0.9, 0.9000000000000001))


def test_a_misshaped_out_of_view_mask_is_a_clear_error():
    mean, _, truth, pers = _positions()
    gone = _by_k(lambda k: np.zeros((30, 1), bool))
    with pytest.raises(ValueError, match="out_of_view"):
        metrics.evaluate_positions(mean, truth, pers, out_of_view=gone)


# --- parity with baselines.evaluate (the numbers are placed side by side) -----------------------


def test_matches_baselines_evaluate_on_the_same_windows():
    rng = _rng(7)
    n, T, dt, horizons, starts = 3, 30, 1 / 15, [1, 2, 5, 10], np.arange(3, 20)  # noqa: N806
    pos = np.cumsum(rng.normal(size=(n, T, 2)), axis=1)
    vel = rng.normal(size=(n, T, 2))
    oov = rng.random((n, T)) < 0.3
    base = baselines.evaluate(pos, vel, dt, starts, horizons, context_frames=4, out_of_view=oov)
    ep, t = (a.ravel() for a in np.meshgrid(np.arange(n), starts, indexing="ij"))
    truth = {k: pos[ep, t + k] for k in horizons}
    pers = {k: baselines.predict("persistence", pos, vel, ep, t, k, dt) for k in horizons}
    model = {k: baselines.predict("const_velocity", pos, vel, ep, t, k, dt) for k in horizons}
    gone = {k: oov[ep, t + k] for k in horizons}
    ours = metrics.evaluate_positions(model, truth, pers, out_of_view=gone)
    # Exact: the same numpy ops on the same arrays, so any drift between the two is a real change.
    assert ours["windows"] == base["windows"]
    assert ours["error_m"] == base["error_m"]["const_velocity"]
    assert ours["ratio_to_persistence"] == base["ratio_to_persistence"]["const_velocity"]
    assert (
        ours["mean_error_m_by_view_at_t_plus_k"]
        == base["mean_error_m_by_view_at_t_plus_k"]["const_velocity"]
    )


# --- hidden vs visible windows (M7's camera and obstacle ablations) -----------------------------


def test_std_ratio_and_coverage_split_by_visibility():
    rng = _rng(3)
    m = 20_000
    hidden = np.arange(m) % 4 == 0
    mean = rng.normal(size=(m, 2))
    var = np.where(hidden[:, None], 1.0, 0.25) * np.ones((m, 2))  # sigma 1.0 hidden, 0.5 visible
    truth = mean + np.sqrt(var) * rng.standard_normal((m, 2))
    v = metrics.by_visibility(mean, var, truth, hidden)
    assert v["has_hidden"] and v["n_hidden"] == 5000 and v["n_visible"] == 15_000
    assert v["std_hidden"] == pytest.approx(1.0) and v["std_visible"] == pytest.approx(0.5)
    assert v["std_ratio"] == pytest.approx(2.0)
    for side in ("hidden", "visible"):
        for p in (0.5, 0.9):
            assert v[f"coverage_{side}"][str(p)]["joint"] == pytest.approx(p, abs=0.025)
    assert v["error_m_hidden"] > v["error_m_visible"]


def test_no_hidden_windows_is_flagged_not_a_crash():
    mean, var, truth = _gaussian(m=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = metrics.by_visibility(mean, var, truth, np.zeros(50, bool))
    assert v["has_hidden"] is False and v["n_hidden"] == 0
    assert v["std_hidden"] is None and v["std_ratio"] is None
    assert v["coverage_hidden"] is None and v["error_m_hidden"] is None
    assert v["coverage_visible"] is not None
    json.dumps(v)


def test_visibility_reports_out_of_view_occluded_and_their_union():
    mean, var, truth = _gaussian(m=40)
    oov = np.arange(40) < 10
    occ = np.zeros(40, bool)  # until M7 adds obstacles
    out = metrics.visibility(mean, var, truth, {"out_of_view": oov, "occluded": occ})
    assert set(out) == {"out_of_view", "occluded", "hidden"}
    assert out["occluded"]["has_hidden"] is False
    assert out["hidden"] == out["out_of_view"]
    occ = occ.copy()
    occ[30:] = True
    both = metrics.visibility(mean, var, truth, {"out_of_view": oov, "occluded": occ})
    assert both["hidden"]["n_hidden"] == 20


def test_visibility_validates_shapes():
    mean, var, truth = _gaussian(m=10)
    with pytest.raises(ValueError, match="hidden"):
        metrics.by_visibility(mean, var, truth, np.zeros((10, 1), bool))
    with pytest.raises(ValueError, match="shape"):
        metrics.visibility(mean, var, truth, {"a": np.zeros(10, bool), "b": np.zeros(9, bool)})
