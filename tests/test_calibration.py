"""Post-hoc calibration of the goal head (jepa_world/calibration.py): one variance scale per k.

Fast tier: synthetic Gaussian residuals with known answers for the pure parts; a tiny model on
fake dev and test splits for the command.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from fake_episodes import SHORT, SMALL_SPLITS, Recorder, flagged_episode
from tiny import AUX, BASE, TINY

from jepa_world import calibration, metrics, regimes, train, viewer
from jepa_world import eval as eval_cli
from jepa_world.data import build
from jepa_world.data.splits import LeakageError
from jepa_world.models.goal import goal_nll
from jepa_world.utils.config import load_config

EPISODES = ["data.source=episodes", *SHORT, "sim.frame_size=16"]  # T=30, 16 px for TINY's L1
GOAL_KS = [1, 3, 5]  # AUX's model.goal_horizons
EVAL_SETS = [
    *SMALL_SPLITS,
    "eval.device=cpu",
    "eval.batch_size=5",
    "eval.sim_params=[]",
    "eval.probes.enabled=false",
]


def run_cfg(*extra: str) -> dict:
    return load_config(BASE, [*TINY, *AUX, *EPISODES, *extra])


def eval_cfg(*extra: str) -> dict:
    return load_config(BASE, [*EVAL_SETS, *extra])


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JEPA_ARTIFACTS", str(root))
        for split in ("dev", "test"):
            build.build_split(run_cfg(), split, simulate=Recorder(flagged_episode))
    return root


@pytest.fixture
def artifacts(root, monkeypatch):
    monkeypatch.setenv("JEPA_ARTIFACTS", str(root))
    return root


def make_run(cfg: dict, name="tiny-calibration"):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    opt = train.build_optimizer(model, cfg)
    run = train.start_run(cfg, name, torch.device("cpu"))
    train.save_checkpoint(run / "checkpoints" / "step_0000003.pt", model, opt, 3, cfg)
    return run, model.eval()


def gaussian_windows(m=4000, true_scale=2.0, seed=0):
    """A model whose predicted variance is 1/true_scale of the real one, on every axis."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=(m, 2))
    var = np.exp(rng.normal(size=(m, 2)) * 0.3)
    truth = mean + rng.normal(size=(m, 2)) * np.sqrt(var * true_scale)
    return mean, var, truth


# --- pure pieces --------------------------------------------------------------------------------


def test_r2_is_the_standardised_squared_residual_metrics_coverage_counts():
    mean, var, truth = gaussian_windows(m=200)
    r2 = calibration.standardized_r2(mean, var, truth)
    assert r2.shape == (200,)
    np.testing.assert_allclose(r2, ((truth - mean) ** 2 / var).sum(axis=1))
    for p in (0.5, 0.9):
        inside = (r2 <= calibration.chi2_2(p)).mean()
        assert inside == metrics.coverage(mean, var, truth, [p])[str(p)]["joint"]


@pytest.mark.parametrize("level", [0.5, 0.9])
def test_the_fitted_scale_makes_the_region_hold_its_level_on_the_fit_set(level):
    mean, var, truth = gaussian_windows()
    s = calibration.fit_scale(calibration.standardized_r2(mean, var, truth), level)
    cov = metrics.coverage(mean, var * s, truth, [level])[str(level)]["joint"]
    assert abs(cov - level) <= 1.0 / len(mean) + 1e-12
    assert s == pytest.approx(2.0, rel=0.1)  # the variance was 2x too small


def test_fit_scale_refuses_a_fit_set_with_no_spread():
    with pytest.raises(ValueError, match="no positive variance scale"):
        calibration.fit_scale(np.r_[np.zeros(95), np.ones(5)], 0.9)


def test_the_fitted_level_is_always_scored():
    dev = {1: gaussian_windows(m=200, seed=1)}
    test = {1: gaussian_windows(m=200, seed=2)}
    folds = np.arange(200) % 2
    labels = {1: np.zeros(200, int)}
    out = calibration.calibrate(dev, test, folds, labels, levels=[0.5], level=0.8)
    assert set(out["1"]["test"]["dev_scaled"]["coverage"]) == {"0.5", "0.8"}
    assert abs(out["1"]["dev_fit"]["dev_after"]["coverage"]["0.8"] - 0.8) <= 1 / 200 + 1e-12


def test_fit_scale_is_one_for_a_calibrated_model_and_scales_with_the_residuals():
    p = (np.arange(1, 10_000) - 0.5) / 10_000
    r2 = -2 * np.log(1 - p)  # exact chi-square(2) quantiles: a calibrated model
    assert calibration.fit_scale(r2, 0.9) == pytest.approx(1.0, rel=1e-3)
    assert calibration.fit_scale(3 * r2, 0.9) == pytest.approx(3 * calibration.fit_scale(r2, 0.9))
    with pytest.raises(ValueError, match="level"):
        calibration.fit_scale(r2, 1.0)


def test_nll_is_the_training_goal_nll():
    mean, var, truth = gaussian_windows(m=50)
    ours = calibration.nll(mean, var, truth)
    assert ours.shape == (50,)
    t = lambda a: torch.from_numpy(a)[:, None, :]  # noqa: E731 (B, K=1, 2)
    ref = goal_nll(t(mean), t(np.log(var)), t(truth))
    assert ours.mean() == pytest.approx(ref.item(), rel=1e-12)


def test_the_ml_scale_minimises_the_mean_nll():
    mean, var, truth = gaussian_windows()
    s = calibration.ml_scale(calibration.standardized_r2(mean, var, truth))
    best = calibration.nll(mean, var * s, truth).mean()
    for f in (0.8, 0.95, 1.05, 1.25):
        assert calibration.nll(mean, var * s * f, truth).mean() > best


def test_folds_split_terrains_never_windows_and_are_balanced():
    seeds = np.repeat([7, 3, 9, 1, 5, 11], 5)  # 6 terrains, 5 windows each
    folds = calibration.terrain_folds(seeds, 3)
    assert folds.shape == seeds.shape
    for s in np.unique(seeds):
        assert len(set(folds[seeds == s])) == 1  # a terrain is in exactly one fold
    assert sorted(np.bincount(folds)) == [10, 10, 10]
    np.testing.assert_array_equal(folds, calibration.terrain_folds(seeds[::-1], 3)[::-1])
    with pytest.raises(ValueError, match="terrain"):
        calibration.terrain_folds(seeds, 7)  # more folds than terrains


def test_cross_fit_scales_come_only_from_the_other_folds():
    mean, var, truth = gaussian_windows(m=300)
    r2 = calibration.standardized_r2(mean, var, truth)
    folds = np.arange(300) % 3
    s = calibration.cross_fit_scales(r2, folds, 0.9)
    for f in range(3):
        assert np.all(s[folds == f] == calibration.fit_scale(r2[folds != f], 0.9))
    r2b = r2.copy()
    r2b[folds == 0] *= 100  # fold 0's own residuals must not move fold 0's scale
    s2 = calibration.cross_fit_scales(r2b, folds, 0.9)
    np.testing.assert_array_equal(s2[folds == 0], s[folds == 0])


def test_calibrate_reports_before_and_after_per_k_and_per_regime():
    rng = np.random.default_rng(1)
    dev = {k: gaussian_windows(m=500, true_scale=2.0, seed=k) for k in (1, 5)}
    test = {k: gaussian_windows(m=600, true_scale=2.0, seed=10 + k) for k in (1, 5)}
    seeds = np.repeat(np.arange(6), 100)
    labels = {k: rng.integers(0, 3, 600) for k in (1, 5)}
    folds = calibration.terrain_folds(seeds, 3)
    out = calibration.calibrate(dev, test, folds, labels, levels=[0.5, 0.9], level=0.9)
    assert set(out) == {"1", "5"}
    for k in (1, 5):
        r = out[str(k)]
        m, v, y = test[k]
        s = calibration.fit_scale(calibration.standardized_r2(*dev[k]), 0.9)
        assert r["dev_fit"]["scale"] == s and r["dev_fit"]["std_factor"] == pytest.approx(
            math.sqrt(s)
        )
        before = metrics.coverage(m, v, y, [0.5, 0.9])
        after = metrics.coverage(m, v * s, y, [0.5, 0.9])
        assert r["test"]["before"]["coverage"] == {p: before[p]["joint"] for p in ("0.5", "0.9")}
        assert r["test"]["dev_scaled"]["coverage"] == {p: after[p]["joint"] for p in ("0.5", "0.9")}
        assert r["test"]["before"]["nll"] == pytest.approx(calibration.nll(m, v, y).mean())
        assert r["test"]["dev_scaled"]["nll"] == pytest.approx(calibration.nll(m, v * s, y).mean())
        cf = r["test"]["cross_fit"]
        assert len(cf["fold_scales"]) == 3 and cf["n_folds"] == 3
        assert r["test"]["dev_scaled"]["coverage"]["0.9"] == pytest.approx(0.9, abs=0.05)
        for i, name in enumerate(regimes.REGIMES):
            mask = labels[k] == i
            row = r["by_regime"][name]
            assert row["windows"] == int(mask.sum())
            cov = metrics.coverage(m[mask], v[mask] * s, y[mask], [0.9])["0.9"]["joint"]
            assert row["dev_scaled"]["coverage"]["0.9"] == cov
    json.dumps(out, allow_nan=False)


def test_the_table_prints_each_regimes_coverage_at_the_fitted_level():
    dev = {1: gaussian_windows(m=300, seed=1)}
    test = {1: gaussian_windows(m=300, seed=2)}
    labels = {1: np.arange(300) % 3}
    folds = calibration.terrain_folds(np.repeat(np.arange(4), 75), 2)
    out = calibration.calibrate(dev, test, folds, labels, levels=[0.5, 0.9], level=0.9)
    text = calibration.format_table({"meta": {"level": 0.9}, "horizons": out})
    (line,) = [s for s in text.splitlines() if "by regime" in s]
    assert line.startswith("  cov0.9 by regime") and "n/a" not in line
    for name in regimes.REGIMES:
        row = out["1"]["by_regime"][name]
        before, after = row["before"]["coverage"]["0.9"], row["dev_scaled"]["coverage"]["0.9"]
        assert f"{name} {before:.3f} -> {after:.3f}" in line


# --- the command, end to end --------------------------------------------------------------------


def test_calibrate_run_fits_on_dev_scores_test_and_writes_outside_the_eval_glob(artifacts):
    cfg = run_cfg()
    run, _ = make_run(cfg)
    report, _ = eval_cli.evaluate_run(run, eval_cfg(), ["test", "dev"])
    doc, path = calibration.calibrate_run(run, eval_cfg(), n_folds=2)
    assert path.parent == run / "eval" / "calibration" and json.loads(path.read_text()) == doc
    assert [p.name for p in (run / "eval").glob("*.json")] != [] and all(
        "report" in json.loads(p.read_text()) for p in (run / "eval").glob("*.json")
    )
    assert viewer.eval_summary(run, "step_0000003.pt", "test") is not None
    meta = doc["meta"]
    assert meta["fit_split"] == "dev" and meta["scored_split"] == "test"
    assert meta["windows"] == {"dev": 28, "test": 28} and meta["level"] == 0.9
    assert set(doc["horizons"]) == {str(k) for k in GOAL_KS}
    head = report["report"]["headline"]["metrics"]["goal_head"]["calibration"]
    dev = report["report"]["development"]["dev"]["goal_head"]["calibration"]
    for k in map(str, GOAL_KS):
        r = doc["horizons"][k]
        # "before" is exactly eval --run's coverage, on test and on dev
        assert r["test"]["before"]["coverage"] == {p: head[k][p]["joint"] for p in ("0.5", "0.9")}
        assert r["dev_fit"]["dev_before"]["coverage"] == {
            p: dev[k][p]["joint"] for p in ("0.5", "0.9")
        }
        # on dev itself, the fitted scale makes the 90% region hold 90% (to one window)
        assert abs(r["dev_fit"]["dev_after"]["coverage"]["0.9"] - 0.9) <= 1 / 28 + 1e-12
        assert set(r["by_regime"]) == set(regimes.REGIMES)
        assert sum(r["by_regime"][n]["windows"] for n in regimes.REGIMES) == 28


def test_calibrate_run_needs_held_out_dev_and_test_and_a_goal_head(artifacts):
    dev_run, _ = make_run(run_cfg("data.split=dev"), name="tiny-cal-dev-trained")
    with pytest.raises(LeakageError, match="trained on"):
        calibration.calibrate_run(dev_run, eval_cfg(), n_folds=2)
    near, _ = make_run(load_config(BASE, [*TINY, *EPISODES]), name="tiny-cal-no-goal")
    with pytest.raises(ValueError, match="goal head"):
        calibration.calibrate_run(near, eval_cfg(), n_folds=2)


@pytest.mark.parametrize(("level", "folds"), [(90, 2), (1.0, 2), (0.9, 1), (0.9, 3)])
def test_calibrate_run_checks_level_and_folds_before_loading_anything(
    artifacts, monkeypatch, level, folds
):
    loads = []
    monkeypatch.setattr(calibration, "load_run", lambda *a, **kw: loads.append(a))
    with pytest.raises(ValueError, match="level|n_folds"):
        calibration.calibrate_run(
            "unused", eval_cfg(), level=level, n_folds=folds
        )  # 2 test terrains
    assert loads == []


def test_cli_prints_before_and_after_per_k(artifacts, capsys):
    run, _ = make_run(run_cfg(), name="tiny-cal-cli")
    args = ["--run", str(run), "--folds", "2"]
    for s in EVAL_SETS:
        args += ["--set", s]
    assert calibration.main(args) == 0
    out = capsys.readouterr().out
    (path,) = (run / "eval" / "calibration").glob("*.json")
    assert str(path) in out
    for word in ("k=1", "k=5", "before", "dev-fit", "cross-fit", "nll"):
        assert word in out
