"""Linear probes on frozen embeddings (SPEC §7.2), on synthetic features with known answers."""

from __future__ import annotations

import json

import numpy as np
import pytest
from jepa_world.metrics import SplitResult

from jepa_world import probes


def _hidden(m, rng):
    """Targets shaped like the real ones: pos (M,2) in metres, scalars for the hidden params."""
    return {
        "pos": rng.normal(0.0, 2.0, size=(m, 2)),
        "friction": rng.uniform(0.2, 1.0, size=m),
        "restitution": rng.uniform(0.3, 0.9, size=m),
        "radius": rng.uniform(0.1, 0.3, size=m),
    }


def _stack(targets):
    return np.column_stack([np.asarray(v).reshape(len(v), -1) for v in targets.values()])


def _linear_data(split, m, seed, first_terrain, mix, noise=1e-3):
    rng = np.random.default_rng(seed)
    targets = _hidden(m, rng)
    y = _stack(targets)
    feats = {lvl: y @ a + noise * rng.standard_normal((m, a.shape[1])) for lvl, a in mix.items()}
    return probes.ProbeData(
        split=split, features=feats, targets=targets, terrain_seed=first_terrain + np.arange(m) // 4
    )


def _mix(seed=0, dims=(16, 24, 32)):
    rng = np.random.default_rng(seed)
    return {lvl: rng.normal(size=(5, d)) for lvl, d in zip(probes.LEVELS, dims, strict=True)}


def test_pool_averages_space():
    f = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)
    p = probes.pool(f)
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p, f.mean(axis=(2, 3)))
    np.testing.assert_array_equal(probes.pool(p), p)
    with pytest.raises(ValueError):
        probes.pool(np.zeros(3))


def test_probe_data_validates_rows_and_feature_rank():
    rng = np.random.default_rng(0)
    ok = {"split": "train", "targets": {"radius": rng.normal(size=5)}, "terrain_seed": np.arange(5)}
    with pytest.raises(ValueError, match="rows"):
        probes.ProbeData(features={"L1": rng.normal(size=(4, 3))}, **ok)
    with pytest.raises(ValueError, match="pool"):
        probes.ProbeData(features={"L1": rng.normal(size=(5, 3, 2, 2))}, **ok)
    with pytest.raises(ValueError, match="split"):
        probes.ProbeData(**{**ok, "split": "val"}, features={"L1": rng.normal(size=(5, 3))})


def test_linearly_embedded_targets_are_recovered_on_held_out_terrains():
    mix = _mix()
    train = _linear_data("train", 300, seed=1, first_terrain=1000, mix=mix)
    held = _linear_data("test", 120, seed=2, first_terrain=100000, mix=mix)
    res = probes.run_probes(train, held)
    assert isinstance(res, SplitResult) and res.split == "test" and res.headline
    by_level = res.metrics["by_level"]
    assert set(by_level) == {"L1", "L2", "L3", "pooled"}
    for level, per_target in by_level.items():
        assert set(per_target) == {"pos", "friction", "restitution", "radius"}
        for name, s in per_target.items():
            assert s["r2"] > 0.99, (level, name, s)
        assert len(per_target["pos"]["r2_per_dim"]) == 2
        assert len(per_target["radius"]["rmse_per_dim"]) == 1
    assert res.metrics["train_split"] == "train"
    assert res.metrics["n_train"] == 300 and res.metrics["n_held_out"] == 120
    json.loads(json.dumps(res.metrics))


def _random_data(split, m, d, seed, first_terrain):
    rng = np.random.default_rng(seed)
    return probes.ProbeData(
        split=split,
        features={"L1": rng.normal(size=(m, d))},
        targets={"friction": rng.normal(size=m)},
        terrain_seed=first_terrain + np.arange(m),
    )


def test_features_unrelated_to_the_target_score_near_zero():
    res = probes.run_probes(
        _random_data("train", 400, 8, seed=3, first_terrain=0),
        _random_data("dev", 200, 8, seed=4, first_terrain=10_000),
    )
    assert res.metrics["by_level"]["L1"]["friction"]["r2"] < 0.1
    assert not res.headline


def test_the_probe_is_fit_on_train_rows_only():
    # 40 features, 50 held-out rows: a fit on the held-out rows themselves would score R^2 ~ 0.8.
    train = _random_data("train", 400, 40, seed=5, first_terrain=0)
    held = _random_data("test", 50, 40, seed=6, first_terrain=10_000)
    x, y = held.features["L1"], held.targets["friction"]
    in_sample = 1 - np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2) / np.sum(
        (y - y.mean()) ** 2
    )
    assert in_sample > 0.5  # the trap is real
    assert probes.run_probes(train, held).metrics["by_level"]["L1"]["friction"]["r2"] < 0.1


def test_scoring_does_not_change_the_fitted_probe():
    mix = _mix()
    train = _linear_data("train", 200, seed=7, first_terrain=1000, mix=mix)
    fitted = probes.fit_probes(train)
    before = fitted.predict("L2", "pos", train.features["L2"]).copy()
    probes.score_probes(fitted, _linear_data("test", 80, seed=8, first_terrain=100000, mix=mix))
    np.testing.assert_array_equal(fitted.predict("L2", "pos", train.features["L2"]), before)


def test_never_fit_on_test():
    held = _random_data("test", 50, 4, seed=9, first_terrain=100000)
    with pytest.raises(ValueError, match="test"):
        probes.fit_probes(held)


def test_held_out_must_not_share_a_terrain_or_the_split_with_train():
    train = _random_data("train", 100, 4, seed=10, first_terrain=0)
    fitted = probes.fit_probes(train)
    with pytest.raises(ValueError, match="terrain"):
        probes.score_probes(fitted, _random_data("test", 50, 4, seed=11, first_terrain=90))
    with pytest.raises(ValueError, match="split"):
        probes.score_probes(fitted, _random_data("train", 50, 4, seed=12, first_terrain=5000))


def test_pooled_sees_what_the_levels_hold_jointly():
    # friction = u + v, with u only in L1 and v only in L2: each level alone explains about half.
    def data(split, m, seed, first):
        rng = np.random.default_rng(seed)
        u, v = rng.normal(size=m), rng.normal(size=m)
        noise = lambda: 0.05 * rng.normal(size=(m, 3))  # noqa: E731
        return probes.ProbeData(
            split=split,
            features={
                "L1": np.column_stack([u, noise()]),
                "L2": np.column_stack([v, noise()]),
                "L3": noise(),
            },
            targets={"friction": u + v},
            terrain_seed=first + np.arange(m),
        )

    r2 = {
        lvl: s["friction"]["r2"]
        for lvl, s in probes.run_probes(data("train", 300, 13, 0), data("test", 150, 14, 9000))
        .metrics["by_level"]
        .items()
    }
    assert r2["pooled"] > 0.99
    assert r2["L1"] < 0.8 and r2["L2"] < 0.8 and r2["L3"] < 0.1


def test_held_out_needs_the_trained_levels_and_targets():
    fitted = probes.fit_probes(_random_data("train", 60, 4, seed=15, first_terrain=0))
    bad = probes.ProbeData(
        split="test",
        features={"L2": np.zeros((5, 4))},
        targets={"friction": np.zeros(5)},
        terrain_seed=np.arange(5) + 10_000,
    )
    with pytest.raises(ValueError, match="L1"):
        probes.score_probes(fitted, bad)
