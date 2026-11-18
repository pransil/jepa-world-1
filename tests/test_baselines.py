"""Position baselines (SPEC §7.1): persistence and constant velocity, and the eval CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fake_episodes import SHORT, G, Recorder
from tiny import BASE

from jepa_world import baselines
from jepa_world import eval as eval_cli
from jepa_world.data import build
from jepa_world.data.dataset import valid_starts
from jepa_world.utils import paths
from jepa_world.utils.config import load_config

FPS, DT, T, C = 15, 1 / 15, 40, 4
HORIZONS = [1, 2, 5, 10, 20]


def motion(kind: str, n: int = 3, seed: int = 0):
    """(n, T, 2) positions and velocities, exact in closed form."""
    rng = np.random.default_rng(seed)
    p0, v0 = rng.uniform(-5, 5, (n, 1, 2)), rng.uniform(-3, 3, (n, 1, 2))
    t = (np.arange(T) * DT)[None, :, None]
    pos, vel = p0 + v0 * t, np.broadcast_to(v0, (n, T, 2)).copy()
    if kind == "ballistic":
        pos[..., 1] -= 0.5 * G * t[..., 0] ** 2
        vel[..., 1] -= G * t[..., 0]
    return pos, vel


def windows(n: int, k_max: int):
    starts = valid_starts(T, C, k_max)
    ep, t = np.meshgrid(np.arange(n), starts, indexing="ij")
    return ep.ravel(), t.ravel()


def err(name, pos, vel, k):
    ep, t = windows(len(pos), max(HORIZONS))
    pred = baselines.predict(name, pos, vel, ep, t, k, DT)
    return np.linalg.norm(pred - pos[ep, t + k], axis=-1)


@pytest.mark.parametrize("kind", ["uniform", "ballistic"])
def test_persistence_error_is_the_displacement(kind):
    pos, vel = motion(kind)
    ep, t = windows(len(pos), max(HORIZONS))
    for k in HORIZONS:
        want = np.linalg.norm(pos[ep, t + k] - pos[ep, t], axis=-1)
        np.testing.assert_array_equal(err("persistence", pos, vel, k), want)


def test_constant_velocity_is_exact_on_uniform_motion():
    pos, vel = motion("uniform")
    for name in ("const_velocity", "const_velocity_fd"):
        for k in HORIZONS:
            np.testing.assert_allclose(err(name, pos, vel, k), 0.0, atol=1e-12)


def test_constant_velocity_errors_on_ballistic_motion_are_exact():
    """True v(t): error 1/2 g (k dt)^2. Finite difference x(t) - x(t-1) = v(t) dt + 1/2 g dt^2,
    so its error is 1/2 g dt^2 k (k + 1)."""
    pos, vel = motion("ballistic")
    for k in HORIZONS:
        np.testing.assert_allclose(err("const_velocity", pos, vel, k), 0.5 * G * (k * DT) ** 2)
        np.testing.assert_allclose(
            err("const_velocity_fd", pos, vel, k), 0.5 * G * DT**2 * k * (k + 1)
        )


def test_baselines_see_nothing_after_t():
    pos, vel = motion("ballistic")
    ep, t = windows(len(pos), max(HORIZONS))
    noisy_pos, noisy_vel = pos.copy(), vel.copy()
    for e, s in zip(ep, t, strict=True):
        noisy_pos[e, s + 1 :] += 100.0
        noisy_vel[e, s + 1 :] += 100.0
        for name in baselines.BASELINES:
            for k in HORIZONS:
                a = baselines.predict(name, pos, vel, np.array([e]), np.array([s]), k, DT)
                b = baselines.predict(
                    name, noisy_pos, noisy_vel, np.array([e]), np.array([s]), k, DT
                )
                np.testing.assert_array_equal(a, b, err_msg=f"{name} k={k}")
        noisy_pos[e], noisy_vel[e] = pos[e], vel[e]


def test_evaluate_reports_every_baseline_and_horizon():
    pos, vel = motion("ballistic", n=4)
    starts = valid_starts(T, C, max(HORIZONS))
    out = baselines.evaluate(pos, vel, DT, starts, HORIZONS, C)
    assert out["windows"] == 4 * len(starts)
    assert out["horizons"] == HORIZONS
    for name in baselines.BASELINES:
        for k in HORIZONS:
            s = out["error_m"][name][str(k)]
            assert np.isfinite([s["mean"], s["median"], s["p90"]]).all()
            assert 0 <= s["median"] <= s["p90"]
        assert out["ratio_to_persistence"]["persistence"] == {str(k): 1.0 for k in HORIZONS}
        for k in HORIZONS:
            ratio = out["ratio_to_persistence"][name][str(k)]
            means = (
                out["error_m"][name][str(k)]["mean"],
                out["error_m"]["persistence"][str(k)]["mean"],
            )
            assert ratio == pytest.approx(means[0] / means[1])


def test_evaluate_ratio_is_none_when_persistence_is_exact():
    pos = np.zeros((2, T, 2))
    out = baselines.evaluate(pos, pos.copy(), DT, valid_starts(T, C, 5), [1, 5], C)
    assert out["ratio_to_persistence"]["const_velocity"] == {"1": None, "5": None}


def test_one_context_frame_drops_only_the_finite_difference_baseline():
    """With C = 1, x(t-1) is outside the context, so const_velocity_fd would see more than the
    model does. It is left out, and the others are still scored from t = 0."""
    pos, vel = motion("uniform")
    out = baselines.evaluate(pos, vel, DT, valid_starts(T, 1, 5), [1, 5], context_frames=1)
    assert out["starts"][0] == 0
    assert set(out["error_m"]) == {"persistence", "const_velocity"}
    with pytest.raises(ValueError, match="C=4"):
        baselines.evaluate(pos, vel, DT, valid_starts(T, 1, 5), [1, 5], context_frames=4)


def test_errors_split_by_whether_the_ball_is_out_of_view_at_t_plus_k():
    pos, vel = motion("ballistic", n=2)
    oov = np.zeros((2, T), bool)
    oov[1, 25:] = True  # episode 1 leaves the frame at frame 25 for good
    starts = valid_starts(T, C, 10)
    out = baselines.evaluate(pos, vel, DT, starts, [1, 10], C, out_of_view=oov)
    view = out["mean_error_m_by_view_at_t_plus_k"]
    for name in out["error_m"]:
        for k in (1, 10):
            v = view[name][str(k)]
            assert v["n_out_of_view"] == int(oov[1, starts + k].sum())
            ep, t = np.meshgrid([0, 1], starts, indexing="ij")
            e = np.linalg.norm(
                baselines.predict(name, pos, vel, ep.ravel(), t.ravel(), k, DT)
                - pos[ep.ravel(), t.ravel() + k],
                axis=-1,
            )
            gone = oov[ep.ravel(), t.ravel() + k]
            assert v["out_of_view"] == pytest.approx(e[gone].mean())
            assert v["in_view"] == pytest.approx(e[~gone].mean())


def test_eval_baselines_cli_writes_a_run_dir():
    overrides = [*SHORT, "model.horizons=[1, 2, 3, 4, 5, 10, 20]"]
    cfg = load_config(BASE, overrides)
    build.build_split(cfg, "test", simulate=Recorder())
    args = ["--baselines", "--config", str(BASE), "--split", "test"]
    for o in overrides:
        args += ["--set", o]
    assert eval_cli.main(args) == 0
    (run,) = (paths.artifacts_root() / "runs").iterdir()
    assert "baselines-test" in run.name
    out = json.loads((run / "baselines.json").read_text())
    assert out["split"] == "test" and out["episodes"] == 4
    assert set(out["error_m"]) == set(baselines.BASELINES)
    assert set(out["error_m"]["persistence"]) == {"1", "2", "3", "4", "5", "10", "20"}
    assert set(out["mean_error_m_by_view_at_t_plus_k"]) == set(baselines.BASELINES)
    meta = json.loads((run / "meta.json").read_text())
    assert "git" in meta and meta["dataset"].endswith(
        paths.dataset_dir(build.split_key(cfg, "test"), "episodes-test").name
    )
    assert (run / "config.yaml").exists()


def test_eval_without_a_mode_is_an_error():
    with pytest.raises(SystemExit):
        eval_cli.main([])
