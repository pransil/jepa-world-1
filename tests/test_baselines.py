"""Position baselines (SPEC §7.1): persistence, constant velocity, the simulator, and the eval CLI."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from fake_episodes import SHORT, G, Recorder, physics_episode
from tiny import BASE

from jepa_world import baselines
from jepa_world import eval as eval_cli
from jepa_world.data import build
from jepa_world.data.dataset import valid_starts
from jepa_world.sim.ball import mean_ball
from jepa_world.sim.episode import simulate_from_state
from jepa_world.sim.terrain import Terrain
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
    # --sim-params with no values: the closed-form baselines only (the simulator has its own tests)
    args = ["--baselines", "--config", str(BASE), "--split", "test", "--sim-params"]
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


# --- simulator-aware baseline (SPEC §7.1) -------------------------------------------------------
# Re-simulates the known terrain from the true state at t (sim.episode.simulate_from_state).
# Params: docs/decisions/2026-11-17-simulator-baseline-params.md. Policy: jepa_world/DESIGN.md.

SIM_SEEDS = (100001, 1000)  # SHORT config: an impact start at t = 6 (true gap -1.8 cm)
SIM_STARTS = np.arange(3, 10)  # every valid t for T = 30, C = 4, k_max = 20


@pytest.fixture(scope="module")
def short_sim_cfg():
    return load_config(BASE, SHORT)["sim"]


@pytest.fixture(scope="module")
def real_episode(short_sim_cfg):
    return physics_episode(short_sim_cfg, *SIM_SEEDS)


def flat(height=0.0):
    x = np.linspace(-5.0, 5.0, 101)
    return Terrain(x=x, heights=np.full_like(x, height))


def test_sim_ball_modes(short_sim_cfg, real_episode):
    true, mean = real_episode.ball, mean_ball(short_sim_cfg)
    assert baselines.sim_ball("true", true, short_sim_cfg) is true
    assert baselines.sim_ball("mean", true, short_sim_cfg) == mean
    tr = baselines.sim_ball("true_radius", true, short_sim_cfg)
    assert (tr.radius, tr.friction, tr.restitution) == (
        true.radius,
        mean.friction,
        mean.restitution,
    )
    with pytest.raises(ValueError, match="oracle"):
        baselines.sim_ball("oracle", true, short_sim_cfg)
    assert set(baselines.DEFAULT_SIM_PARAMS) <= set(baselines.SIM_PARAMS)


def test_match_clearance_is_exactly_zero_for_the_true_radius():
    pos = np.array([[0.0, 0.15], [1.0, 0.149], [2.0, 3.0]])
    dz, unmatched = baselines.match_clearance(flat(), pos, 0.15, 0.15)
    assert (dz == 0.0).all() and not unmatched.any()


@pytest.mark.parametrize("r_true, r", [(0.15, 0.25), (0.35, 0.25)])
def test_match_clearance_on_flat_ground_is_the_radius_difference(r_true, r):
    pos = np.array([[0.0, r_true], [1.0, r_true - 0.01], [2.0, 2.0]])  # touching, pressed, flying
    dz, unmatched = baselines.match_clearance(flat(), pos, r_true, r)
    np.testing.assert_allclose(dz, r - r_true, atol=1e-12)
    assert not unmatched.any()


@pytest.mark.parametrize("r_true, r", [(0.15, 0.25), (0.35, 0.15)])
def test_match_clearance_keeps_the_true_gap_on_a_slope(r_true, r):
    x = np.linspace(-5.0, 5.0, 201)
    slope = Terrain(x=x, heights=0.6 * x)
    xs = np.array([-1.0, 0.0, 1.5])
    pos = np.stack(
        [xs, 0.6 * xs + r_true / np.cos(np.arctan(0.6)) + np.array([-0.01, 0.0, 0.3])], 1
    )
    dz, unmatched = baselines.match_clearance(slope, pos, r_true, r)
    shifted = pos + np.stack([np.zeros_like(dz), dz], 1)
    np.testing.assert_allclose(slope.gap(shifted, r), slope.gap(pos, r_true), atol=1e-9)
    assert not unmatched.any()


def test_match_clearance_searches_exactly_max_factor_radius_differences():
    # On a slope s, a touching ball needs dz = Δr·sqrt(1 + s²). With s = 5.92 that is 6.0·Δr:
    # beyond a range of 5·Δr, within one of 8·Δr.
    x = np.linspace(-5.0, 5.0, 201)
    s = np.sqrt(35.0)
    steep = Terrain(x=x, heights=s * x)
    pos = np.array([[0.2, s * 0.2 + 0.15 * np.sqrt(1 + s * s)]])  # touching with r_true = 0.15
    dz8, un8 = baselines.match_clearance(steep, pos, 0.15, 0.25, max_factor=8.0)
    assert not un8.any() and dz8[0] == pytest.approx(0.6, abs=1e-9)
    dz5, un5 = baselines.match_clearance(steep, pos, 0.15, 0.25, max_factor=5.0)
    assert un5.all() and dz5[0] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="max_factor"):
        baselines.match_clearance(steep, pos, 0.15, 0.25, max_factor=0.5)


def test_a_diverged_rollout_raises_instead_of_scoring_nan(short_sim_cfg, real_episode):
    ep = real_episode

    def blown(sim_cfg, terrain, ball, pos, vel, spin, frames, **kw):
        out = _still(sim_cfg, terrain, ball, pos, vel, spin, frames)
        out.ball_pos[-1] = np.nan
        return out

    with pytest.raises(FloatingPointError, match="t=4"):
        baselines.simulator_predict(
            short_sim_cfg, ep.terrain, ep.ball, ep.ball.radius,
            ep.ball_pos, ep.ball_vel, ep.ball_spin, np.array([4]), [1, 5], simulate=blown,
        )  # fmt: skip
    pos, vel = motion("uniform", n=1)
    starts = valid_starts(T, C, 5)
    bad = {1: np.full((len(starts), 2), np.nan), 5: np.zeros((len(starts), 2))}
    with pytest.raises(ValueError, match="non-finite"):
        baselines.evaluate(pos, vel, DT, starts, [1, 5], C, extra={"bad": bad})


def test_match_clearance_falls_back_beside_a_wall():
    # A 3 m step at x = 0. The ball rests against its face, well above the floor, so no vertical
    # shift short of climbing over the step changes its clearance.
    x = np.linspace(-5.0, 5.0, 1001)
    step = Terrain(x=x, heights=np.where(x < 0.0, 0.0, 3.0))
    dz, unmatched = baselines.match_clearance(step, np.array([[-0.16, 1.0]]), 0.15, 0.25)
    assert unmatched.tolist() == [True]
    np.testing.assert_allclose(dz, [0.1])


def test_simulator_with_true_params_reproduces_the_episode(short_sim_cfg, real_episode):
    ep = real_episode
    assert (ep.terrain.gap(ep.ball_pos[SIM_STARTS], ep.ball.radius) < 0).any(), "no contact start"
    preds, info = baselines.simulator_predict(
        short_sim_cfg, ep.terrain, ep.ball, ep.ball.radius,
        ep.ball_pos, ep.ball_vel, ep.ball_spin, SIM_STARTS, HORIZONS, cam_pos=ep.cam_pos,
    )  # fmt: skip
    assert set(preds) == set(HORIZONS) and (info["dz"] == 0).all()
    e = np.concatenate(
        [np.linalg.norm(preds[k] - ep.ball_pos[SIM_STARTS + k], axis=-1) for k in HORIZONS]
    )
    assert e.max() <= 0.01 and np.median(e) <= 1e-6  # sim/DESIGN.md's round-trip bounds


def test_simulator_with_another_radius_starts_at_the_true_clearance_and_slip(
    short_sim_cfg, real_episode
):
    ep, starts = real_episode, np.array([5, 6, 7])  # 6 is the impact
    mean = mean_ball(short_sim_cfg)
    calls = []

    def recording(sim_cfg, terrain, ball, pos, vel, spin, frames, **kw):
        roll = simulate_from_state(sim_cfg, terrain, ball, pos, vel, spin, frames, **kw)
        calls.append((np.array(pos), float(spin), roll))
        return roll

    preds, info = baselines.simulator_predict(
        short_sim_cfg, ep.terrain, mean, ep.ball.radius, ep.ball_pos, ep.ball_vel, ep.ball_spin,
        starts, [1, 5], simulate=recording,
    )  # fmt: skip
    assert len(calls) == len(starts)
    for i, (t, (pos, spin, roll)) in enumerate(zip(starts, calls, strict=True)):
        true_gap = ep.terrain.gap(ep.ball_pos[t], ep.ball.radius)
        assert ep.terrain.gap(pos, mean.radius) == pytest.approx(true_gap, abs=1e-9)
        assert roll.meta["from_state"]["start_gap"] == pytest.approx(true_gap, abs=1e-9)
        assert spin * mean.radius == pytest.approx(ep.ball_spin[t] * ep.ball.radius)
        assert pos[0] == ep.ball_pos[t, 0] and pos[1] == ep.ball_pos[t, 1] + info["dz"][i]
        for k in (1, 5):
            np.testing.assert_array_equal(preds[k][i], roll.ball_pos[k] - [0.0, info["dz"][i]])
    assert np.isfinite(preds[5]).all()


def _still(sim_cfg, terrain, ball, pos, vel, spin, frames, **kw):
    """A rollout stand-in: the ball stays where it starts."""
    pos = np.asarray(pos, dtype=float)
    return SimpleNamespace(ball_pos=np.tile(pos, (frames, 1)), meta={"from_state": {}})


def test_simulator_sees_nothing_after_t(short_sim_cfg, real_episode):
    ep = real_episode
    seen = []

    def spy(*args, **kw):
        seen.append(args[3:6])
        return _still(*args, **kw)

    for t in (3, 6, 9):
        poisoned = [a.astype(float).copy() for a in (ep.ball_pos, ep.ball_vel, ep.ball_spin)]
        for a in poisoned:
            a[t + 1 :] = np.nan
        seen.clear()
        preds, _ = baselines.simulator_predict(
            short_sim_cfg, ep.terrain, ep.ball, ep.ball.radius, *poisoned, np.array([t]), [1, 20],
            simulate=spy,
        )  # fmt: skip
        assert all(np.isfinite(np.hstack([np.ravel(v) for v in s])).all() for s in seen)
        np.testing.assert_array_equal(preds[20][0], ep.ball_pos[t])


def test_simulator_rejects_starts_without_a_full_horizon(short_sim_cfg, real_episode):
    ep = real_episode
    with pytest.raises(ValueError, match="k_max"):
        baselines.simulator_predict(
            short_sim_cfg, ep.terrain, ep.ball, ep.ball.radius,
            ep.ball_pos, ep.ball_vel, ep.ball_spin, np.array([10]), [20], simulate=_still,
        )  # fmt: skip


def test_evaluate_scores_extra_predictions_like_the_closed_form_ones():
    pos, vel = motion("ballistic", n=3)
    starts = valid_starts(T, C, max(HORIZONS))
    ep, t = (a.ravel() for a in np.meshgrid(np.arange(3), starts, indexing="ij"))
    copy = {k: baselines.predict("persistence", pos, vel, ep, t, k, DT) for k in HORIZONS}
    oov = np.zeros((3, T), bool)
    oov[2, 20:] = True
    out = baselines.evaluate(
        pos, vel, DT, starts, HORIZONS, C, out_of_view=oov, extra={"copy": copy}
    )
    assert out["error_m"]["copy"] == out["error_m"]["persistence"]
    assert out["ratio_to_persistence"]["copy"] == {str(k): 1.0 for k in HORIZONS}
    by_view = out["mean_error_m_by_view_at_t_plus_k"]
    assert by_view["copy"] == by_view["persistence"]
    with pytest.raises(ValueError, match="copy"):
        baselines.evaluate(pos, vel, DT, starts, HORIZONS, C, extra={"copy": {1: copy[1][:-1]}})


def test_run_baselines_feeds_the_simulator_each_episodes_own_terrain_ball_and_spin():
    overrides = [*SHORT, "model.horizons=[1, 2, 5, 20]"]
    cfg = load_config(BASE, overrides)
    build.build_split(cfg, "test", simulate=physics_episode)
    seen = []

    def spy(sim_cfg, terrain, ball, pos, vel, spin, frames, **kw):
        seen.append((terrain, ball, float(spin), frames))
        return _still(sim_cfg, terrain, ball, pos, vel, spin, frames, **kw)

    # "mean" twice: repeated params run once (len(seen) below counts every rollout)
    run, out = eval_cli.run_baselines(
        cfg, "test", sim_params=("true", "mean", "mean"), simulate=spy
    )
    assert list(out["simulator"]) == ["true", "mean"]
    names = {"simulator_true", "simulator_mean"}
    assert names <= set(out["error_m"]) and names <= set(out["mean_error_m_by_view_at_t_plus_k"])
    # The stand-in stays put, so the simulator must score exactly like persistence.
    assert out["error_m"]["simulator_true"] == out["error_m"]["persistence"]
    for k, s in out["error_m"]["simulator_mean"].items():
        assert s == pytest.approx(out["error_m"]["persistence"][k])
    windows_ = out["windows"]
    assert len(seen) == 2 * windows_ and {f for *_, f in seen} == {21}
    sim, mean = out["simulator"], mean_ball(cfg["sim"])
    assert sim["true"]["params"] == dict.fromkeys(
        ("radius", "friction", "restitution"), "per-episode"
    )
    assert (
        sim["true"]["shifted"] == 0 and sim["true"]["inside_as_given"] == sim["true"]["inside_true"]
    )
    assert sim["mean"]["params"] == {
        "radius": mean.radius, "friction": mean.friction, "restitution": mean.restitution
    }  # fmt: skip
    assert sim["mean"]["windows"] == windows_ and sim["mean"]["unmatched"] >= 0
    assert sim["mean"]["shifted"] == windows_  # a radius other than the true one always shifts
    true_balls = {b.radius for _, b, _, _ in seen[:windows_]}
    assert len(true_balls) == out["episodes"]  # each episode's own ball
    assert {b.radius for _, b, _, _ in seen[windows_:]} == {mean_ball(cfg["sim"]).radius}
    assert json.loads((run / "baselines.json").read_text())["simulator"] == sim


def test_cli_passes_sim_params_through(monkeypatch):
    got = []

    def fake_run(cfg, split, sim_params):
        got.append(list(sim_params))
        return paths.artifacts_root(), {
            "episodes": 1, "windows": 1, "starts": [3, 3], "horizons": [1],
            "error_m": {"simulator_true_radius": {"1": {"mean": 0.5}}},
            "ratio_to_persistence": {"simulator_true_radius": {"1": None}},
        }  # fmt: skip

    monkeypatch.setattr(eval_cli, "run_baselines", fake_run)
    base = ["--baselines", "--config", str(BASE), "--split", "test"]
    assert eval_cli.main(base) == 0
    assert eval_cli.main([*base, "--sim-params", "true_radius"]) == 0
    assert eval_cli.main([*base, "--sim-params"]) == 0
    assert got == [list(baselines.DEFAULT_SIM_PARAMS), ["true_radius"], []]
    with pytest.raises(SystemExit):
        eval_cli.main([*base, "--sim-params", "oracle"])


def test_table_columns_fit_long_baseline_names():
    name = "simulator_true_radius"
    result = {
        "horizons": [1],
        "error_m": {name: {"1": {"mean": 0.5}}, "persistence": {"1": {"mean": 1.0}}},
        "ratio_to_persistence": {name: {"1": 0.5}, "persistence": {"1": 1.0}},
    }
    head, row = eval_cli.format_table(result).splitlines()
    assert head.index(name) + len(name) == row.index("(0.500)") + len("(0.500)")


@pytest.mark.slow
def test_eval_cli_simulator_on_real_physics_end_to_end():
    overrides = [*SHORT, "model.horizons=[1, 2, 5, 10, 20]"]
    cfg = load_config(BASE, overrides)
    build.build_split(cfg, "test", simulate=physics_episode)
    args = ["--baselines", "--config", str(BASE), "--split", "test", "--sim-params", "true", "mean"]
    for o in overrides:
        args += ["--set", o]
    assert eval_cli.main(args) == 0
    (run,) = (paths.artifacts_root() / "runs").iterdir()
    out = json.loads((run / "baselines.json").read_text())
    for k in ("1", "5", "20"):
        assert out["error_m"]["simulator_true"][k]["p90"] <= 1e-3  # the oracle reproduces GT
        assert np.isfinite(out["error_m"]["simulator_mean"][k]["mean"])
    assert out["ratio_to_persistence"]["simulator_true"]["20"] < 0.01
