"""Scoring by motion regime (jepa_world/regimes.py): rolling, airborne, bounce, per window and k.

Fast tier: tiny model on CPU, fake episodes with contact and bounce patterns chosen per episode,
so every regime appears and the expected labels are known.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
from fake_episodes import SHORT, SMALL_SPLITS, Recorder, flagged_episode
from tiny import AUX, BASE, TINY

from jepa_world import eval as eval_cli
from jepa_world import metrics, regimes, train, viewer
from jepa_world.data import build
from jepa_world.data.dataset import load_split, valid_starts, window_frames
from jepa_world.data.splits import LeakageError
from jepa_world.models.jepa import batch_var, relative_error
from jepa_world.utils.config import load_config

EPISODES = ["data.source=episodes", *SHORT, "sim.frame_size=16"]  # T=30, 16 px for TINY's L1
FAR_AND_AUX = ["train.stages=[near, far, goal, pixel]"]  # every head built: k in 1..5, 10, 20
EVAL_SETS = [
    *SMALL_SPLITS,
    "eval.device=cpu",
    "eval.batch_size=5",  # does not divide the 28 test windows: the last batch is short
    "eval.sim_params=[]",
    "eval.probes.enabled=false",
]
ROLL, AIR, BOUNCE = (regimes.REGIMES.index(r) for r in ("rolling", "airborne", "bounce"))


def run_cfg(*extra: str) -> dict:
    return load_config(BASE, [*TINY, *AUX, *FAR_AND_AUX, *EPISODES, *extra])


def eval_cfg(*extra: str) -> dict:
    return load_config(BASE, [*EVAL_SETS, *extra])


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """One artifact root with dev and test built from flagged fake episodes."""
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


def make_run(cfg: dict, name="tiny-regimes"):
    torch.manual_seed(0)
    model = train.build_model(cfg)
    opt = train.build_optimizer(model, cfg)
    run = train.start_run(cfg, name, torch.device("cpu"))
    train.save_checkpoint(run / "checkpoints" / "step_0000003.pt", model, opt, 3, cfg)
    return run, model.eval()


def brute_force_label(contact, bounce, e, t, k, c):
    """The orchestrator's definition, written out as its baseline-only script does it."""
    if bounce[e, t + 1 : t + k + 1].any():
        return BOUNCE
    span = slice(t - c + 1, t + k + 1)
    if contact[e, span].all() and not bounce[e, span].any():
        return ROLL
    return AIR


# --- labels ---------------------------------------------------------------------------------------


def test_regime_labels_follow_the_definitions_at_their_edges():
    T, C, k = 12, 4, 3
    contact, bounce = np.ones((5, T), bool), np.zeros((5, T), bool)
    bounce[1, 7] = True  # ep 1: one landing at frame 7
    contact[2, 2] = False  # ep 2: one airborne frame at 2
    bounce[3, 4] = True  # ep 3: a landing at frame 4
    contact[4] = False  # ep 4: never touches
    cases = [
        (0, 5, ROLL),  # contact in 2..8, no bounce
        (1, 6, BOUNCE),  # 7 is in the horizon 7..9
        (1, 4, BOUNCE),  # 7 = t + k is the horizon's last frame
        (1, 3, ROLL),  # 7 = t + k + 1: after the horizon
        (1, 7, AIR),  # 7 = t: the landing is in the context, not ahead
        (2, 4, AIR),  # the airborne frame 2 is in the context 1..4
        (2, 6, ROLL),  # context 3..6: frame 2 is before it
        (3, 3, BOUNCE),  # 4 is in the horizon 4..6
        (3, 5, AIR),  # 4 is in the context 2..5
        (4, 5, AIR),
    ]
    ep, t, want = (np.array(col) for col in zip(*cases, strict=True))
    got = regimes.regime_labels(contact, bounce, ep, t, k, C)
    np.testing.assert_array_equal(got, want)
    assert got.shape == (len(cases),)


def test_regime_labels_match_the_definition_on_every_window_and_horizon():
    rng = np.random.default_rng(0)
    N, T, C = 6, 40, 4
    contact = rng.random((N, T)) < 0.85
    bounce = contact & (rng.random((N, T)) < 0.1)
    for k in (1, 2, 5, 10, 20):
        starts = valid_starts(T, C, k)
        ep, t = np.repeat(np.arange(N), len(starts)), np.tile(starts, N)
        got = regimes.regime_labels(contact, bounce, ep, t, k, C)
        want = [brute_force_label(contact, bounce, e, s, k, C) for e, s in zip(ep, t, strict=True)]
        np.testing.assert_array_equal(got, want)


def test_regime_labels_refuse_a_window_outside_the_episode():
    contact, bounce = np.ones((1, 10), bool), np.zeros((1, 10), bool)
    with pytest.raises(ValueError, match="outside"):
        regimes.regime_labels(contact, bounce, np.array([0]), np.array([2]), 1, 4)  # t-C+1 < 0
    with pytest.raises(ValueError, match="outside"):
        regimes.regime_labels(contact, bounce, np.array([0]), np.array([7]), 3, 4)  # t+k > T-1


# --- scoring ----------------------------------------------------------------------------------------


def _fake_scores_input(seed=0, m=60):
    rng = np.random.default_rng(seed)
    labels = {1: rng.integers(0, 3, m), 5: np.where(rng.random(m) < 0.5, ROLL, BOUNCE)}  # no AIR
    truth = {k: rng.normal(size=(m, 2)) for k in labels}
    base = {
        "persistence": {k: truth[k] + rng.normal(size=(m, 2)) for k in labels},
        "const_velocity": {k: truth[k] + 0.5 * rng.normal(size=(m, 2)) for k in labels},
    }
    goal = {
        "pred": {k: truth[k] + 0.3 * rng.normal(size=(m, 2)) for k in labels},
        "var": {k: np.exp(rng.normal(size=(m, 2))) for k in labels},
    }
    sq = {k: rng.random(m) for k in labels}
    spread = {1: 2.0, 5: 3.0}
    return labels, truth, base, goal, sq, spread


def test_scores_split_each_horizon_into_regimes_that_add_up_to_all_windows():
    labels, truth, base, goal, sq, spread = _fake_scores_input()
    s = regimes.score_regimes(labels, truth, base, goal, sq, spread, levels=[0.5, 0.9])
    assert set(s) == {"1", "5"}
    for k, lab in labels.items():
        rows = s[str(k)]
        assert set(rows) == {"all", *regimes.REGIMES}
        assert rows["all"]["windows"] == len(lab) and rows["all"]["share"] == 1.0
        assert sum(rows[r]["windows"] for r in regimes.REGIMES) == len(lab)
        assert sum(rows[r]["share"] for r in regimes.REGIMES) == pytest.approx(1.0)
        for name in ("goal_head", "persistence", "const_velocity"):
            pred = goal["pred"][k] if name == "goal_head" else base[name][k]
            err = np.linalg.norm(pred - truth[k], axis=-1)
            weighted = sum(
                rows[r]["windows"] * rows[r]["error_m"][name]
                for r in regimes.REGIMES
                if rows[r]["windows"]
            )  # fmt: skip
            assert weighted / len(lab) == pytest.approx(rows["all"]["error_m"][name])
            for r, i in [("all", None), *((r, regimes.REGIMES.index(r)) for r in regimes.REGIMES)]:
                mask = np.ones(len(lab), bool) if i is None else lab == i
                if not mask.any():
                    continue
                assert rows[r]["error_m"][name] == pytest.approx(err[mask].mean())
                assert rows[r]["mse_m2"][name] == pytest.approx((err[mask] ** 2).mean())
        for r in ("all", *regimes.REGIMES):
            row = rows[r]
            if not row["windows"]:
                continue
            for name in base:
                assert row["ratio"][name] == pytest.approx(
                    row["error_m"]["goal_head"] / row["error_m"][name]
                )
            mask = np.ones(len(lab), bool) if r == "all" else lab == regimes.REGIMES.index(r)
            cov = metrics.coverage(goal["pred"][k][mask], goal["var"][k][mask], truth[k][mask])
            assert row["coverage"] == {p: cov[p]["joint"] for p in ("0.5", "0.9")}
            assert row["relerr"] == pytest.approx(sq[k][mask].mean() / spread[k])
        # Normalised by the full set's spread, the regimes' relerr average back to the full set's.
        weighted = sum(
            rows[r]["share"] * rows[r]["relerr"] for r in regimes.REGIMES if rows[r]["windows"]
        )
        assert weighted == pytest.approx(rows["all"]["relerr"])


def test_an_empty_regime_reports_zero_windows_and_no_numbers():
    labels, truth, base, goal, sq, spread = _fake_scores_input()
    s = regimes.score_regimes(labels, truth, base, goal, sq, spread, levels=[0.9])
    air = s["5"]["airborne"]
    assert air["windows"] == 0 and air["share"] == 0.0
    assert air["relerr"] is None and air["coverage"] is None
    assert set(air["error_m"].values()) == {None} and set(air["ratio"].values()) == {None}
    json.dumps(s, allow_nan=False)  # JSON-safe, no NaN


def test_a_collapsed_target_gives_a_finite_relerr_as_training_does():
    labels, truth, base, goal, sq, _ = _fake_scores_input()
    s = regimes.score_regimes(labels, truth, base, goal, sq, {1: 0.0, 5: 0.0}, [0.9])
    assert s["1"]["all"]["relerr"] == pytest.approx(sq[1].mean() / regimes.SPREAD_EPS)
    json.dumps(s, allow_nan=False)


@pytest.mark.parametrize("bad", ["nan", "shape"])
def test_scores_refuse_a_prediction_that_is_not_finite_or_not_one_per_window(bad):
    labels, truth, base, goal, sq, spread = _fake_scores_input()
    p = base["const_velocity"][5]
    base["const_velocity"][5] = (
        np.where(np.arange(len(p))[:, None] == 3, np.nan, p) if bad == "nan" else p[:-1]
    )
    with pytest.raises(ValueError, match="const_velocity at k=5"):
        regimes.score_regimes(labels, truth, base, goal, sq, spread, [0.9])


def test_scores_without_a_goal_head_or_a_latent_horizon_leave_those_fields_none():
    labels, truth, base, goal, sq, spread = _fake_scores_input()
    s = regimes.score_regimes(labels, truth, base, None, {1: sq[1]}, {1: spread[1]}, [0.9])
    assert s["1"]["all"]["relerr"] is not None and s["5"]["all"]["relerr"] is None
    assert s["1"]["all"]["coverage"] is None and "goal_head" not in s["1"]["all"]["error_m"]
    assert set(s["1"]["all"]["ratio"].values()) == {None}


# --- the model pass ---------------------------------------------------------------------------------


def test_window_outputs_match_one_forward_over_every_window(artifacts):
    # Batched (5 windows a batch, the last one short) must give what one forward over all the
    # windows gives: per-window squared error, the full set's target spread, the goal head.
    cfg = run_cfg()
    _, model = make_run(cfg)
    es = load_split(cfg, "test")
    c = model.context_frames
    ep = np.repeat(np.arange(es.n), 3)
    t = np.tile([3, 6, 9], es.n)
    out = regimes.window_outputs(model, es, ep, t, batch_size=5, device=torch.device("cpu"))
    frames = window_frames(es, ep, t, c, max(model.horizons))
    with torch.no_grad():
        ref = model(frames)
    assert set(out["sqerr"]) == set(model.horizons) == {1, 2, 3, 4, 5, 10, 20}
    for k in model.horizons:
        pred, target = ref["pred"][k], ref["target"][k]
        sq = (target - pred).square().sum(dim=1).mean(dim=(1, 2)).double().numpy()
        assert out["sqerr"][k].shape == (len(ep),) and out["sqerr"][k].dtype == np.float64
        np.testing.assert_allclose(out["sqerr"][k], sq, rtol=1e-5)
        spread = batch_var(target.double()).sum(dim=0).mean().item()
        # float32 targets from batches of 5 and of 12 differ in their last bits; the exact
        # accumulation is checked on its own below
        assert out["spread"][k] == pytest.approx(spread, rel=1e-5)
        # mean sqerr over spread is the training relerr with all the windows as one batch
        assert out["sqerr"][k].mean() / out["spread"][k] == pytest.approx(
            relative_error(pred, target).item(), rel=1e-5
        )
    ctx = eval_cli.model_outputs(model, es, ep, t, 5, torch.device("cpu"), features=False)
    np.testing.assert_allclose(out["goal_mean"], ctx["goal_mean"], atol=1e-6)
    np.testing.assert_allclose(out["goal_logvar"], ctx["goal_logvar"], atol=1e-6)


def test_window_outputs_never_run_the_pixel_decoder(artifacts, monkeypatch):
    cfg = run_cfg()
    _, model = make_run(cfg)
    assert model.pixel is not None

    def refuse(*a, **kw):
        raise AssertionError("the pixel decoder ran")

    monkeypatch.setattr(model.pixel, "forward", refuse)
    es = load_split(cfg, "test")
    out = regimes.window_outputs(
        model, es, np.array([0, 1]), np.array([3, 4]), 2, torch.device("cpu")
    )
    assert out["goal_mean"].shape == (2, 3, 2)


def test_spread_accumulates_exactly_over_uneven_batches_whatever_the_mean():
    # A large offset with a small spread: the case a plain sum of squares gets wrong.
    z = 1e4 + torch.randn(
        23, 3, 4, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    m = regimes._Moments()
    for lo, hi in [(0, 5), (5, 6), (6, 23)]:
        m.add(z[lo:hi])
    assert m.n == 23
    assert m.spread() == pytest.approx(batch_var(z).sum(dim=0).mean().item(), rel=1e-9)


def test_episode_flags_come_from_the_episode_files_in_row_order(artifacts):
    cfg = run_cfg()
    es = load_split(cfg, "test")
    contact, bounce = regimes.episode_flags(cfg, "test", es)
    assert contact.shape == bounce.shape == (es.n, es.T) and contact.dtype == bool
    for i in range(es.n):
        want = flagged_episode(cfg["sim"], int(es.terrain_seed[i]), int(es.ball_seed[i]))
        np.testing.assert_array_equal(contact[i], want.contact)
        np.testing.assert_array_equal(bounce[i], want.bounce)
    moved = es.ball_pos.copy()
    moved[1, 4] += 1.0
    with pytest.raises(ValueError, match="differ"):
        regimes.episode_flags(cfg, "test", dataclasses.replace(es, ball_pos=moved))


# --- the command, end to end ------------------------------------------------------------------------


class ConstVelocityRollout:
    """A `simulate` stand-in: x(t0 + j) = x(t0) + v(t0) j dt, so simulator_mean = const_velocity."""

    def __call__(self, sim_cfg, terrain, ball, pos, vel, spin, n, cam_pos=None):
        j = np.arange(n)[:, None] / sim_cfg["fps"]
        return dataclasses.make_dataclass("Roll", ["ball_pos"])(
            np.asarray(pos) + np.asarray(vel) * j
        )


def test_evaluate_regimes_writes_one_file_whose_all_row_is_the_eval_headline(artifacts):
    cfg = run_cfg()
    run, _ = make_run(cfg)
    doc, path = regimes.evaluate_regimes(run, eval_cfg(), "test")
    assert path.parent == run / "eval" / "regimes" and "regimes-test" in path.name
    assert json.loads(path.read_text()) == doc
    meta, scores = doc["meta"], doc["regimes"]
    assert meta["split"] == "test" and meta["step"] == 3 and meta["windows"] == 4 * 7
    assert meta["starts"] == [3, 9]  # t in [C-1, T-1-max k], k max 20, as eval's windows
    assert set(scores) == {str(k) for k in (1, 2, 3, 4, 5, 10, 20)}

    # Window counts per regime: the definition applied to the episode files themselves.
    es = load_split(cfg, "test")
    contact = np.stack([flagged_episode(cfg["sim"], int(a), int(b)).contact
                        for a, b in zip(es.terrain_seed, es.ball_seed, strict=True)])  # fmt: skip
    bounce = np.stack([flagged_episode(cfg["sim"], int(a), int(b)).bounce
                       for a, b in zip(es.terrain_seed, es.ball_seed, strict=True)])  # fmt: skip
    ep, t = np.repeat(np.arange(es.n), 7), np.tile(np.arange(3, 10), es.n)
    for k in (1, 5, 20):
        want = [brute_force_label(contact, bounce, e, s, k, 4) for e, s in zip(ep, t, strict=True)]
        for i, r in enumerate(regimes.REGIMES):
            assert scores[str(k)][r]["windows"] == want.count(i), (k, r)
    seen = {r for k in scores for r in regimes.REGIMES if scores[k][r]["windows"]}
    assert seen == set(regimes.REGIMES)

    # The all-windows row is the eval's headline for the same checkpoint and windows.
    report, _ = eval_cli.evaluate_run(run, eval_cfg(), ["test"])
    goal = report["report"]["headline"]["metrics"]["goal_head"]
    for k in ("1", "3", "5"):
        row = scores[k]["all"]
        assert row["error_m"]["goal_head"] == pytest.approx(goal["error_m"][k]["mean"])
        for name in ("persistence", "const_velocity", "const_velocity_fd"):
            assert row["ratio"][name] == pytest.approx(goal["ratio_to_baseline"][name][k])
        for p in ("0.5", "0.9"):
            assert row["coverage"][p] == pytest.approx(goal["calibration"][k][p]["joint"])
    # Latent horizons without a goal horizon (2, 4, 10, 20) still get relerr; the rest is None.
    assert scores["10"]["all"]["relerr"] is not None
    assert scores["10"]["all"]["coverage"] is None


def test_evaluate_regimes_scores_the_simulator_baseline_on_the_same_windows(artifacts):
    cfg = run_cfg()
    run, _ = make_run(cfg, name="tiny-regimes-sim")
    doc, _ = regimes.evaluate_regimes(
        run, eval_cfg('eval.sim_params=["mean"]'), "test", simulate=ConstVelocityRollout()
    )
    for k in ("1", "5", "20"):
        for r in ("all", *regimes.REGIMES):
            row = doc["regimes"][k][r]
            if row["windows"]:
                assert row["error_m"]["simulator_mean"] == pytest.approx(
                    row["error_m"]["const_velocity"]
                )
    assert doc["meta"]["simulator"]["mean"]["windows"] == 28


def test_evaluate_regimes_scores_held_out_splits_only_and_needs_a_goal_head(artifacts, monkeypatch):
    cfg = run_cfg()
    run, _ = make_run(cfg, name="tiny-regimes-refuse")
    loads = []
    monkeypatch.setattr(regimes, "load_run", lambda *a, **kw: loads.append(a) or None)
    with pytest.raises(ValueError, match="dev or test"):
        regimes.evaluate_regimes(run, eval_cfg(), "train")
    assert loads == []  # refused before anything was loaded
    monkeypatch.undo()
    dev_run, _ = make_run(run_cfg("data.split=dev"), name="tiny-dev-trained")
    with pytest.raises(LeakageError, match="trained on"):
        regimes.evaluate_regimes(dev_run, eval_cfg(), "dev")
    near, _ = make_run(load_config(BASE, [*TINY, *EPISODES]), name="tiny-no-goal")
    with pytest.raises(ValueError, match="goal head"):
        regimes.evaluate_regimes(near, eval_cfg(), "test")


def test_cli_prints_a_table_per_horizon_and_the_file(artifacts, capsys):
    cfg = run_cfg()
    run, _ = make_run(cfg, name="tiny-regimes-cli")
    args = ["--run", str(run), "--split", "test"]
    for s in EVAL_SETS:
        args += ["--set", s]
    assert regimes.main(args) == 0
    out = capsys.readouterr().out
    (path,) = (run / "eval" / "regimes").glob("*regimes-test*.json")
    assert str(path) in out
    for word in ("rolling", "airborne", "bounce", "k=20", "relerr"):
        assert word in out


def test_regimes_files_stay_out_of_the_eval_report_glob(artifacts):
    # viewer.eval_summary and the experiment scripts read every <run>/eval/*.json as an eval
    # --run report (meta.splits, meta.checkpoint, report). A regimes file must not be among them.
    cfg = run_cfg()
    run, _ = make_run(cfg, name="tiny-regimes-beside-eval")
    report, _ = eval_cli.evaluate_run(run, eval_cfg(), ["test"])
    regimes.evaluate_regimes(run, eval_cfg(), "test")
    docs = [json.loads(p.read_text()) for p in (run / "eval").glob("*.json")]
    assert len(docs) == 1 and all("report" in d and "splits" in d["meta"] for d in docs)
    summary = viewer.eval_summary(run, "step_0000003.pt", "test")
    goal = report["report"]["headline"]["metrics"]["goal_head"]
    assert summary["by_k"]["1"]["error_m"] == goal["error_m"]["1"]["mean"]
