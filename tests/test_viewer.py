"""The pred/actual viewer, `python -m jepa_world.viewer` (jepa_world/DESIGN.md, "The pred/actual
viewer").

Pure pieces on synthetic arrays, then the exporter end to end with tiny CPU models on tiny fake
datasets. The numbers are meaningless; what is tested is that the page shows the goal head's own
output at the right times, only on held-out data, and fits in one self-contained file."""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import math
import re

import numpy as np
import pytest
import torch
from fake_episodes import SHORT, SMALL_SPLITS, Recorder
from PIL import Image
from tiny import AUX, BASE, TINY

from jepa_world import eval as eval_cli
from jepa_world import metrics, train, viewer
from jepa_world.data import build
from jepa_world.data.dataset import load_split
from jepa_world.data.splits import LeakageError
from jepa_world.sim import camera
from jepa_world.utils.config import load_config

EPISODES = ["data.source=episodes", *SHORT, "sim.frame_size=16"]  # T=30, 16 px for TINY's L1
GOAL_KS = [1, 3, 5]  # AUX's model.goal_horizons
EVAL_SETS = [*SMALL_SPLITS, "eval.device=cpu", "eval.batch_size=8"]


def run_cfg(*extra: str) -> dict:
    return load_config(BASE, [*TINY, *AUX, *EPISODES, *extra])


def eval_cfg(*extra: str) -> dict:
    return load_config(BASE, [*EVAL_SETS, *extra])


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """One artifact root with dev and test built (fake episodes), shared by this module."""
    root = tmp_path_factory.mktemp("artifacts")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("JEPA_ARTIFACTS", str(root))
        for split in ("dev", "test"):
            build.build_split(run_cfg(), split, simulate=Recorder())
    return root


@pytest.fixture
def artifacts(root, monkeypatch):
    monkeypatch.setenv("JEPA_ARTIFACTS", str(root))
    return root


def make_run(cfg: dict, steps=(3,), name="tiny-viewer", seed=0):
    torch.manual_seed(seed)
    model = train.build_model(cfg)
    opt = train.build_optimizer(model, cfg)
    run = train.start_run(cfg, name, torch.device("cpu"))
    for s in steps:
        train.save_checkpoint(run / "checkpoints" / f"step_{s:07d}.pt", model, opt, s, cfg)
    return run, model


def strip_frames(uri: str, n: int) -> np.ndarray:
    """A data:image/png;base64 horizontal strip -> (n, S, S, 3) uint8."""
    head, b64 = uri.split(",", 1)
    assert head == "data:image/png;base64"
    a = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    s = a.shape[0]
    assert a.shape == (s, n * s, 3)
    return a.reshape(s, n, s, 3).transpose(1, 0, 2, 3)


def page_payload(html: str) -> dict:
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    assert m, "the page carries its data in one application/json script"
    return json.loads(m.group(1))


# --- pure pieces ------------------------------------------------------------------------------


def test_model_spec_is_label_run_and_optional_step(tmp_path):
    s = viewer.parse_model_spec(f"24k headline={tmp_path}@24000")
    assert (s.label, s.run_dir, s.step) == ("24k headline", tmp_path, 24000)
    assert viewer.parse_model_spec(f"48k={tmp_path}").step is None
    for bad in (str(tmp_path), f"={tmp_path}", f"x={tmp_path}@24k", "x="):
        with pytest.raises(ValueError, match="LABEL=RUN_DIR"):
            viewer.parse_model_spec(bad)


def test_episodes_are_evenly_spaced_and_never_chosen_by_the_model():
    idx = viewer.pick_episodes(200, 8)
    assert idx.tolist() == sorted(set(idx.tolist())) and len(idx) == 8
    assert idx[0] == 0 and idx[-1] == 199
    assert np.ptp(np.diff(idx)) <= 1  # even spacing, up to rounding
    np.testing.assert_array_equal(viewer.pick_episodes(5, 8), np.arange(5))
    np.testing.assert_array_equal(viewer.pick_episodes(200, 1), [0])
    with pytest.raises(ValueError, match="episodes"):
        viewer.pick_episodes(200, 0)


def test_goal_track_is_ball_pos_at_t_plus_the_mean_with_std_from_logvar():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(12, 2))
    t = np.arange(3, 12)
    mean, logvar = rng.normal(size=(2, len(t), 3, 2))
    pred, std = viewer.goal_track(pos, t, mean, logvar)
    assert pred.shape == std.shape == (len(t), 3, 2)
    np.testing.assert_allclose(pred, pos[t][:, None] + mean)
    np.testing.assert_allclose(std**2, np.exp(logvar))
    # The same positions eval scores (goal_positions), wherever t+k is inside the episode.
    g = eval_cli.goal_positions(pos[None], np.zeros(3, int), t[:3], mean[:3], logvar[:3], [1, 2, 3])
    for j, k in enumerate([1, 2, 3]):
        np.testing.assert_allclose(pred[:3, j], g["pred"][k])
        np.testing.assert_allclose(std[:3, j] ** 2, g["var"][k])
    with pytest.raises(ValueError, match="shape"):
        viewer.goal_track(pos, t, mean, logvar[:, :2])


def test_region_scales_are_the_joint_regions_metrics_counts():
    # A point just inside scale * sigma along one axis is covered at that level, just outside is
    # not: the ellipses drawn are exactly the regions behind eval's coverage numbers.
    for r in viewer.region_scales([0.5, 0.9]):
        p, scale = r["p"], r["scale"]
        assert math.isclose(scale, math.sqrt(-2 * math.log(1 - p)))
        var = np.array([[4.0, 0.25]])
        for axis in (0, 1):
            for f, covered in ((0.999, 1.0), (1.001, 0.0)):
                truth = np.zeros((1, 2))
                truth[0, axis] = f * scale * math.sqrt(var[0, axis])
                cov = metrics.coverage(np.zeros((1, 2)), var, truth, [p])
                assert cov[str(p)]["joint"] == covered, (p, axis, f)


def test_frames_round_trip_losslessly_through_a_png_strip():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(5, 4, 4, 3), dtype=np.uint8)
    np.testing.assert_array_equal(strip_frames(viewer.png_strip(frames), 5), frames)
    out = viewer.to_uint8(np.array([[[[-0.1, 0.0, 0.5 / 255, 1.0, 1.7]]]] * 3).reshape(1, 3, 1, 5))
    assert out.shape == (1, 1, 5, 3) and out.dtype == np.uint8
    assert out[0, 0, :, 0].tolist() == [0, 0, 1, 255, 255]  # clipped to [0, 1], rounded


def _eval_doc(checkpoint: str, splits=("test", "dev")) -> dict:
    goal = {
        "horizons": [1, 5],
        "windows": 7,
        "error_m": {"1": {"mean": 0.01}, "5": {"mean": 0.05}},
        "ratio_to_persistence": {"1": 0.3, "5": 0.25},
        "ratio_to_baseline": {"const_velocity": {"1": 1.5, "5": 0.4}},
        "calibration": {
            "1": {"0.5": {"joint": 0.6}, "0.9": {"joint": 0.91}},
            "5": {"0.5": {"joint": 0.55}, "0.9": {"joint": 0.88}},
        },
    }
    base = {"episodes": 4, "windows": 7, "starts": [3, 9]}
    metrics_ = {"goal_head": goal, "baselines": base}
    return {
        "meta": {"checkpoint": checkpoint, "step": 3, "splits": list(splits)},
        "report": {
            "headline": {"split": "test", "metrics": metrics_} if "test" in splits else None,
            "development": {"dev": metrics_} if "dev" in splits else {},
        },
    }


def test_eval_summary_reads_only_an_eval_of_the_same_checkpoint(tmp_path):
    ckpt = tmp_path / "checkpoints" / "step_0000003.pt"
    assert viewer.eval_summary(tmp_path, ckpt, "test") is None  # no eval/ yet
    (tmp_path / "eval").mkdir()
    other = tmp_path / "checkpoints" / "step_0000002.pt"
    (tmp_path / "eval" / "a.json").write_text(json.dumps(_eval_doc(str(other))))
    assert viewer.eval_summary(tmp_path, ckpt, "test") is None
    (tmp_path / "eval" / "b.json").write_text(json.dumps(_eval_doc(str(ckpt), ("dev",))))
    assert viewer.eval_summary(tmp_path, ckpt, "test") is None  # that eval has no test
    dev = viewer.eval_summary(tmp_path, ckpt, "dev")
    assert dev["file"] == "b.json" and dev["split"] == "dev"
    (tmp_path / "eval" / "c.json").write_text(json.dumps(_eval_doc(str(ckpt))))
    s = viewer.eval_summary(tmp_path, ckpt, "test")
    assert s["file"] == "c.json" and s["split"] == "test" and s["headline"] is True
    assert (s["episodes"], s["windows"], s["starts"]) == (4, 7, [3, 9])
    assert s["by_k"]["5"] == {
        "error_m": 0.05,
        "ratio_to_persistence": 0.25,
        "ratio_to_const_velocity": 0.4,
        "ratio_to_simulator_mean": None,  # that eval ran no simulator
        "coverage": {"0.5": 0.55, "0.9": 0.88},
    }


def test_the_page_is_one_self_contained_file_whose_data_cannot_close_its_script():
    payload = {"title": "x", "models": [{"label": "</script><script>alert(1)</script>"}]}
    html = viewer.render_html(payload)
    assert "<title>" in html[:8192]
    assert '<meta charset="utf-8">' in html[:1024]  # read as UTF-8 when opened from disk too
    assert "</script><script>alert(1)" not in html
    assert page_payload(html) == payload
    assert not re.search(r"<(script|link|img)[^>]+(src|href)=\"?(https?:)?//", html)
    assert "fetch(" not in html and "@import" not in html


# --- the context path, with the pixel decoder ---------------------------------------------------


def test_model_outputs_pixel_matches_forward_and_ignores_frames_after_t(artifacts):
    cfg = run_cfg()
    _, model = make_run(cfg)
    model.eval()
    es = load_split(cfg, "test")
    ep, t = np.arange(es.n), np.full(es.n, 5)
    ours = eval_cli.model_outputs(model, es, ep, t, 3, torch.device("cpu"), pixel=True)
    frames = eval_cli.window_frames(es, ep, t, model.context_frames, max(model.horizons))
    with torch.no_grad():
        out = model(frames)
    assert ours["pixel"].shape == (es.n, 3, 16, 16) and ours["pixel"].dtype == np.float32
    np.testing.assert_allclose(ours["pixel"], out["pixel"].numpy(), atol=1e-6)
    later = es.frames.clone()
    later[:, 6:] = torch.randint(0, 256, later[:, 6:].shape, dtype=torch.uint8)
    b = eval_cli.model_outputs(
        model, dataclasses.replace(es, frames=later), ep, t, 3, torch.device("cpu"),
        pixel=True,
    )  # fmt: skip
    np.testing.assert_array_equal(ours["pixel"], b["pixel"])
    assert eval_cli.model_outputs(model, es, ep, t, 3, torch.device("cpu"))["pixel"] is None


# --- the exporter, end to end -------------------------------------------------------------------


def test_export_writes_one_page_with_each_models_outputs_on_held_out_episodes(artifacts):
    cfg = run_cfg()
    run_a, model_a = make_run(cfg, steps=(2, 3), name="aux")
    run_b, _ = make_run(run_cfg("train.stages=[near, goal]"), name="goal-only", seed=1)
    specs = [
        viewer.ModelSpec("aux at 2", run_a, 2),
        viewer.ModelSpec("goal only", run_b, None),
    ]
    path, payload = viewer.export(specs, "test", 2, eval_cfg())
    assert path.parent == artifacts / "viewers" and path.suffix == ".html"
    html = path.read_text()
    assert page_payload(html) == payload
    assert payload["split"] == "test" and "100000" in payload["split_note"]

    es = load_split(cfg, "test")
    T, S, C = es.T, es.frames.shape[-1], cfg["model"]["context_frames"]
    picked = viewer.pick_episodes(es.n, 2)
    assert [e["terrain_seed"] for e in payload["episodes"]] == es.terrain_seed[picked].tolist()
    for e, i in zip(payload["episodes"], picked, strict=True):
        np.testing.assert_allclose(e["pos"], es.ball_pos[i], atol=1e-5)
        np.testing.assert_array_equal(
            strip_frames(e["frames"], T), es.frames[i].permute(0, 2, 3, 1).numpy()
        )

    a, b = payload["models"]
    assert (a["label"], a["step"], a["checkpoint"]) == ("aux at 2", 2, "step_0000002.pt")
    assert b["step"] == 3 and b["pixel"] is False and b["goal"] is True
    assert a["trained_on"] == "train" and a["t0"] == C - 1 and a["horizons"] == GOAL_KS
    assert a["scored"] == [C - 1, T - 1 - 20]  # eval's windows: k_eval = max(model.horizons)

    # The goal head's own output, through the same context-only path eval scores.
    loaded = eval_cli.load_run(run_a, "step_0000002.pt")
    t = np.arange(C - 1, T)
    for e, i in zip(a["episodes"], picked, strict=True):
        out = eval_cli.model_outputs(
            loaded.model, es, np.full(len(t), i), t, 8, torch.device("cpu"), features=False
        )
        pred, std = viewer.goal_track(es.ball_pos[i], t, out["goal_mean"], out["goal_logvar"])
        np.testing.assert_allclose(e["pred"], pred, atol=1e-5)
        np.testing.assert_allclose(e["std"], std, rtol=1e-3)
        cam = es.cam_pos[i][t][:, None]
        np.testing.assert_allclose(e["pred_px"], camera.project(pred, cam, cfg["sim"]), atol=1e-3)
        truth = np.array(e["true_px"], dtype=float)  # null -> nan: t+k past the episode's end
        scored = t <= a["scored"][1]
        for j, k in enumerate(GOAL_KS):
            inside = t + k <= T - 1
            assert np.isnan(truth[~inside, j]).all()
            np.testing.assert_allclose(
                truth[inside, j],
                camera.project(es.ball_pos[i][t[inside] + k], cam[inside, 0], cfg["sim"]),
                atol=1e-3,
            )
            future = es.ball_pos[i][t[scored] + k]  # this episode's errors: eval's windows only
            err = np.linalg.norm(pred[scored, j] - future, axis=1).mean()
            stay = np.linalg.norm(es.ball_pos[i][t[scored]] - future, axis=1).mean()
            assert e["errors"][str(k)]["model"] == pytest.approx(err)
            assert e["errors"][str(k)]["persistence"] == pytest.approx(stay)
        dec = strip_frames(e["pixel"], len(t))  # decoder output from windows t: frames t+1
        assert dec.shape == (len(t), S, S, 3)
        np.testing.assert_array_equal(dec, viewer.to_uint8(out_pixel(loaded.model, es, i, t)))
        real = es.frames[i].float().div(255).numpy()
        mae = np.abs(out_pixel(loaded.model, es, i, t)[:-1] - real[t[:-1] + 1]).mean(axis=(1, 2, 3))
        np.testing.assert_allclose(e["pixel_mae"][:-1], mae, rtol=1e-3)
        assert e["pixel_mae"][-1] is None and e["copy_mae"][-1] is None  # no frame T
        copy = np.abs(real[t[:-1]] - real[t[:-1] + 1]).mean(axis=(1, 2, 3))
        np.testing.assert_allclose(e["copy_mae"][:-1], copy, rtol=1e-3)
    assert all("pixel" not in e for e in b["episodes"])
    assert payload["regions"] == viewer.region_scales(eval_cfg()["eval"]["coverage_levels"])


def out_pixel(model, es, i, t):
    o = eval_cli.model_outputs(
        model, es, np.full(len(t), i), t, 8, torch.device("cpu"), goal=False, features=False,
        pixel=True,
    )  # fmt: skip
    return o["pixel"]


def test_export_refuses_the_training_split_before_writing_anything(artifacts):
    run, _ = make_run(run_cfg())
    pages = artifacts / "viewers"
    before = sorted(pages.iterdir()) if pages.exists() else []  # the module shares one root
    with pytest.raises(LeakageError, match="trained on"):
        viewer.export([viewer.ModelSpec("m", run, None)], "train", 2, eval_cfg())
    with pytest.raises(ValueError, match="dev.*test"):
        viewer.export([viewer.ModelSpec("m", run, None)], "val", 2, eval_cfg())
    assert (sorted(pages.iterdir()) if pages.exists() else []) == before


def test_export_refuses_models_that_read_different_datasets(artifacts):
    run_a, _ = make_run(run_cfg())
    run_b, _ = make_run(run_cfg("sim.camera.lerp_x=0.5"), name="other-data")
    specs = [viewer.ModelSpec("a", run_a, None), viewer.ModelSpec("b", run_b, None)]
    with pytest.raises(ValueError, match="same dataset"):
        viewer.export(specs, "test", 2, eval_cfg())


def test_export_refuses_a_page_over_the_size_limit_or_inside_the_repo(artifacts, tmp_path):
    run, _ = make_run(run_cfg())
    spec = [viewer.ModelSpec("m", run, None)]
    with pytest.raises(ValueError, match="MB"):
        viewer.export(spec, "test", 2, eval_cfg(), max_bytes=1000)
    with pytest.raises(ValueError, match="outside the git"):
        viewer.export(spec, "test", 2, eval_cfg(), out=viewer.REPO_ROOT / "viewer.html")
    out = tmp_path / "v.html"
    path, _ = viewer.export(spec, "test", 2, eval_cfg(), out=out)
    assert path == out and out.exists()
    with pytest.raises(FileExistsError):
        viewer.export(spec, "test", 2, eval_cfg(), out=out)  # never over an earlier page


def test_cli_takes_repeatable_models(artifacts, capsys):
    run, _ = make_run(run_cfg())
    rc = viewer.main(
        ["--model", f"m={run}", "--model", f"m3={run}@3", "--episodes", "1", "--config",
         str(BASE), *(a for s in EVAL_SETS for a in ("--set", s))]
    )  # fmt: skip
    assert rc == 0
    out = capsys.readouterr().out
    path = re.search(r"viewer: (\S+\.html)", out).group(1)
    payload = page_payload(open(path).read())
    assert [m["label"] for m in payload["models"]] == ["m", "m3"]
