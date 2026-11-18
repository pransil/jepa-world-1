"""Obstacles and occlusion (M7 plan §5; SPEC §3.2 track_lagged_obstacles, §3.4 `occluded`).

docs/decisions/2026-11-17-obstacles-dataset-keying.md: an absent `sim.obstacles` means off. Then the
scene and every output are what they were before obstacles existed, so M2's datasets stay valid.
The fast tier covers placement, the occlusion rule and physics invariance (no GL). The two
render-tier tests at the bottom cross-check the geometric flag against segmentation.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from jepa_world.sim import camera, obstacles, preview
from jepa_world.sim.ball import sample_ball
from jepa_world.sim.episode import simulate_episode, simulate_from_state
from jepa_world.sim.io import load_episode, save_episode
from jepa_world.sim.scene import body_id, build_model, geom_id
from jepa_world.sim.terrain import Terrain, sample_terrain
from jepa_world.utils.config import load_config

ROOT = Path(__file__).parents[1]
# Every ground-truth array except `occluded`: obstacles must not change any of them.
PHYSICS = (
    "t",
    "ball_pos",
    "ball_vel",
    "ball_spin",
    "cam_pos",
    "ball_px",
    "contact",
    "bounce",
    "out_of_view",
)


@pytest.fixture(scope="module")
def sim_cfg():
    return load_config(ROOT / "configs" / "base.yaml")["sim"]


@pytest.fixture(scope="module")
def obs_cfg():
    return load_config(ROOT / "configs" / "m7_obstacles.yaml")["sim"]


def with_obstacles(sim_cfg, **block) -> dict:
    cfg = copy.deepcopy(sim_cfg)
    cfg["obstacles"] = block
    return cfg


def posts(*bands, z=(-100.0, 100.0)) -> obstacles.Obstacles:
    lo, hi = zip(*bands, strict=True)
    return obstacles.Obstacles(np.array(lo, float), np.array(hi, float), *z)


def occluded_at(sim_cfg, xz, radius, obs, cam=(0.0, 0.0)) -> list[bool]:
    xz = np.atleast_2d(np.asarray(xz, float))
    cam = np.broadcast_to(np.asarray(cam, float), xz.shape)
    return obstacles.occluded(camera.project(xz, cam, sim_cfg), radius, cam, obs, sim_cfg).tolist()


# ------------------------------------------------- absent means off (the dataset-hash guarantee)


def test_base_config_has_no_obstacles_key(sim_cfg):
    # Even an "off" key here would change the sim hash and re-key every M2 dataset.
    assert "obstacles" not in sim_cfg


def test_m7_config_adds_only_the_obstacles_block(sim_cfg, obs_cfg):
    assert {k: v for k, v in obs_cfg.items() if k != "obstacles"} == sim_cfg
    assert set(obs_cfg["obstacles"]) == set(obstacles.KEYS)


def test_absent_key_builds_the_m1_scene_and_records_nothing(sim_cfg):
    terrain, ball = sample_terrain(sim_cfg, 3), sample_ball(sim_cfg, 3, 0)
    assert obstacles.sample_obstacles(sim_cfg, terrain) is None
    m = build_model(terrain, ball, sim_cfg)
    assert (m.ngeom, m.nbody) == (2, 3)  # terrain + ball; world, rig, ball
    ep = simulate_episode(sim_cfg, 3, 0, render=False)
    assert "obstacles" not in ep.meta and not ep.occluded.any()


# ------------------------------------------------- obstacles change nothing physical


@pytest.mark.parametrize("terrain_seed", [0, 1, 2, 3])
def test_obstacles_leave_every_physics_array_bitwise_unchanged(sim_cfg, obs_cfg, terrain_seed):
    off = simulate_episode(sim_cfg, terrain_seed, 0, render=False)
    on = simulate_episode(obs_cfg, terrain_seed, 0, render=False)
    for f in PHYSICS:
        np.testing.assert_array_equal(getattr(on, f), getattr(off, f), err_msg=f)


def test_posts_are_static_and_never_collide(obs_cfg):
    terrain, ball = sample_terrain(obs_cfg, 3), sample_ball(obs_cfg, 3, 0)
    obs = obstacles.sample_obstacles(obs_cfg, terrain)
    m = build_model(terrain, ball, obs_cfg, obs)
    ids = np.flatnonzero(m.geom_bodyid == body_id(m, "obstacles"))
    assert len(ids) == len(obs.x_lo) > 0
    assert (m.geom_contype[ids] == 0).all() and (m.geom_conaffinity[ids] == 0).all()
    assert (geom_id(m, "terrain"), geom_id(m, "ball")) == (0, 1)  # ids unchanged by the posts
    assert m.nq == 3 and m.nv == 3  # no new degrees of freedom


def test_from_state_gets_the_same_posts_and_flags(sim_cfg, obs_cfg):
    ep = simulate_episode(obs_cfg, 2, 0, render=False)
    # An airborne start reproduces the episode bitwise (sim/DESIGN.md), occluded included.
    t = next(i for i in range(5, 30) if ep.terrain.gap(ep.ball_pos[i], ep.ball.radius) > 0.05)
    kw = {
        "pos": ep.ball_pos[t],
        "vel": ep.ball_vel[t],
        "spin": ep.ball_spin[t],
        "frames": 20,
        "cam_pos": ep.cam_pos[t],
    }
    on = simulate_from_state(obs_cfg, ep.terrain, ep.ball, **kw)
    off = simulate_from_state(sim_cfg, ep.terrain, ep.ball, **kw)
    for f in PHYSICS:
        np.testing.assert_array_equal(getattr(on, f), getattr(off, f), err_msg=f)
    assert on.meta["obstacles"] == ep.meta["obstacles"] and "obstacles" not in off.meta
    np.testing.assert_array_equal(on.occluded, ep.occluded[t : t + 20])
    assert not off.occluded.any()


# ------------------------------------------------- placement


def test_placement_belongs_to_the_terrain_seed(obs_cfg):
    a = obstacles.sample_obstacles(obs_cfg, sample_terrain(obs_cfg, 3))
    b = obstacles.sample_obstacles(obs_cfg, sample_terrain(obs_cfg, 3))
    c = obstacles.sample_obstacles(obs_cfg, sample_terrain(obs_cfg, 4))
    assert a.to_json() == b.to_json() != c.to_json()
    one = simulate_episode(obs_cfg, 3, 0, render=False)
    other_ball = simulate_episode(obs_cfg, 3, 1, render=False)
    assert one.meta["obstacles"] == other_ball.meta["obstacles"] == a.to_json()


def test_posts_respect_count_width_gap_and_walls(obs_cfg):
    oc, tc = obs_cfg["obstacles"], obs_cfg["terrain"]
    inner = tc["length"] / 2 - tc["wall_width"]
    counts = set()
    for ts in range(40):
        terrain = sample_terrain(obs_cfg, ts)
        o = obstacles.sample_obstacles(obs_cfg, terrain)
        counts.add(len(o.x_lo))
        width = o.x_hi - o.x_lo
        assert oc["count"][0] <= len(o.x_lo) <= oc["count"][1]
        assert (width >= oc["width"][0]).all() and (width <= oc["width"][1]).all()
        assert (o.x_lo[1:] - o.x_hi[:-1] >= oc["gap"] - 1e-9).all()  # sorted, and gap apart
        assert o.x_lo[0] >= -inner + oc["gap"] - 1e-9  # a full gap from each wall zone, too
        assert o.x_hi[-1] <= inner - oc["gap"] + 1e-9
        assert o.z_lo < terrain.heights.min() and o.z_hi > terrain.heights.max()
    assert len(counts) > 1  # the count is drawn, not fixed


def test_bad_obstacle_configs_raise(sim_cfg, obs_cfg):
    terrain = sample_terrain(sim_cfg, 0)
    good = obs_cfg["obstacles"]
    bad = [
        ({**good, "colour": 1}, "unknown"),
        ({k: v for k, v in good.items() if k != "gap"}, "missing"),
        ({**good, "count": [3, 2]}, "count"),
        ({**good, "count": [-1, 2]}, "count"),
        ({**good, "width": [0.0, 1.0]}, "width"),
        ({**good, "width": [1.2, 1.0]}, "width"),
        ({**good, "gap": -0.5}, "gap"),
        ({**good, "gap": float("nan")}, "gap"),
        ({**good, "count": 4}, "count"),
        ({**good, "width": ["a", 1.0]}, "width"),
        ({**good, "width": [0.8, float("inf")]}, "width"),
        ({**good, "count": [40, 40]}, "too short"),
    ]
    for block, match in bad:
        with pytest.raises(ValueError, match=match):
            obstacles.sample_obstacles(with_obstacles(sim_cfg, **block), terrain)
    hand_built = Terrain(x=terrain.x, heights=terrain.heights)  # no seed in params
    with pytest.raises(ValueError, match="seed"):
        obstacles.sample_obstacles(obs_cfg, hand_built)


def test_posts_grow_with_how_high_the_ball_can_go(obs_cfg):
    terrain = sample_terrain(obs_cfg, 3)
    base = obstacles.sample_obstacles(obs_cfg, terrain)
    assert base.z_hi - terrain.heights.max() >= obstacles.POST_MARGIN_M
    high = copy.deepcopy(obs_cfg)
    high["ball"]["drop"] = [0.0, 20.0]
    tall = obstacles.sample_obstacles(high, terrain)
    view = obs_cfg["camera"]["view_height"]
    assert tall.z_hi - terrain.heights.max() > 20.0 + view  # start height plus a whole view
    np.testing.assert_array_equal(tall.x_lo, base.x_lo)  # placement doesn't depend on height


def test_a_from_state_start_outside_the_posts_fails_before_simulating(obs_cfg, monkeypatch):
    from jepa_world.sim import episode as episode_mod

    ep = simulate_episode(obs_cfg, 2, 0, render=False)

    def no_physics(*_):
        raise AssertionError("simulated before checking the start camera")

    monkeypatch.setattr(episode_mod.mujoco, "mj_step", no_physics)
    far = ep.cam_pos[0] + np.array([0.0, 100.0])  # a view 100 m above the terrain
    with pytest.raises(ValueError, match="z-span"):
        simulate_from_state(
            obs_cfg, ep.terrain, ep.ball, ep.ball_pos[0], ep.ball_vel[0], 0.0, 5, cam_pos=far
        )


def test_posts_need_the_camera_clearly_in_front_of_them(sim_cfg, obs_cfg):
    terrain, ball = sample_terrain(obs_cfg, 3), sample_ball(obs_cfg, 3, 0)
    obs = obstacles.sample_obstacles(obs_cfg, terrain)
    close = copy.deepcopy(obs_cfg)
    close["camera"]["distance"] = 1.4  # the camera would sit inside the post plane (y = -1.5)
    with pytest.raises(ValueError, match="distance"):
        build_model(terrain, ball, close, obs)
    close_off = copy.deepcopy(sim_cfg)
    close_off["camera"]["distance"] = 1.4
    build_model(terrain, ball, close_off)  # without posts the old scene builds as before


# ------------------------------------------------- the occlusion rule (pure)


def test_occluded_only_when_the_whole_disc_is_behind_a_post(sim_cfg):
    o = posts((-0.5, 0.5))
    # Ball edge at 0.49 (inside), 0.51 (a sliver shows), clear of the post, and level with it.
    got = occluded_at(sim_cfg, [[0.0, 0.0], [0.29, 0.0], [0.31, 0.0], [1.0, 0.0]], 0.2, o)
    assert got == [True, True, False, False]


def test_touching_posts_hide_the_ball_together_and_a_gap_shows_it(sim_cfg):
    assert occluded_at(sim_cfg, [0.0, 0.0], 0.2, posts((-0.5, 0.0), (0.0, 0.5))) == [True]
    assert occluded_at(sim_cfg, [0.0, 0.0], 0.2, posts((-0.5, -0.02), (0.02, 0.5))) == [False]


def test_out_of_view_is_never_occluded(sim_cfg):
    xz = [[10.0, 0.0]]  # far right of a camera at x = 0 (view 3.6 m wide)
    cam = np.zeros((1, 2))
    assert camera.out_of_view(camera.project(np.array(xz), cam, sim_cfg), 0.2, sim_cfg).all()
    assert occluded_at(sim_cfg, xz, 0.2, posts((5.0, 15.0))) == [False]


def test_only_the_part_inside_the_image_has_to_be_hidden(sim_cfg):
    edge = sim_cfg["camera"]["view_height"] / 2  # the image spans x, z in [-edge, edge]
    # Disc straddling the right edge: x in [edge - 0.15, edge + 0.25]; the image shows up to edge.
    x = edge + 0.05
    assert occluded_at(sim_cfg, [x, 0.0], 0.2, posts((edge - 0.2, edge + 0.1))) == [True]
    assert occluded_at(sim_cfg, [x, 0.0], 0.2, posts((edge - 0.1, edge + 0.5))) == [False]
    # Centre 0.1 m below the image: only the chord at the bottom edge shows, half-width sqrt(0.03).
    half = np.sqrt(0.2**2 - 0.1**2)
    z = -edge - 0.1
    assert occluded_at(sim_cfg, [0.0, z], 0.2, posts((-half - 0.01, half + 0.01))) == [True]
    assert occluded_at(sim_cfg, [0.0, z], 0.2, posts((-half + 0.01, half + 0.01))) == [False]


def test_visible_width_is_the_uncovered_part_in_pixels(sim_cfg):
    k = camera.px_per_m(sim_cfg)
    xz, cam = np.array([[0.0, 0.0]]), np.zeros((1, 2))
    uv = camera.project(xz, cam, sim_cfg)
    w = obstacles.visible_width_px(uv, 0.2, cam, posts((-0.5, 0.1)), sim_cfg)
    np.testing.assert_allclose(w, [0.1 * k])
    gap = obstacles.visible_width_px(uv, 0.2, cam, posts((-0.5, -0.02), (0.02, 0.5)), sim_cfg)
    np.testing.assert_allclose(gap, [0.04 * k])  # only the slit between two posts shows
    np.testing.assert_allclose(obstacles.visible_width_px(uv, 0.2, cam, None, sim_cfg), [0.4 * k])


def test_no_obstacles_means_never_occluded(sim_cfg):
    xz, cam = np.zeros((5, 2)), np.zeros((5, 2))
    got = obstacles.occluded(camera.project(xz, cam, sim_cfg), 0.2, cam, None, sim_cfg)
    assert got.shape == (5,) and got.dtype == bool and not got.any()


@pytest.mark.parametrize("z", [(-1.0, 100.0), (-100.0, 1.0)], ids=["below", "above"])
def test_a_view_past_the_posts_z_span_raises(sim_cfg, z):
    # The 1-D rule assumes each post covers the full image height (view 3.6 m). If not, fail loudly.
    with pytest.raises(ValueError, match="z-span"):
        occluded_at(sim_cfg, [0.0, 0.0], 0.2, posts((-0.5, 0.5), z=z))


# ------------------------------------------------- in real episodes


def test_occlusion_happens_and_is_disjoint_from_out_of_view(obs_cfg):
    eps = [simulate_episode(obs_cfg, ts, 0, render=False) for ts in range(8)]  # 4 have occlusion
    occ = np.stack([ep.occluded for ep in eps])
    oov = np.stack([ep.out_of_view for ep in eps])
    assert not (occ & oov).any()
    assert 0.0 < occ.mean() < 0.5
    assert sum(bool(row.any()) for row in occ) >= 2


def test_posts_are_recorded_and_survive_a_save(obs_cfg, tmp_path):
    ep = simulate_episode(obs_cfg, 3, 0, render=False)
    rec = ep.meta["obstacles"]
    assert set(rec) == {"x_lo", "x_hi", "z_lo", "z_hi"}
    assert all(type(v) is float for v in rec["x_lo"] + rec["x_hi"])
    back = load_episode(save_episode(tmp_path / "ep.npz", ep))
    assert back.meta["obstacles"] == rec
    np.testing.assert_array_equal(back.occluded, ep.occluded)


def test_preview_counts_occlusion_runs(sim_cfg):
    ep = simulate_episode(sim_cfg, 3, 0, render=False)
    T = len(ep.t)
    flags = np.r_[np.zeros(4, bool), np.ones(6, bool), np.zeros(T - 10, bool)]
    stats = preview.episode_stats([dataclasses.replace(ep, occluded=flags), ep])
    assert stats["occluded_rate"] == pytest.approx(6 / T / 2)
    assert stats["episodes_with_occlusion"] == 1
    assert stats["occluded_run_frames"] == {"median": 6.0, "p90": 6.0, "max": 6}


# ------------------------------------------------- render tier: the flag agrees with the pixels


@pytest.fixture(scope="module")
def rendered(obs_cfg):
    """The first terrain seed whose episode has occluded frames and mostly visible ones."""
    for ts in range(60):
        ep = simulate_episode(obs_cfg, ts, 0, render=False)
        if ep.occluded.sum() >= 3 and (~ep.occluded & ~ep.out_of_view).sum() > 20:
            ep = simulate_episode(obs_cfg, ts, 0, render=True, segmentation=True)
            obs = obstacles.sample_obstacles(obs_cfg, ep.terrain)
            return ep, obs, build_model(ep.terrain, ep.ball, obs_cfg, obs)
    pytest.fail("no terrain seed in 0..59 has 3+ occluded frames")


@pytest.mark.render
def test_occluded_frames_show_no_ball_and_clearly_visible_ones_do(obs_cfg, rendered):
    ep, obs, m = rendered
    n_ball = (ep.seg == geom_id(m, "ball")).sum(axis=(1, 2))
    assert (n_ball[ep.occluded] == 0).all()
    visible = obstacles.visible_width_px(ep.ball_px, ep.ball.radius, ep.cam_pos, obs, obs_cfg)
    clear = ~ep.out_of_view & (visible >= 2.0)  # 2 px: rasterising a sliver may hit no pixel centre
    assert clear.sum() > 20 and (n_ball[clear] > 0).all()


@pytest.mark.render
def test_posts_render_where_they_project(obs_cfg, rendered):
    ep, obs, m = rendered
    S = obs_cfg["frame_size"]
    post_ids = np.flatnonzero(m.geom_bodyid == body_id(m, "obstacles"))
    is_post = np.isin(ep.seg, post_ids)  # (T, S, S)
    centres = np.arange(S) + 0.5
    zeros = np.zeros_like(obs.x_lo)
    for t in range(len(ep.t)):
        u_lo = camera.project(np.c_[obs.x_lo, zeros], ep.cam_pos[t], obs_cfg)[:, 0]
        u_hi = camera.project(np.c_[obs.x_hi, zeros], ep.cam_pos[t], obs_cfg)[:, 0]
        inside = ((centres[:, None] > u_lo + 0.5) & (centres[:, None] < u_hi - 0.5)).any(1)
        outside = ~((centres[:, None] > u_lo - 0.5) & (centres[:, None] < u_hi + 0.5)).any(1)
        assert is_post[t][:, inside].all(), f"frame {t}: a post column is not fully covered"
        assert not is_post[t][:, outside].any(), f"frame {t}: post pixels outside every post"
