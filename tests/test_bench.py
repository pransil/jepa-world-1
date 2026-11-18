"""L1 head benchmark (SPEC §9.4, §11). The real numbers come from the MPS run, not from here."""

from __future__ import annotations

import warnings

import torch
from tiny import tiny_aux_cfg, tiny_cfg, tiny_far_cfg

from jepa_world import bench


def test_bench_reports_one_row_per_variant():
    result = bench.bench_heads(
        tiny_cfg(), torch.device("cpu"), variants=[(4, 3), (2, 1)], iters=1, warmup=0
    )
    rows = result["rows"]
    assert [(r["hidden"], r["kernel"]) for r in rows] == [(4, 3), (2, 1)]
    for r in rows:
        assert r["ms_heads_fwd_bwd"] > 0 and r["ms_train_step"] > 0
        assert r["head_params"] > 0 and r["batch_size"] == 2
        assert r["mps_driver_mb"] is None  # CPU run
    assert rows[0]["head_params"] > rows[1]["head_params"]
    assert result["fallbacks"] == []


def test_bench_works_for_any_data_source():
    # Timing needs shapes, not data: frames are always synthetic.
    result = bench.bench_heads(
        tiny_cfg("data.source=episodes"), torch.device("cpu"), [(4, 1)], iters=1, warmup=0
    )
    assert len(result["rows"]) == 1


def test_bench_with_far_stage_times_the_l1_heads_and_the_full_step():
    near = bench.bench_heads(tiny_cfg(), torch.device("cpu"), [(4, 3)], iters=1, warmup=0)
    far = bench.bench_heads(tiny_far_cfg(), torch.device("cpu"), [(4, 3)], iters=1, warmup=0)
    (rn,), (rf,) = near["rows"], far["rows"]
    assert rf["head_params"] == rn["head_params"]  # the variant under test: the L1 heads only
    assert rn["far_head_params"] == 0 < rf["far_head_params"]
    assert rf["total_trainable_params"] > rn["total_trainable_params"]
    assert rn["td_head_params"] == rf["td_head_params"] == 0  # top_down is off by default


def test_bench_reports_the_top_down_branches():
    cfg = tiny_far_cfg("model.top_down=true")
    (row,) = bench.bench_heads(cfg, torch.device("cpu"), [(4, 3)], iters=1, warmup=0)["rows"]
    off = bench.bench_heads(tiny_far_cfg(), torch.device("cpu"), [(4, 3)], iters=1, warmup=0)
    assert row["td_head_params"] > 0
    assert (
        row["total_trainable_params"]
        == off["rows"][0]["total_trainable_params"] + row["td_head_params"]
    )


def test_fallback_recorder_catches_the_mps_fallback_warning():
    with bench.record_mps_fallbacks() as got:
        warnings.warn(
            "The operator 'aten::foo' is not currently supported on the MPS backend and will "
            "fall back to run on the CPU.",
            UserWarning,
            stacklevel=1,
        )
        warnings.warn("unrelated", UserWarning, stacklevel=1)
    assert len(got) == 1 and "aten::foo" in got[0]


def test_bench_with_goal_and_pixel_stages_times_the_full_step():
    """The full step needs a with_gt batch long enough for the goal horizons (review finding)."""
    cfg = tiny_aux_cfg("model.goal_horizons=[1, 20]")  # longer than any model horizon
    (row,) = bench.bench_heads(cfg, torch.device("cpu"), [(4, 3)], iters=1, warmup=0)["rows"]
    assert row["ms_train_step"] > 0
