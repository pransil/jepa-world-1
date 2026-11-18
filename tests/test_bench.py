"""L1 head benchmark (SPEC §9.4, §11). The real numbers come from the MPS run, not from here."""

from __future__ import annotations

import warnings

import torch
from tiny import tiny_cfg

from jepa_world import bench


def test_bench_reports_one_row_per_variant():
    rows = bench.bench_heads(
        tiny_cfg(), torch.device("cpu"), variants=[(4, 3), (2, 1)], iters=1, warmup=0
    )
    assert [(r["hidden"], r["kernel"]) for r in rows] == [(4, 3), (2, 1)]
    for r in rows:
        assert r["ms_heads_fwd_bwd"] > 0 and r["ms_train_step"] > 0
        assert r["head_params"] > 0 and r["batch_size"] == 2
        assert r["fallbacks"] == []
    assert rows[0]["head_params"] > rows[1]["head_params"]


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
