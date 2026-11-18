"""U-Net context/target encoder (SPEC §4.2-4.3)."""

from __future__ import annotations

import pytest
import torch
from tiny import spec_cfg, tiny_cfg

from jepa_world.models.unet import UNetEncoder, encoder_from_cfg


def test_levels_have_configured_shapes_tiny():
    cfg = tiny_cfg()
    enc = encoder_from_cfg(cfg)
    out = enc(torch.rand(2, 12, 16, 16))
    assert list(out) == ["L1", "L2", "L3"]
    assert out["L1"].shape == (2, 8, 16, 16)
    assert out["L2"].shape == (2, 16, 8, 8)
    assert out["L3"].shape == (2, 32, 2, 2)


def test_levels_have_spec_shapes():
    torch.manual_seed(0)
    enc = encoder_from_cfg(spec_cfg())
    out = enc(torch.rand(1, 12, 64, 64))
    assert out["L1"].shape == (1, 64, 64, 64)
    assert out["L2"].shape == (1, 128, 32, 32)
    assert out["L3"].shape == (1, 256, 8, 8)


def test_upto_stops_after_the_named_level():
    enc = encoder_from_cfg(tiny_cfg())
    x = torch.rand(2, 12, 16, 16)
    assert list(enc(x, upto="L1")) == ["L1"]
    assert list(enc(x, upto="L2")) == ["L1", "L2"]
    full = enc(x)
    torch.testing.assert_close(enc(x, upto="L2")["L2"], full["L2"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="L9"):
        enc(x, upto="L9")


def test_l2_to_l3_takes_two_stride2_steps():
    enc = encoder_from_cfg(tiny_cfg())
    strides = [m.stride for m in enc.stages[2].modules() if isinstance(m, torch.nn.Conv2d)]
    assert strides.count((2, 2)) == 2  # 8 -> 4 -> 2


@pytest.mark.parametrize(
    "levels, groups, match",
    [
        ([("L1", 8, 16), ("L2", 16, 6)], 2, "power of 2"),  # 16 / 6 is not an integer
        ([("L1", 8, 16), ("L2", 16, 12)], 2, "power of 2"),  # resolution must shrink by 2^n
        ([("L1", 8, 16), ("L2", 16, 8), ("L3", 32, 4), ("L4", 32, 8)], 2, "power of 2"),  # grows
        ([("L1", 6, 16)], 4, "groups"),  # 4 GroupNorm groups do not divide 6 channels
        ([("L1", 8, 16), ("L2", 16, 0)], 2, "power of 2"),  # size 0: no ZeroDivisionError
        ([("L1", 8, 8)], 2, "input resolution"),  # L1 must not downsample the raw input
    ],
)
def test_bad_level_specs_raise(levels, groups, match):
    with pytest.raises(ValueError, match=match):
        UNetEncoder(12, levels, input_size=16, downsample="stride2", norm_groups=groups)


def test_only_stride2_downsampling_is_accepted():
    with pytest.raises(ValueError, match="stride2"):
        UNetEncoder(12, [("L1", 8, 16)], input_size=16, downsample="pool", norm_groups=2)


def test_wrong_input_shape_raises():
    enc = encoder_from_cfg(tiny_cfg())
    with pytest.raises(ValueError, match="expected"):
        enc(torch.rand(2, 9, 16, 16))
    with pytest.raises(ValueError, match="expected"):
        enc(torch.rand(2, 12, 32, 32))


def test_encoder_has_no_buffers():
    # EMA covers parameters only. BatchNorm-style buffers would need explicit handling.
    assert list(encoder_from_cfg(tiny_cfg()).buffers()) == []


def test_continuing_after_a_level_matches_one_full_pass():
    enc = encoder_from_cfg(tiny_cfg())
    x = torch.rand(2, 12, 16, 16)
    full = enc(x)
    rest = enc(full["L1"], after="L1")
    assert list(rest) == ["L2", "L3"]
    for level in rest:
        torch.testing.assert_close(rest[level], full[level], rtol=0, atol=0)
    assert list(enc(full["L1"], upto="L2", after="L1")) == ["L2"]
    with pytest.raises(ValueError, match="expected"):
        enc(x, after="L1")  # x is the input stack, not L1's output
    with pytest.raises(ValueError, match="before"):
        enc(full["L2"], upto="L1", after="L2")
    assert enc(full["L3"], upto="L3", after="L3") == {}  # nothing left to compute
    assert enc(full["L1"], upto="L1", after="L1") == {}  # ...also mid-encoder (L2/L3 not run)
    assert enc(full["L3"], after="L3") == {}
