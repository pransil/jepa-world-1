"""Shared infrastructure that keeps parallel lanes from clobbering each other."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest
import torch

from jepa_world.utils import config, locks, paths, repro
from jepa_world.utils.device import pick_device

# ---------------------------------------------------------------- artifacts root


def test_tests_never_use_the_shared_artifact_root(tmp_path):
    assert paths.artifacts_root().is_relative_to(tmp_path)
    assert not paths.artifacts_root().is_relative_to(paths.DEFAULT_ROOT)


# ---------------------------------------------------------------- locks


def test_lock_excludes_a_second_holder_then_frees():
    path = paths.artifacts_root() / "locks" / "x.lock"
    with locks.file_lock(path, label="first"):
        assert locks.is_locked(path)
        assert locks.read_holder(path)["label"] == "first"
        with pytest.raises(locks.LockTimeout):
            with locks.file_lock(path, timeout=0):
                pass
    assert not locks.is_locked(path)
    with locks.file_lock(path, timeout=0):  # free again
        pass


def test_waiting_acquirer_gets_lock_when_released():
    path = paths.artifacts_root() / "locks" / "q.lock"
    order = []
    held = threading.Event()

    def first():
        with locks.file_lock(path):
            held.set()
            time.sleep(0.3)
            order.append("first-done")

    t = threading.Thread(target=first)
    t.start()
    held.wait(2)
    with locks.file_lock(path, timeout=5, poll=0.05):
        order.append("second-in")
    t.join()
    assert order == ["first-done", "second-in"]


def test_lock_is_released_when_holder_process_dies():
    """flock dies with its process, so a killed training job leaves no stale lock."""
    path = paths.artifacts_root() / "locks" / "crash.lock"
    code = textwrap.dedent(f"""
        import os, sys, time
        from jepa_world.utils.locks import file_lock
        with file_lock({str(path)!r}):
            print("held", flush=True)
            time.sleep(60)
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        assert locks.is_locked(path)
        proc.kill()
        proc.wait(5)
        assert not locks.is_locked(path)
    finally:
        proc.kill()


def test_gpu_lock_can_be_disabled_for_quick_checks():
    with locks.gpu_lock("a"):
        with locks.gpu_lock("b", enabled=False):
            pass


# ---------------------------------------------------------------- datasets and runs


def test_config_hash_ignores_key_order_and_changes_with_values():
    assert paths.config_hash({"a": 1, "b": 2}) == paths.config_hash({"b": 2, "a": 1})
    assert paths.config_hash({"a": 1}) != paths.config_hash({"a": 2})


def test_dataset_is_built_once_then_reused():
    calls = []

    def build(d: Path):
        calls.append(d)
        (d / "x.npy").write_bytes(b"data")

    cfg = {"generator_version": 1, "seed": 3}
    first = paths.build_dataset_once(cfg, build)
    second = paths.build_dataset_once(cfg, build)
    assert first == second and len(calls) == 1
    assert (first / "x.npy").read_bytes() == b"data"
    assert (first / "config.json").exists()


def test_failed_build_leaves_nothing_behind():
    def build(d: Path):
        (d / "partial").write_text("half")
        raise RuntimeError("generator crashed")

    cfg = {"generator_version": 1, "seed": 4}
    with pytest.raises(RuntimeError):
        paths.build_dataset_once(cfg, build)
    assert not paths.dataset_dir(cfg).exists()
    assert not list((paths.artifacts_root() / "datasets").glob(".staging-*"))


def test_concurrent_builders_of_same_config_build_once():
    calls = []

    def build(d: Path):
        calls.append(1)
        time.sleep(0.2)

    cfg = {"generator_version": 1, "seed": 5}
    threads = [
        threading.Thread(target=paths.build_dataset_once, args=(cfg, build)) for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1


def test_write_new_writes_once_and_never_replaces_a_file(tmp_path):
    target = tmp_path / "sub" / "page.html"
    paths.write_new(target, "first \u2014 \u00d7")
    assert target.read_text(encoding="utf-8") == "first \u2014 \u00d7"  # UTF-8, whatever the locale
    with pytest.raises(FileExistsError):
        paths.write_new(target, "second")
    assert target.read_text(encoding="utf-8") == "first \u2014 \u00d7"
    assert sorted(p.name for p in target.parent.iterdir()) == ["page.html"]  # no temp left over


def test_run_dirs_are_unique():
    a, b = paths.new_run_dir("exp"), paths.new_run_dir("exp")
    assert a != b and a.is_dir() and b.is_dir()


# ---------------------------------------------------------------- config


def test_config_inherits_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("train: {lr: 0.001, batch_size: 32}\nseed: 0\n")
    (tmp_path / "exp.yaml").write_text("base: base.yaml\ntrain: {batch_size: 64}\n")
    cfg = config.load_config(tmp_path / "exp.yaml", ["train.lr=3e-4", "seed=7"])
    assert cfg == {"train": {"lr": 3e-4, "batch_size": 64}, "seed": 7}


def test_exponent_floats_without_decimal_point_are_floats(tmp_path):
    """PyYAML alone reads `lr: 3e-4` as the string '3e-4'."""
    (tmp_path / "c.yaml").write_text("train: {lr: 3e-4, wd: 1E+2, n: 10, tag: v1e5}\n")
    train = config.load_config(tmp_path / "c.yaml")["train"]
    assert train == {"lr": 3e-4, "wd": 100.0, "n": 10, "tag": "v1e5"}
    assert isinstance(train["lr"], float) and isinstance(train["n"], int)


def test_config_override_typo_fails_loudly(tmp_path):
    (tmp_path / "c.yaml").write_text("train: {lr: 0.001}\n")
    with pytest.raises(KeyError):
        config.load_config(tmp_path / "c.yaml", ["train.lrr=1"])


def test_repo_base_config_loads():
    cfg = config.load_config(Path(__file__).parents[1] / "configs" / "base.yaml")
    assert cfg["model"]["horizons"] == [1, 2, 3, 4, 5, 10, 20]
    assert set(cfg["model"]["head_level"]) == set(cfg["model"]["horizons"])


# ---------------------------------------------------------------- reproducibility and device


def test_seed_everything_is_deterministic():
    repro.seed_everything(123)
    a = torch.randn(4)
    repro.seed_everything(123)
    assert torch.equal(a, torch.randn(4))


def test_run_metadata_records_code_and_versions():
    meta = repro.run_metadata()
    assert set(meta["git"]) == {"sha", "branch", "dirty"}
    assert meta["libraries"]["torch"] and torch.__version__.startswith(meta["libraries"]["torch"])


def test_mps_fallback_env_is_set_by_package_import():
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_pick_device_cpu_and_auto():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type in {"cpu", "mps"}


@pytest.mark.mps
def test_mps_tensor_roundtrip():
    x = torch.arange(6.0, device="mps").reshape(2, 3)
    assert torch.equal((x * 2).cpu(), torch.arange(6.0).reshape(2, 3) * 2)
