"""Test tiers and guards. See CLAUDE.md "Testing".

Tiers (markers):  default = fast (CPU, tiny tensors, <~1s each)  |  slow  |  render  |  mps
./test runs only the fast tier. Anything in it that takes > FAST_BUDGET_S is listed at the end of
the run, so the tier stays fast because it can be seen, not by hoping.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

FAST_BUDGET_S = 2.0
_over_budget: list[tuple[str, float]] = []


def pytest_configure(config):
    # Small CPU tensors gain nothing from many threads, and parallel agents' test runs contend.
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def _gl_available() -> bool:
    # macOS: MuJoCo uses CGL, which needs no display. Linux: only if a headless backend is chosen.
    if sys.platform == "darwin":
        return True
    return os.environ.get("MUJOCO_GL") in {"egl", "osmesa"}


def pytest_collection_modifyitems(config, items):
    skip_mps = pytest.mark.skip(reason="MPS not available on this machine")
    skip_gl = pytest.mark.skip(reason="no OpenGL backend (set MUJOCO_GL=egl or osmesa on Linux)")
    has_mps = torch.backends.mps.is_available()
    has_gl = _gl_available()
    for item in items:
        if "mps" in item.keywords and not has_mps:
            item.add_marker(skip_mps)
        if "render" in item.keywords and not has_gl:
            item.add_marker(skip_gl)


@pytest.fixture(autouse=True)
def _isolated_artifacts(tmp_path, monkeypatch):
    """Tests never touch the shared artifact root that real runs and other lanes use."""
    monkeypatch.setenv("JEPA_ARTIFACTS", str(tmp_path / "artifacts"))


def pytest_runtest_logreport(report):
    if report.when == "call" and report.duration > FAST_BUDGET_S:
        if not ({"slow", "render", "mps"} & set(report.keywords)):
            _over_budget.append((report.nodeid, report.duration))


def pytest_terminal_summary(terminalreporter):
    if _over_budget:
        terminalreporter.section(
            f"FAST-TIER BUDGET: {len(_over_budget)} test(s) over {FAST_BUDGET_S}s"
        )
        for nodeid, d in sorted(_over_budget, key=lambda x: -x[1]):
            terminalreporter.write_line(
                f"  {d:6.2f}s  {nodeid}   -> shrink it, or mark it @pytest.mark.slow"
            )
