"""tools/finish-lane.sh: removes a merged lane, and nothing that might still hold work.

One throwaway main clone per module, with one lane per scenario under .claude/worktrees/. A fake
`gh` reports which branch heads have a merged PR. CC_SOCKS_DIR points at a temp dir, so a "live
session" is a real child process plus a real unix socket named by its pid. This repo's own
worktrees are never touched. Design: tools/DESIGN.md.

The script gathers every problem, then refuses through one gate. So a single run on a lane with
several faults checks each detector, by its message, without paying for a run per fault.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "finish-lane.sh"

FAKE_GH = """#!/bin/sh
# `gh pr list --head B --state merged --json headRefOid --jq .[].headRefOid`: print B's merged heads
echo "$*" >> "$FAKE_GH_LOG"
if [ -n "$FAKE_GH_FAIL" ]; then echo "gh: could not connect" >&2; exit 1; fi
head=""; prev=""
for a in "$@"; do [ "$prev" = "--head" ] && head=$a; prev=$a; done
while read -r b oid; do [ "$b" = "$head" ] && echo "$oid"; done < "$FAKE_GH_MERGED"
exit 0
"""


class Lanes:
    """A main clone at <tmp>/main whose lanes live in .claude/worktrees/wt-*."""

    def __init__(self, tmp: Path, socks: Path):
        self.main, self.socks = tmp / "main", socks
        self.merged, self.gh_log = tmp / "gh-merged", tmp / "gh-log"
        self.merged.write_text("")
        (tmp / "bin").mkdir()
        (tmp / "bin" / "gh").write_text(FAKE_GH)
        (tmp / "bin" / "gh").chmod(0o755)
        self.env = {
            **os.environ,
            "PATH": f"{tmp / 'bin'}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_MERGED": str(self.merged),
            "FAKE_GH_LOG": str(self.gh_log),
            "CC_SOCKS_DIR": str(socks),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        }
        self.env.pop("FAKE_GH_FAIL", None)
        self.main.mkdir()
        self.git("init", "-q", "-b", "main")
        (self.main / ".gitignore").write_text(
            ".claude/worktrees/\n.venv/\n__pycache__/\n*.py[cod]\n*.npz\n"
        )
        self.git("add", ".gitignore")
        self.git("commit", "-q", "-m", "init")

    def git(self, *args: str, cwd: Path | None = None) -> str:
        out = subprocess.run(
            ["git", *args], cwd=cwd or self.main, env=self.env, check=True, capture_output=True
        )
        return out.stdout.decode().strip()

    def wt(self, name: str) -> Path:
        return self.main / ".claude" / "worktrees" / name

    def lane(self, name: str, commit: bool = True) -> Path:
        """A lane as EnterWorktree makes it, optionally with one commit."""
        self.git("worktree", "add", "-q", "-b", f"worktree-{name}", str(self.wt(name)), "main")
        if commit:
            (self.wt(name) / "pkg").mkdir(exist_ok=True)
            (self.wt(name) / "pkg" / f"{name}.py").write_text("x = 1\n")
            self.git("add", "pkg", cwd=self.wt(name))
            self.git("commit", "-q", "-m", f"work on {name}", cwd=self.wt(name))
        return self.wt(name)

    def merge(self, name: str, pulled: bool = True) -> None:
        """Its PR merges at the branch tip. `pulled`: the main clone has pulled that merge."""
        tip = self.git("rev-parse", f"worktree-{name}")
        with self.merged.open("a") as f:
            f.write(f"worktree-{name} {tip}\n")
        if pulled:
            self.git("merge", "-q", "--no-ff", "-m", f"Merge {name}", f"worktree-{name}")

    def finish(self, *args: str, cwd: Path | None = None, **env: str):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            cwd=cwd or self.main,
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=30,
        )

    def exists(self, name: str) -> tuple[bool, bool]:
        """(worktree registered and on disk, branch exists)."""
        listed = f"worktree {self.wt(name)}" in self.git("worktree", "list", "--porcelain").split(
            "\n"
        )
        return listed and self.wt(name).is_dir(), bool(
            self.git("branch", "--list", f"worktree-{name}")
        )

    @contextmanager
    def session(self, cwd: Path, alive: bool = True):
        """A Claude session as lane-lib sees one: a process with this cwd, and <pid>.sock."""
        proc = subprocess.Popen(["sleep", "30"] if alive else ["true"], cwd=cwd)
        if not alive:
            proc.wait()
        sock = socket.socket(socket.AF_UNIX)
        try:
            sock.bind(str(self.socks / f"{proc.pid}.sock"))
            yield proc.pid
        finally:
            proc.kill()
            proc.wait()
            sock.close()
            (self.socks / f"{proc.pid}.sock").unlink(missing_ok=True)


@pytest.fixture(scope="module")
def lanes(tmp_path_factory):
    # Unix socket paths are limited to ~104 bytes, and macOS tmp paths are longer than that.
    socks = Path(tempfile.mkdtemp(prefix="ccs", dir="/tmp"))
    yield Lanes(tmp_path_factory.mktemp("lanes").resolve(), socks)
    shutil.rmtree(socks, ignore_errors=True)


def output(r: subprocess.CompletedProcess) -> str:
    return r.stdout + r.stderr


def test_removes_a_merged_clean_lane(lanes):
    # Caches, a session in the main clone, and a dead session's leftover socket don't block it.
    wt = lanes.lane("wt-done")
    lanes.merge("wt-done")
    (wt / ".venv" / "bin").mkdir(parents=True)
    (wt / ".venv" / "bin" / "python").write_text("")
    (wt / "pkg" / "__pycache__").mkdir()
    (wt / "pkg" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"")
    with lanes.session(lanes.main), lanes.session(wt, alive=False):
        r = lanes.finish("wt-done")
    assert r.returncode == 0, output(r)
    assert lanes.exists("wt-done") == (False, False)
    assert not wt.exists()
    asked = "pr list --head worktree-wt-done --state merged --json headRefOid --jq .[].headRefOid"
    assert asked in lanes.gh_log.read_text().splitlines()


def test_refuses_a_lane_that_may_still_hold_work(lanes):
    wt = lanes.lane("wt-busy")
    lanes.merge("wt-busy")
    (wt / "pkg" / "wt-busy.py").write_text("x = 2\n")  # uncommitted
    (wt / "notes.md").write_text("draft\n")  # untracked
    (wt / "episode.npz").write_bytes(b"data")  # ignored, and `git worktree remove` would delete it
    (wt / ".venv").mkdir()  # ignored, but a cache
    lanes.git("switch", "-q", "-c", "side", cwd=wt)
    lanes.git("worktree", "lock", "--reason", "parked", str(wt))
    with lanes.session(wt / "pkg") as pid:
        r = lanes.finish("wt-busy")
    assert r.returncode == 1, output(r)
    for why in (
        "pkg/wt-busy.py",
        "notes.md",
        "episode.npz",
        "locked: parked",
        f"pid {pid}",
        "side",
    ):
        assert why in output(r), why
    assert ".venv" not in output(r)
    assert lanes.exists("wt-busy") == (True, True)


@pytest.mark.parametrize("case", ["fresh", "reused-name", "unpulled", "gh-down"])
def test_refuses_a_branch_without_a_merge_at_its_tip(lanes, case):
    name, env = f"wt-{case}", {}
    if case == "fresh":
        # 0 commits ahead of main, like a merged branch: only the PR tells them apart.
        lanes.lane(name, commit=False)
        why = "no merged PR"
    elif case == "reused-name":
        # An old PR from a branch of this name merged, at another commit.
        lanes.lane(name, commit=False)
        with lanes.merged.open("a") as f:
            f.write(f"worktree-{name} {'0' * 40}\n")
        why = "no merged PR"
    elif case == "unpulled":
        lanes.lane(name)
        lanes.merge(name, pulled=False)
        why = "git pull --ff-only"
    else:
        lanes.lane(name)
        lanes.merge(name)
        env, why = {"FAKE_GH_FAIL": "1"}, "gh failed"
    r = lanes.finish(name, **env)
    assert r.returncode == 1, output(r)
    assert why in output(r)
    assert lanes.exists(name) == (True, True)


def test_deletes_the_branch_when_the_worktree_is_already_gone(lanes):
    lanes.lane("wt-half")
    lanes.merge("wt-half")
    lanes.git("worktree", "remove", str(lanes.wt("wt-half")))
    r = lanes.finish("wt-half")
    assert r.returncode == 0, output(r)
    assert lanes.exists("wt-half") == (False, False)


def test_refuses_to_run_from_inside_a_worktree(lanes):
    lanes.lane("wt-target")
    lanes.merge("wt-target")
    here = lanes.lane("wt-here", commit=False)
    r = lanes.finish("wt-target", cwd=here)
    assert r.returncode == 1
    assert "main clone" in output(r)
    assert lanes.exists("wt-target") == (True, True)


def test_a_name_with_neither_worktree_nor_branch_is_an_error(lanes):
    r = lanes.finish("wt-nope")
    assert r.returncode == 1
    assert "wt-nope" in output(r)


def test_rejects_anything_but_one_lane_name(lanes):
    lanes.lane("wt-kept")
    lanes.merge("wt-kept")
    for args in ([], ["--force"], ["wt-kept", "--force"], ["-f", "wt-kept"], ["wt-kept", "wt-kept"],
                 ["../wt-kept"], ["wt-kept/../wt-kept"], ["kept"], ["wt-"],
                 [".claude/worktrees/wt-kept"]):  # fmt: skip
        r = lanes.finish(*args)
        assert r.returncode == 2, (args, output(r))
    assert lanes.exists("wt-kept") == (True, True)


def test_project_settings_allow_the_script_and_no_raw_removal():
    allow = json.loads((REPO / ".claude" / "settings.json").read_text())["permissions"]["allow"]
    assert "Bash(tools/finish-lane.sh *)" in allow
    raw = [a for a in allow if a.startswith(("Bash(git worktree remove", "Bash(git branch -D"))]
    assert raw == []
