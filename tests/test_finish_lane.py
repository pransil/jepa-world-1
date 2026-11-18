"""tools/finish-lane.sh: removes a merged lane, and nothing that might still hold work.

One throwaway main clone per module, with one lane per scenario under .claude/worktrees/. A fake
`gh` reports which branch heads have a merged PR. A "running process" is a real `sleep` child, and
a "Claude session" is a pid with a real unix socket in CC_SOCKS_DIR.
This repo's own worktrees are never touched. Design: tools/DESIGN.md.

The script gathers every problem from its fast checks, then refuses through one gate. So a single
run on a lane with several faults checks each detector, by its message, without a run per fault.
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

# The parts of this repo's .gitignore that matter here: caches, local settings, and a data file.
GITIGNORE = """.claude/worktrees/
.claude/settings.local.json
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
*.npz
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
        (self.main / ".gitignore").write_text(GITIGNORE)
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
        """A lane as EnterWorktree makes it, optionally with one commit (pkg/<name>.py)."""
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
        branch = bool(self.git("branch", "--list", f"worktree-{name}"))
        return listed and self.wt(name).is_dir(), branch


@contextmanager
def process_in(cwd: Path, argv: list[str] | None = None, executable: str | None = None):
    """A process of this user whose working directory is `cwd`, as a session or a job would be."""
    proc = subprocess.Popen(argv or ["sleep", "30"], cwd=cwd, executable=executable)
    try:
        yield proc
    finally:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def lanes(tmp_path_factory):
    # Unix socket paths are limited to ~104 bytes, and macOS tmp paths are longer than that.
    socks = Path(tempfile.mkdtemp(prefix="ccs", dir="/tmp"))
    yield Lanes(tmp_path_factory.mktemp("lanes").resolve(), socks)
    shutil.rmtree(socks, ignore_errors=True)


@contextmanager
def session(lanes: Lanes, pid: int):
    """Register `pid` as a live Claude Code session, the way /tmp/cc-socks does."""
    sock = socket.socket(socket.AF_UNIX)
    path = lanes.socks / f"{pid}.sock"
    try:
        sock.bind(str(path))
        yield
    finally:
        sock.close()
        path.unlink(missing_ok=True)


def caffeinate_in(cwd: Path):
    """A process named caffeinate, child of this test process. Claude Code starts
    `caffeinate -i -t 300` in the session's cwd to keep the Mac awake. macOS has the real one, and
    SIGKILLs copies of Apple binaries. Linux has none, so there it is sleep exec'd through a
    symlink named caffeinate: the kernel names the process after the path, and argv[0] stays
    "sleep" for multicall coreutils."""
    if real := shutil.which("caffeinate"):
        return process_in(cwd, [real, "-i", "-t", "30"])
    link = Path(tempfile.mkdtemp(prefix="caff", dir="/tmp")) / "caffeinate"
    link.symlink_to(shutil.which("sleep"))
    return process_in(cwd, ["sleep", "30"], executable=str(link))


def output(r: subprocess.CompletedProcess) -> str:
    return r.stdout + r.stderr


def listed(r: subprocess.CompletedProcess, label: str) -> set[str]:
    """The paths a refusal lists after `label:`, e.g. 'uncommitted or untracked: a b'."""
    line = next(ln for ln in output(r).splitlines() if f"{label}:" in ln)
    return set(line.split(f"{label}:", 1)[1].split())


def test_refuses_a_lane_that_may_still_hold_work(lanes):
    wt = lanes.lane("wt-busy")
    lanes.merge("wt-busy")
    lanes.git("mv", "pkg/wt-busy.py", "pkg/moved.py", cwd=wt)  # staged rename: a 2-field -z entry
    (wt / "notes.md").write_text("draft\n")  # untracked
    (wt / "episode.npz").write_bytes(b"data")  # ignored, and `git worktree remove` would delete it
    (wt / ".venv").mkdir()  # ignored, but disposable
    lanes.git("switch", "-q", "-c", "side", cwd=wt)
    lanes.git("worktree", "lock", "--reason", "parked", str(wt))
    # A user config that hides untracked files from `git status` must not hide them here.
    hide = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "status.showUntrackedFiles"}
    r = lanes.finish("wt-busy", **hide, GIT_CONFIG_VALUE_0="no")
    assert r.returncode == 1, output(r)
    assert listed(r, "uncommitted or untracked") == {"pkg/moved.py", "notes.md"}
    assert listed(r, "ignored files that are not disposable") == {"episode.npz"}
    assert "wt-busy has side checked out" in output(r)
    assert "locked: parked" in output(r)
    assert lanes.exists("wt-busy") == (True, True)


def test_removes_a_merged_lane_once_no_process_runs_inside(lanes):
    wt = lanes.lane("wt-done")
    lanes.merge("wt-done")
    # Caches and the lane's own permission grants are disposable.
    (wt / ".venv" / "bin").mkdir(parents=True)
    (wt / ".venv" / "bin" / "python").write_text("")
    (wt / "pkg" / "__pycache__").mkdir()
    (wt / "pkg" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"")
    (wt / ".claude").mkdir()
    (wt / ".claude" / "settings.local.json").write_text("{}")
    # A process in the main clone is not inside the lane. A caffeinate inside blocks it like any
    # process, unless its parent is a live Claude session.
    with process_in(lanes.main), caffeinate_in(wt) as caff:
        with process_in(wt / "pkg") as job:
            r = lanes.finish("wt-done")
            assert r.returncode == 1, output(r)
            assert f"process {job.pid} " in output(r)
            assert f"process {caff.pid} " in output(r)  # no session registered yet
            assert lanes.exists("wt-done") == (True, True)
        assert caff.poll() is None
        with session(lanes, os.getpid()):  # this test process is caffeinate's parent
            r = lanes.finish("wt-done")
        assert caff.poll() is None, "caffeinate must have been running inside during that run"
    assert r.returncode == 0, output(r)
    assert lanes.exists("wt-done") == (False, False)
    assert not wt.exists()
    asked = "pr list --head worktree-wt-done --state merged --json headRefOid --jq .[].headRefOid"
    assert asked in lanes.gh_log.read_text().splitlines()


@pytest.mark.parametrize("case", ["fresh", "reused-name", "unpulled", "gh-down", "head-elsewhere"])
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
    elif case == "gh-down":
        lanes.lane(name)
        lanes.merge(name)
        env, why = {"FAKE_GH_FAIL": "1"}, "gh failed"
    else:
        # `main` has the merge, but the main clone's HEAD doesn't, so `git branch -d` would refuse
        # after the worktree was already gone.
        lanes.lane(name)
        lanes.merge(name)
        lanes.git("switch", "-q", "--detach", "HEAD~1")
        why = "git branch -d would refuse"
    try:
        r = lanes.finish(name, **env)
    finally:
        lanes.git("switch", "-q", "main")
    assert r.returncode == 1, output(r)
    assert why in output(r)
    assert lanes.exists(name) == (True, True)


# Stand-ins for lsof (macOS) and GNU find (Linux), the two process backends. "empty": the tool
# is missing or prints nothing. "garbled": every pid is listed, but with a cwd that isn't its own,
# as a parse error would. Either way the list can't be trusted to show what runs in the worktree.
BLIND_TOOL = {
    "empty": "#!/bin/sh\nexit 0\n",
    "garbled": """#!/bin/sh
for p in $(ps -eo pid=); do
  case "${0##*/}" in
    lsof) printf 'p%s\\nn/nowhere\\n' "$p" ;;
    find) printf '/proc/%s\\t/nowhere\\n' "$p" ;;
  esac
done
""",
}


@pytest.mark.parametrize("listing", ["empty", "garbled"])
def test_refuses_when_running_processes_cant_be_listed(lanes, tmp_path, listing):
    for tool in ("lsof", "find"):
        (tmp_path / tool).write_text(BLIND_TOOL[listing])
        (tmp_path / tool).chmod(0o755)
    name = f"wt-blind-{listing}"
    lanes.lane(name)
    lanes.merge(name)
    r = lanes.finish(name, PATH=f"{tmp_path}{os.pathsep}{lanes.env['PATH']}")
    assert r.returncode == 1, output(r)
    assert "can't list running processes" in output(r)
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
