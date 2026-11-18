"""The PreToolUse lane guard (.claude/hooks/lane_guard.py), run against a throwaway repo."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).parents[1] / ".claude" / "hooks" / "lane_guard.py"
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def sh(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, env=GIT_ENV)


@pytest.fixture
def repo(tmp_path):
    """main clone + two worktrees (wt-a, wt-b), with the hook installed in the clone."""
    main = tmp_path / "main"
    main.mkdir()
    sh(main, "git", "init", "-q", "-b", "main")
    (main / "x.py").write_text("x = 1\n")
    (main / "y.py").write_text("y = 1\n")
    sh(main, "git", "add", ".")
    sh(main, "git", "commit", "-qm", "init")
    (main / ".claude" / "hooks").mkdir(parents=True)
    shutil.copy(HOOK, main / ".claude" / "hooks" / "lane_guard.py")
    wts = main / ".claude" / "worktrees"
    sh(main, "git", "worktree", "add", "-q", "-b", "wt-a", str(wts / "wt-a"))
    sh(main, "git", "worktree", "add", "-q", "-b", "wt-b", str(wts / "wt-b"))
    (wts / "wt-b" / "x.py").write_text("x = 2  # uncommitted work in lane b\n")
    return main, wts / "wt-a", wts / "wt-b"


def run_hook(main: Path, payload: dict) -> tuple[int, str, str]:
    r = subprocess.run(
        [sys.executable, str(main / ".claude" / "hooks" / "lane_guard.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "JEPA_ALLOW_MAIN"},
    )
    return r.returncode, r.stdout, r.stderr


def bash(main, cwd, command):
    return run_hook(
        main, {"tool_name": "Bash", "cwd": str(cwd), "tool_input": {"command": command}}
    )


def edit(main, cwd, path):
    return run_hook(
        main, {"tool_name": "Edit", "cwd": str(cwd), "tool_input": {"file_path": str(path)}}
    )


def test_commit_in_main_clone_is_blocked(repo):
    main, a, _ = repo
    rc, _, err = bash(main, main, "git commit -m wip")
    assert rc == 2 and "MAIN CLONE" in err
    assert bash(main, a, f"cd {main} && git add .")[0] == 2  # cd into the main clone
    assert bash(main, a, f"git -C {main} push origin main")[0] == 2  # git -C the main clone


def test_commit_in_worktree_and_reads_in_main_are_allowed(repo):
    main, a, _ = repo
    assert bash(main, a, "git add . && git commit -m wip && git push -u origin HEAD")[0] == 0
    assert bash(main, main, "git status && git fetch && git log -1")[0] == 0


def test_dry_runs_in_main_clone_are_allowed(repo):
    main, _, _ = repo
    assert bash(main, main, "git add -A --dry-run")[0] == 0
    assert bash(main, main, "git push -n origin main")[0] == 0


def test_explicit_escape_hatch(repo):
    main, _, _ = repo
    assert bash(main, main, "JEPA_ALLOW_MAIN=1 git commit -m bootstrap")[0] == 0


def test_edit_in_main_clone_warns_without_blocking(repo):
    main, _, _ = repo
    rc, out, _ = edit(main, main, main / "y.py")
    assert rc == 0
    ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
    assert "MAIN CLONE" in ctx


def test_edit_of_file_another_lane_holds_warns_collision(repo):
    main, a, _ = repo
    rc, out, _ = edit(main, a, a / "x.py")
    assert rc == 0
    payload = json.loads(out)
    assert "COLLISION" in payload["hookSpecificOutput"]["additionalContext"]
    assert "wt-b" in payload["systemMessage"]


def test_edit_of_free_file_in_worktree_is_silent(repo):
    main, a, _ = repo
    assert edit(main, a, a / "y.py") == (0, "", "")


def test_files_outside_the_repo_are_ignored(repo, tmp_path):
    main, a, _ = repo
    assert edit(main, a, tmp_path / "elsewhere.txt") == (0, "", "")
