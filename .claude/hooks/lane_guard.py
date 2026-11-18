#!/usr/bin/env python3
"""PreToolUse guard for parallel lanes. See docs/process/LANES.md.

Bash  -> BLOCKS `git add` / `git commit` / `git push` run in the MAIN CLONE (exit 2, reason on
         stderr, which Claude sees). The main clone is for fetch / ff-only pull. Work happens in
         worktrees. Escape hatch, only when the user explicitly says so: prefix the command
         with JEPA_ALLOW_MAIN=1.
Write/Edit -> WARNS (never blocks):
         - when the file is in the main clone rather than a worktree;
         - when another worktree already holds the same file, uncommitted or committed but
           unmerged. Overwriting another lane's UNCOMMITTED change leaves no conflict marker
           and nothing in git to recover.

Why JSON additionalContext: for PreToolUse, plain stdout is NOT shown to the model. Only
hookSpecificOutput.additionalContext reaches Claude's context (hooks reference, "PreToolUse").

Why `cwd` from the input, not $CLAUDE_PROJECT_DIR: the latter stays at the main clone, while
`cwd` follows the session into its worktree (worktrees docs, "Hook paths don't follow the
worktree").

Fails OPEN: any unexpected error exits 0 silently. A guard that breaks every tool call gets
switched off, and that is worse than one that occasionally misses.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = "origin/main"
CACHE_TTL_S = 30
GUARDED_GIT = {"add", "commit", "push"}


def git(cwd: str | Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=10)
    return (
        r.stdout.rstrip() if r.returncode == 0 else ""
    )  # rstrip: porcelain lines start with a space


def repo_info(path: str | Path) -> tuple[str, str, bool] | None:
    """(toplevel, common_dir, is_main_clone) for the repo containing `path`, or None."""
    d = Path(path)
    if not d.is_dir():
        d = d.parent
    while not d.exists():
        d = d.parent
    top = git(d, "rev-parse", "--show-toplevel")
    if not top:
        return None
    gd = git(d, "rev-parse", "--path-format=absolute", "--git-dir")
    common = git(d, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return top, common, gd == common


OUR_COMMON = (repo_info(Path(__file__).resolve().parent) or ("", "", False))[1]


# ------------------------------------------------------------------------------------- Bash


def git_targets(command: str, cwd: str):
    """Yield (directory, subcommand, allowed) for each `git <sub>` in a shell command line.
    Follows `cd X &&` and `git -C X`. Good enough for agent-written commands, not a shell."""
    here = cwd
    for seg in re.split(r"&&|\|\||;|\||\n", command):
        try:
            words = shlex.split(seg)
        except ValueError:
            continue
        allowed = False
        while words and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]):
            allowed |= words[0] == "JEPA_ALLOW_MAIN=1"
            words = words[1:]
        if not words:
            continue
        if words[0] == "cd" and len(words) > 1:
            here = os.path.join(here, os.path.expanduser(words[1]))
            continue
        if Path(words[0]).name != "git":
            continue
        d, i = here, 1
        while i < len(words) and words[i].startswith("-"):
            if words[i] == "-C" and i + 1 < len(words):
                d = os.path.join(d, os.path.expanduser(words[i + 1]))
                i += 2
            elif words[i] in ("-c", "--git-dir", "--work-tree") and i + 1 < len(words):
                i += 2
            else:
                i += 1
        if i < len(words):
            dry = any(w in ("-n", "--dry-run") for w in words[i + 1 :])
            yield d, words[i], allowed or dry  # a dry run changes nothing


def check_bash(command: str, cwd: str) -> str | None:
    if os.environ.get("JEPA_ALLOW_MAIN") == "1":
        return None
    for d, sub, allowed in git_targets(command, cwd):
        if sub not in GUARDED_GIT or allowed:
            continue
        info = repo_info(d)
        if info and info[1] == OUR_COMMON and info[2]:
            return (
                f"lane-guard: `git {sub}` in the MAIN CLONE ({info[0]}) is blocked.\n"
                "Work happens in a worktree: EnterWorktree (name: wt-<topic>), then commit, "
                "push the branch, and open a PR. See docs/process/LANES.md.\n"
                "If the user has EXPLICITLY asked for this in the main clone, re-run with the "
                "prefix JEPA_ALLOW_MAIN=1."
            )
    return None


# ------------------------------------------------------------------------------- Write/Edit


def files_in_flight(main_top: str) -> list[tuple[str, str, str]]:
    """[(relpath, worktree_path, branch)] for every worktree's dirty + unmerged files. Cached."""
    key = re.sub(r"[^A-Za-z0-9]", "_", main_top)[-60:]
    cache = Path(tempfile.gettempdir()) / f"jepa-lane-guard-{key}.json"
    try:
        if time.time() - cache.stat().st_mtime < CACHE_TTL_S:
            return [tuple(x) for x in json.loads(cache.read_text())]
    except (OSError, ValueError):
        pass
    rows: list[tuple[str, str, str]] = []
    wt, br = None, ""
    for line in git(main_top, "worktree", "list", "--porcelain").splitlines() + [""]:
        if line.startswith("worktree "):
            wt, br = line[9:], ""
        elif line.startswith("branch "):
            br = line[7:].removeprefix("refs/heads/")
        elif not line and wt:
            files = set()
            for s in git(wt, "status", "--porcelain").splitlines():
                files.add(s[3:].split(" -> ")[-1].strip('"'))
            if br:
                files.update(git(wt, "diff", "--name-only", f"{BASE}...{br}").splitlines())
            rows += [(f, wt, br) for f in files if f]
            wt = None
    try:
        cache.write_text(json.dumps(rows))
    except OSError:
        pass
    return rows


def check_edit(file_path: str) -> tuple[str, str] | None:
    """(note for Claude, short message for the user, empty if no collision) or None."""
    info = repo_info(file_path)
    if not info or info[1] != OUR_COMMON:
        return None
    top, common, is_main = info
    rel = os.path.relpath(os.path.realpath(file_path), os.path.realpath(top))
    notes = []
    if is_main:
        notes.append(
            f"You are editing {rel} in the MAIN CLONE, not a worktree. Commits here are blocked. "
            "Unless the user asked for a main-clone edit, stop and move this work into a "
            "worktree (EnterWorktree)."
        )
    main_top = str(Path(common).parent)
    others = sorted(
        {
            f"{Path(wt).name} ({br or 'detached'})"
            for f, wt, br in files_in_flight(main_top)
            if f == rel and os.path.realpath(wt) != os.path.realpath(top)
        }
    )
    if others:
        notes.append(
            f"COLLISION: {rel} is already being changed in: {', '.join(others)}. Their change may "
            "be uncommitted, and writing over it leaves nothing in git to recover. Check with "
            f"`tools/lane-owners.sh {rel}`, then coordinate with that lane or ask the user "
            "before editing."
        )
    if not notes:
        return None
    user_msg = f"lane-guard: {rel} is also being changed in {', '.join(others)}" if others else ""
    return "\n".join(notes), user_msg


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool = data.get("tool_name", "")
        ti = data.get("tool_input", {}) or {}
        cwd = data.get("cwd") or os.getcwd()
        if tool == "Bash":
            reason = check_bash(ti.get("command", ""), cwd)
            if reason:
                print(reason, file=sys.stderr)
                return 2
        elif tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            path = ti.get("file_path") or ti.get("notebook_path") or ""
            found = check_edit(path) if os.path.isabs(path) else None
            if found:
                note, user_msg = found
                out = {
                    "hookSpecificOutput": {"hookEventName": "PreToolUse", "additionalContext": note}
                }
                if user_msg:
                    out["systemMessage"] = user_msg
                print(json.dumps(out))
    except Exception:  # fail open, see docstring
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
