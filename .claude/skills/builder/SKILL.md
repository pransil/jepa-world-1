---
name: builder
description: Disciplined build cycle for one chunk of this repo (a milestone, a module, a model component). Isolate in a worktree, outcome sentence, units and shapes, DESIGN.md, tests FIRST (red), code, /code-review and /verify, ./test, PR into main, self-merge on green CI, clean up. Use when asked to "build", "implement", or "use the builder" for anything beyond a one-line fix.
argument-hint: the chunk to build, e.g. "M3 context encoder + EMA target", with acceptance criteria if not in SPEC.md
---

# /builder: structured build cycle

**Chunk:** $ARGUMENTS

Follow the steps **in order**. Keep a todo list (TaskCreate/TaskUpdate) with one item per step, so
progress is visible to the user and to you after a context compaction. This skill orchestrates
existing tools (`tools/lane-owners.sh`, `./test`, `/code-review`, `/verify`, `gh`). Call them,
don't reimplement them.

If the chunk is genuinely ambiguous (which milestone, which files, what counts as done), ask
before Step 0. Otherwise proceed.

## Step 0: Isolate (worktree, claim, environment)

1. `git -C <main clone> fetch -q origin`, then `tools/lane-owners.sh <paths you expect to touch>`.
   If another lane holds one of those paths, **stop**: tell the user which lane, and ask who takes
   it. Do not "just be careful".
2. `EnterWorktree` with name `wt-<chunk-slug>`. This creates `.claude/worktrees/wt-<slug>` on
   branch `worktree-wt-<slug>`, branched from `origin/main`, and moves the session there. Never
   use `cd` instead: skills and subagents would keep running in the main clone.
3. Push the empty branch at once. **This is your claim**: `git push -u origin HEAD`.
4. `uv sync`. It is fast, because uv hard-links from its shared cache.

## Step A0: Say what must be true, in one sentence

Write the outcome as one testable sentence, tied to its source: *"SPEC §4.4 / M3: the target
encoder receives no gradient, and after each step its parameters equal m·θ_t + (1−m)·θ_c."*
Then list the individual promises, each phrased so a test can assert it. The milestone's
**Tests** line in SPEC §8 is the minimum set.

- **SPEC §9 `[OPEN]` items are the user's call** (SPEC §11). If the chunk needs one decided (camera
  mode, distance function, collapse guard, L1 head width, downsampling...), ask with
  AskUserQuestion, giving options plus your recommendation. Record the answer in
  `docs/decisions/<YYYY-MM-DD>-<slug>.md` (template in that dir's README) **in this PR**, and
  change the `[OPEN]` value in `configs/base.yaml` to match.
- Changing what SPEC.md says the system does is the user's call too. Ask first.

## Step A: Decide the units

List each function or class: name, signature, **tensor shapes in → out** (with batch dim),
dtype and device expectations, and whether it is pure or side-effecting (I/O, rendering, the
artifact root). Push side effects into thin wrappers so the core is testable on CPU with tiny
inputs.

**Reuse first:** `jepa_world/utils/` already has artifact paths and write-once datasets
(`paths`), the GPU lock (`locks`), config loading with overrides (`config`), seeding and run
metadata (`repro`), and device selection (`device`). Search before adding a helper.

## Step B: DESIGN.md (short, required)

Add or update `DESIGN.md` in the module's directory (e.g. `jepa_world/models/DESIGN.md`). Keep it
under about a page:

- **Purpose / non-goals**: what this chunk does, and what it deliberately doesn't.
- **Data flow and shapes**: a small table or diagram.
- **Key decisions**: each choice a reader would question, *with the alternative that lost and
  why*. Link any `docs/decisions/` file.
- **Seams**: what is pure, what touches MPS, MuJoCo or disk, and how the tests reach it.
- **Files**: `file | responsibility`.

The build narrative (what happened, findings, results) goes in the **PR body**, not here.

## Step C: Tests FIRST, and see them fail

**Write no implementation until the tests exist and fail for the right reason.** No later gate
can enforce this, so the builder does.

- Put them in `tests/test_<module>.py`. Default to the **fast tier**: CPU, tiny shapes (B=2, a
  few channels, 16×16 instead of 64×64 where the code is size-generic), a fixed seed, and
  well under 1s each. The full suite budget is ~20s. `conftest.py` lists any fast test over 2s.
- Prefer **invariant tests** over golden numbers. These are the ones this project needs:
  - output shapes match targets at every level and horizon;
  - no gradient reaches the target encoder, and targets are detached;
  - the EMA update matches the formula exactly;
  - the context never includes a frame at or after t+k, and splits never share a terrain seed;
  - same seed gives an identical episode or batch;
  - the NLL is finite and the variance is positive;
  - every loss term is present in the logged dict;
  - one optimizer step on a fixed tiny batch lowers the loss.
- Mark a test `@pytest.mark.slow` only if it cannot be made fast (real training loops,
  full-length episodes). Use `render` for MuJoCo rendering and `mps` for MPS-specific behaviour.
  Every slow test should have a fast sibling that covers the same logic at a smaller size.
- Run `./test tests/test_<module>.py` and confirm the tests **fail**, and fail for the stated
  reason (an ImportError or a NotImplementedError naming the missing unit, not a typo). If a new
  test passes before any code exists, it asserts nothing. Fix the test.

## Step D: Write the code

- **Check APIs against the installed versions, not memory** (SPEC §2, §11). The installed
  MuJoCo is 4.x, and its API differs from older docs and tutorials. Use
  `uv run python -c "import mujoco; help(mujoco.Renderer)"`, or read the package source under
  `.venv/lib/python3.12/site-packages/`. The same goes for torch MPS op support.
- Match the surrounding style. Configs flow in from `configs/*.yaml` via `utils.config`, with no
  hard-coded hyperparameters in modules.
- Keep the JEPA core (`models/jepa.py`) independent of the goal head and pixel decoder (SPEC §10).

## Step E: Review gates (commit first, then review)

1. Commit in the worktree. Commit messages end with the repo's required trailer lines.
2. Run **`/code-review high`** on the branch, then **`/verify`**. **Require each to state its
   target (base ref plus changed-file count) and reject the result if those aren't your files.**
   A review that finds nothing because it looked at the wrong tree is indistinguishable from a
   clean review.

## Step F: Fix findings

Fix every correctness finding. Fix the rest unless deferring is explicitly justified, and list
each deferral in the PR body.

## Step G: Run the tests

`./test`, plus `./test all` if the chunk touches rendering, MPS paths, or anything with a slow
test. **Read the RESULT line**, and name every skipped tier and why in the PR body. A skipped
tier is not a pass.

## Step H: Iterate (max 3)

Fix the cause (not the test, unless the test is wrong) and go back to G. Red CI on the PR counts
as an iteration. After 3 failed rounds, **stop** and report the blocker to the user. Never open
or merge a PR with failing tests.

## Step I: PR into main, self-merge on green

1. `git fetch origin && git rebase origin/main` if `main` moved, re-run `./test`, then
   `git push --force-with-lease` (your own branch only).
2. `gh pr create --base main` with a body that **is the build record**:
   - one paragraph on what was built and where;
   - a link to DESIGN.md and any decision files;
   - review findings and how each was resolved, plus deferrals;
   - the final `./test` RESULT line(s) and any skipped tiers;
   - for a milestone, the SPEC §8 tests and where each one lives.
3. Push once, then wait for CI: `gh pr checks --watch`. Don't keep pushing while checks run.
4. Green: `gh pr merge --merge --delete-branch`. Red: go to Step H.

## Step J: Close out

- **Milestone finished?** `docs/milestones/M<n>.md` (template in that dir's README) goes in
  the PR **before** merging: what was built, which tests prove it, measured numbers (e.g. the L1
  head benchmark), and open issues.
- After the merge, clean up in this order (measured on PR #1). A worktree session can't run git
  against the main clone, and `ExitWorktree remove` refuses while the main clone's `main` lacks
  your commits.
  1. `ExitWorktree` with `action: "keep"`. The session is back in the main clone.
  2. `git pull --ff-only`, so the main clone's `main` now contains the merge.
  3. `tools/finish-lane.sh wt-<slug>`. It removes the worktree and the local branch only if all of these hold:
     - the PR merged at the branch's exact tip;
     - `main` has every commit from the branch;
     - the tree holds no uncommitted, untracked or non-cache ignored files;
     - the tree isn't locked;
     - no process is running inside it (Claude Code's own `caffeinate` doesn't count).

     Otherwise it removes nothing and lists why. If it refuses, leave the tree, and give the
     reasons in the output. Fix only causes that are yours, such as an unpulled `main`.
     Raw `git worktree remove` is denied in the user's settings, and this script is the
     allowed path (`tools/DESIGN.md`). Don't work around a refusal.

     Run it exactly as `tools/finish-lane.sh wt-<slug>`. The allow rule matches that literal
     prefix, so `./tools/...` or `bash tools/...` won't match. If the call itself is
     **denied** (a declined prompt, or the auto-mode classifier), don't retry it another way.
     Leave the tree, and give the user the line to run themselves, `! tools/finish-lane.sh wt-<slug>`.
     Mention that in auto mode they can also approve it under `/permissions` → Recently blocked.
- Never pass `discard_changes: true` or `--force` without the user's say-so.

## Output

Report the units built and where, the review findings and their resolutions, the final test
RESULT lines, the PR URL and merge state, and anything deferred or decided (with links to the
decision files).
