---
name: experiment
description: Run a research experiment in a structured, reproducible way. Hypothesis and deciding metric first, a committed config, a run under the GPU lock in a unique run dir with git sha and versions, comparison against the persistence and physics baselines, and a one-file note in experiments/. Use when asked to train, run, tune, ablate, benchmark, or "try" something and report results.
argument-hint: the question, e.g. "does m=0.999 beat 0.996 on k=20 position error?"
---

# /experiment: structured research runs

**Question:** $ARGUMENTS

An experiment is done when someone can read its note, re-run it from the recorded commit and
config, and get the same answer. Code changes needed for the experiment go through `/builder`
first. This skill runs **committed** code.

## Step 1: Hypothesis and the deciding number

Write down, before running anything:

- **Hypothesis**, in one sentence.
- **Deciding metric**, plus split and horizon. For example: "goal-head position error at k=20 on
  the *test* terrain seeds, as a ratio to persistence". Headline numbers come from held-out
  terrain seeds only, and dev-set numbers are labelled dev (SPEC §7.3).
- **What would change your mind**: the outcome that refutes the hypothesis.
- **Baselines** reported alongside: persistence, and the physics baseline (SPEC §7.1).
- **Budget**: steps or wall-clock, and seeds (≥ 2 seeds before claiming a difference).

If the hypothesis needs a SPEC §9 `[OPEN]` item changed, that is the user's decision: ask first.

## Step 2: Config and code state

1. Work in a worktree (`EnterWorktree wt-exp-<slug>`), not the main clone.
2. Write `configs/<slug>.yaml` with `base: base.yaml`, containing only the keys under test.
   Commit it.
3. **The tree must be clean** (`git status --porcelain` is empty), so the run's recorded sha *is*
   the code that ran. If the user explicitly wants a dirty run, record `dirty: true` in the note
   and say so in the summary.

## Step 3: Run

```bash
tools/gpu-status.sh                     # see the queue
uv run python -m jepa_world.train --config configs/<slug>.yaml --name <slug> [--set k=v ...]
```

- Training takes `locks.gpu_lock` itself. If another job holds it, yours **waits** (the log says
  who it's waiting for). Don't disable the lock to jump the queue, and never benchmark without it.
- Long runs: start with `run_in_background`, and you'll be notified when the process exits. Don't
  poll in a loop.
- The run dir (`$JEPA_ARTIFACTS/runs/<utc>-<branch>-<slug>-<rand>/`) gets `config.yaml`,
  `meta.json` (git sha, dirty flag, library versions, device), `metrics.jsonl`, `checkpoints/`
  and `tensorboard/`. Check that they exist before trusting the run.
- Sweeps: one run per setting, each with its own `--name`. Keep one queued job per lane at a time,
  so other lanes aren't starved of the GPU.

## Step 4: Evaluate

`uv run python -m jepa_world.eval --run <run dir> --split test`. Report the deciding metric for
each seed, the mean, and the spread, next to both baselines. Check calibration if the claim
involves uncertainty (the fraction inside the 50% and 90% intervals).

Before writing a conclusion, check:
- Did training actually converge? Read `metrics.jsonl`, not just the last line.
- Was there collapse? Look at the embedding std per level.
- Does each loss term look sane?

A "better" number from a diverged or collapsed run is not a result.

## Step 5: Write it up

`experiments/<YYYY-MM-DD>-<slug>.md` uses the template in `experiments/README.md`: question,
hypothesis, config link, commit sha, run dir(s), results table with baselines, conclusion
(**supported / refuted / inconclusive**, with the number), and next step. Use one file per
experiment, and never edit another experiment's note.

Commit the note (plus the config) and open a PR into `main`, as in `/builder` Step I. Notes are
small, so self-merge on green CI.

## Output

Give the user the answer in two sentences (the deciding number versus the baselines, and the
verdict), then the link to the note and the run dirs.
