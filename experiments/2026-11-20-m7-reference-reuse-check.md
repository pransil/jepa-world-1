# Does the M7 reference still hold after m5's H (and A)? A 6,500-step bitwise re-run

- **Date:** 2026-11-20   **Lane/branch:** worktree-wt-exp-m7-refcheck
- **Commit:** the one this lane has after merging origin/main right before the run. Main already
  has m5's H PR 1 and A (#40, merged 2026-11-20); the sha goes under Results.
- **Config:** `configs/exp_headline_48k.yaml` with `--set train.steps=6500`, seed 0.
- **Reference:** `20261119T224253Z-worktree-wt-exp-headline-48k-new-headline48k-new-s0-0f2411`
  (commit 6fd7059; [note](2026-11-19-headline-48k-new-defaults.md)). It is the M7 reference's
  seed 0.
- **Asked for:** the orchestrator, for the user (2026-11-20). It is the first GPU job of the M7
  ablation slot, instead of re-running the reference (≈ 4 GPU-h).

## Question
M7-plan's reuse rule: an ablation may use the reference only if
`git diff <ref sha> <ablation sha> -- jepa_world/models jepa_world/train.py jepa_world/data configs/base.yaml`
is empty. Two of m5's changes land in those paths before the ablations run:
- **H PR 1** adds a `data.curriculum` key (default null), so `make_batches` can switch splits at
  step N.
- **A** (#40) adds two diagnostics, both **on** by default in `base.yaml`:
  - `train.copy_every: 10`: `relerr/copy/k`, a no-grad target pass every 10 completed steps;
  - `train.grad_norm_every: 500`: `gradnorm/enc/<term>/<level>`, one `autograd.grad` per loss term
    before the step's backward, every 500 completed steps.

The curriculum is off by default, and A's diagnostics are meant not to touch the update. The
ablations will run with A's diagnostics on, so this check runs them on too. **Does training at the
new commit repeat the reference bitwise?**

## Hypothesis and pass rule (pre-registered)
**H0: with the defaults, the new code trains exactly as 6fd7059 did.**

- **Pass:** `uv run python experiments/compare_metrics.py --run <re-run> --ref <0f2411>` exits 0,
  using main's version of the script at the check's commit. That means:
  - every one of steps 1–6,500 is present in the reference;
  - every key both runs log has an identical value at every step (NaN = NaN);
  - no non-diagnostic key is present on only one side.
- **New diagnostic keys:** main's script (A's version) exempts diagnostic keys by
  `train.DIAGNOSTIC_PREFIXES` (`relerr/copy/`, `gradnorm/`): a diagnostic key logged by only one
  run is reported and not counted. So A's new columns don't fail the check, but every shared key
  still must match bitwise.
  - (An earlier draft added an `--allow-new-keys` flag in this lane. It was dropped in favour of
    A's version, so the two lanes don't collide on the file.)
- **Fail:** any differing value, or a non-diagnostic key missing on either side. Then the M7
  ablations do not start. I report it, and re-running the reference (≈ 4 GPU-h) is the user's call.
- **Also checked:** the re-run's recorded `config.yaml` differs from 0f2411's only in
  `train.steps` (6,500 against 48,000) and in the new keys at their defaults: `data.curriculum`
  null (off), `train.copy_every` 10 and `train.grad_norm_every` 500 (on).

**Why 6,500 steps:** they cover every stage switch, and 500 steps with all four stages active.
- near from step 1;
- far from 2,001;
- goal from 4,001;
- pixel from 6,001, with the share weighting.

A's gradient norms run at completed steps 0, 500, …, 6,000, which are logged steps 1, 501, …, 6,001.
So the last one falls on the first step with all four stages active, and the 499 steps after it are
compared. The copy baseline runs every 10 steps throughout.

**Limit:** this is a check of steps 1–6,500, not a proof for 6,501–48,000. After step 6,000 no
stage starts and the per-step code path doesn't depend on the step number. So a bitwise match here,
with all four stages active, is strong evidence for the rest. It doesn't cover a curriculum switch,
because the curriculum is off by default.

**Budget:** one training run of 6,500 steps, about 15 min on MPS under the GPU lock. It runs only
with no other project job and ≥ 10 GB free. No eval.

## Results
- **Commit:** **2fa0446**, clean (`meta.json`: dirty false). That is this lane merged with main
  b7f4d57, which has H PR 1 and A (df688b3).
- **Run:** `20261120T115641Z-worktree-wt-exp-m7-refcheck-m7-refcheck-s0-9ad7c5`, MPS,
  2026-11-20 03:56–04:09 local (about 13 min). It exited 0, with checkpoints at 2,000, 4,000, 6,000
  and 6,500.
- **Pass rule:** `uv run python experiments/compare_metrics.py --run <9ad7c5> --ref <0f2411>`
  **exited 0**:
  > 6500 rows x 53 keys compared; differing values: 0; max abs diff 0.0; diagnostic keys on one side only: 40
  - Every shared key is bitwise identical at every one of steps 1–6,500.
  - The 40 one-sided keys are all A's diagnostics, present only in the re-run:
    - 33 `gradnorm/enc/<term>/<level>` keys;
    - 7 `relerr/copy/k` keys.
- **Diagnostics ran as planned** (from the re-run's `metrics.jsonl`):
  - gradient norms at logged steps 1, 501, …, 6,001 (13 steps);
  - step 6,001's include the `pix` term, so the last one ran with all four stages active;
  - the copy baseline at 650 steps (1, 11, …, 6,491).
- **Recorded config:** 121 leaves against the reference's 118. Exactly 4 differ:
  - `train.steps`: 6,500 against 48,000;
  - `data.curriculum`: null (off), absent in the reference;
  - `train.copy_every`: 10, absent in the reference;
  - `train.grad_norm_every`: 500, absent in the reference.

## Conclusion
**Pass: H0 holds.** At 2fa0446, training with the defaults and A's diagnostics on repeats the M7
reference bitwise through step 6,500, including:
- every stage switch;
- 500 steps with all four stages active;
- a gradient-norm step on the first pixel step.

So the reference runs `…-0f2411` and `…-ada237` stand for the M7 ablations at this commit, and no
reference re-run is needed. The usual limit applies: this is evidence for steps 6,501–48,000, not
proof. Each ablation still checks before it runs that
`git diff 2fa0446 <its sha> -- jepa_world/models jepa_world/train.py jepa_world/data configs/base.yaml`
is empty. If it isn't, the ablation needs this check again.
