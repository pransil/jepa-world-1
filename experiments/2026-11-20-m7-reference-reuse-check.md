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
(Filled in after the run.)

## Conclusion
(Filled in after the run.)
