# Does the M7 reference still hold after m5's H (and A)? A 6,500-step bitwise re-run

- **Date:** 2026-11-20   **Lane/branch:** worktree-wt-exp-m7-refcheck
- **Commit:** the one this lane has after merging origin/main, with m5's H PR 1 and, if it has
  merged by then, A. Filled in under Results with what it contains.
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
- **A** logs each head's encoder-gradient norm.

Both are meant to leave training unchanged when off: curriculum null, and logging that doesn't
touch the update. **Does training at the new commit repeat the reference bitwise?**

## Hypothesis and pass rule (pre-registered)
**H0: with the defaults, the new code trains exactly as 6fd7059 did.**

- **Pass:** `uv run python experiments/compare_metrics.py --run <re-run> --ref <0f2411>` exits 0,
  using main's version of the script at the check's commit. That means:
  - every one of steps 1–6,500 is present in the reference;
  - every key both runs log has an identical value at every step (NaN = NaN);
  - no non-diagnostic key is present on only one side.
- **New diagnostic keys:** m5's A lands its own version of the script. It exempts diagnostic keys
  by `train.DIAGNOSTIC_PREFIXES` (`relerr/copy/`, `gradnorm/`): a diagnostic key logged by only
  one run is reported and not counted. So A's gradient norms don't fail the check, but every
  shared key still must match bitwise.
  - If A hasn't merged by the check, there are no such keys, and the script requires the keys to
    match exactly.
  - (An earlier draft added an `--allow-new-keys` flag in this lane. It was dropped in favour of
    A's version, so the two lanes don't collide on the file.)
- **Fail:** any differing value, or a non-diagnostic key missing on either side. Then the M7
  ablations do not start. I report it, and re-running the reference (≈ 4 GPU-h) is the user's call.
- **Also checked:** the re-run's recorded `config.yaml` differs from 0f2411's only in
  `train.steps` (6,500 against 48,000) and in new keys at their off defaults (`data.curriculum`
  null, and A's key if any).

**Why 6,500 steps:** they cover every stage switch, and 500 steps with all four stages active.
- near from step 1;
- far from 2,001;
- goal from 4,001;
- pixel from 6,001, with the share weighting.

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
