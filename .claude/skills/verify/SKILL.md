---
name: verify
description: Definition-of-done check for a change in this repo. Runs lint and the tests, checks integration, walks the ML-correctness checklist (stop-grad, EMA, leakage, determinism, device, numerics, logging) and the evidence rules, and ends in a READY / BLOCKED report. Use before committing or merging, when /builder calls it, or when asked to verify or check a change.
argument-hint: optional, what changed (default: detect from git)
---

# /verify: definition of done

**Scope (optional):** $ARGUMENTS

## Step 1: What changed, and against what

```bash
git fetch -q origin
git diff --name-only origin/main...HEAD     # committed on this branch
git status --porcelain                      # not yet committed
```

**State the base ref and the file count up front.** If the list is empty or isn't the change you
were asked to verify, you are in the wrong tree (the main clone instead of the worktree?). Stop
and say so. Do not verify nothing and report READY.

Classify each file: `sim/` · `data/` · `models/` · train/eval · tests · configs · tooling
(`tools/`, `.claude/`, CI) · docs.

## Gate 1: Lint and tests (run them; don't predict them)

- Run `./test`, and report its RESULT line.
- If any of `sim/`, rendering, MPS paths, or a file that has `slow` tests changed, also run
  `./test all`, and report that RESULT line too.
- If any tier was skipped (no MPS, no GL), name it with the reason. A skip is not a pass.
- If a test fails, show the failure and **stop**: the overall result is BLOCKED.

## Gate 2: Integration

State each item as PASS, an issue, or N/A with a reason:

- **Signatures and shapes:** if a public function, module output shape, or config key changed,
  find every caller (`rg -n "<name>" jepa_world tests configs`) and confirm each was updated.
- **Config:** every new key is in `configs/base.yaml` (with a comment) and is actually read by the
  code. No hyperparameter is hard-coded in a module.
- **Artifacts:** new datasets are written only through `paths.build_dataset_once`, and runs only
  through `paths.new_run_dir`. If simulator output changed, `sim.generator_version` was bumped.
- **GPU:** new long-running MPS work (training, benchmarks) runs under `locks.gpu_lock`.

## Gate 3: ML correctness

Read the changed code and reason through each item that applies. Write N/A, with the reason, for
the rest. Never summarise this gate as "looks fine".

- **Stop-gradient:** the target encoder runs under `torch.no_grad()` and its outputs are
  `.detach()`ed before the loss. None of its parameters are in the optimizer. A test asserts
  `p.grad is None` for every target param after `backward()`.
- **EMA:** θ_t ← m·θ_t + (1−m)·θ_c, applied **after** `optimizer.step()`, once per step, over
  parameters *and* buffers handled deliberately (copy or EMA, but say which). m comes from the
  config.
- **Leakage:** the context frames are strictly before t (indices t−3…t), the target frame is
  exactly t+k, and nothing at or after t+k reaches the context. Splits are by terrain seed, and
  train, dev and test seeds are disjoint. Hidden ball parameters never reach the model's input.
- **Determinism:** seeds come from the config and pass through `repro.seed_everything` and
  explicit generators. The generator config is stored with the dataset. Same seed gives the same
  output (there's a test for it).
- **Device:** there are no silent CPU↔MPS copies in the training hot loop (`.item()`, `.cpu()`,
  or numpy calls per step). fp32 unless a lower precision was tested explicitly (SPEC §6). Any op
  that hits `PYTORCH_ENABLE_MPS_FALLBACK` is noted.
- **Numerics:** log-variance is clamped or softplus-bounded, the NLL is finite on the test batch,
  and there is no `log(0)` or divide by a batch std without an epsilon.
- **Losses and logging:** each term (each L_k, L_goal, L_pix, and the collapse guard) is logged
  separately every step (SPEC §5.4). Weights come from the config.
- **Evaluation hygiene:** nothing is tuned or selected on the test split. Headline numbers are on
  held-out terrain seeds, and dev numbers are labelled dev (SPEC §7.3).

## Gate 4: Evidence

- **Every new test was seen to fail** before the code existed, or would fail if the change were
  reverted. Break the *behaviour* (skip the `detach`, swap the EMA order), not the syntax. A
  syntax error tells you nothing.
- **Every claim cites an artifact.** "Works", "converges", "beats persistence" and "doesn't
  leak" each point to a test name, a `./test` RESULT line, or a run directory with
  `metrics.jsonl`. Code reading shows what *can* happen. Only an artifact shows what *did*.
- **Absence claims state their scope.** "X is never called" requires a search whose scope you
  state in the sentence ("in `jepa_world/` and `tests/`"). Use a count (`rg -c`), not a listing
  you scrolled through, and don't pipe into `head`.
- **Docs still true:** if behaviour changed, the module's `DESIGN.md` and README sections that
  describe it were updated, and any `[OPEN]` value changed only alongside a `docs/decisions/` file.

## Report

```
VERIFICATION REPORT: base origin/main, <N> files changed (<classes>)
Gate 1  lint+fast: PASS/FAIL (RESULT: ...)   all: PASS/FAIL/N/A   skipped tiers: none/<tier: reason>
Gate 2  shapes/callers: ...  config: ...  artifacts: ...  gpu lock: ...
Gate 3  stop-grad: ...  EMA: ...  leakage: ...  determinism: ...  device: ...  numerics: ...  logging: ...  eval hygiene: ...
Gate 4  seen-to-fail: ...  claims cited: ...  absence scoped: ...  docs: ...
Overall: READY / BLOCKED: <what must change>
```

Do not report READY while any gate has an unresolved issue.
