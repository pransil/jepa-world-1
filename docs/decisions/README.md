# Decisions

One file per decision: `docs/decisions/<YYYY-MM-DD>-<slug>.md`. These are mostly the SPEC §9 open
questions (camera mode, top-down prediction, collapse guard, L1 target size, distance function,
horizon ordering) and any `[OPEN]` value in `configs/base.yaml`.

**The user decides these** (SPEC §11: "Ask before changing any item in section 9"). A lane that
needs one asks, with options and a recommendation, records the answer here in the same PR as the
code that depends on it, and updates the `[OPEN]` marker in `configs/base.yaml`.

Files are named by date and slug, not by number, so two lanes deciding things on the same day
can't collide.

## Template

```markdown
# <decision, e.g. "Camera mode: track_lagged">

- **Date:** YYYY-MM-DD   **Decided by:** <user>   **SPEC ref:** §9.<n> / §<x>
- **Status:** accepted | superseded by <file>

## Question
What had to be decided, and why now.

## Options
- **A:** … (cost / benefit)
- **B:** …

## Decision
The choice, in one sentence, and the reason that carried it.

## Consequences
What code, config or tests change. What would make us revisit it.
```
