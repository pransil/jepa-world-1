# AGENTS.md

This repository's agent instructions live in **[`CLAUDE.md`](CLAUDE.md)**. Read it first.

This file only exists so that tools following the `AGENTS.md` convention find them. It has no rules
of its own, because a second copy would drift, and an agent would obey whichever copy it happened
to read.

The three pointers that matter most:

- **Parallel work:** one worktree per lane, push the branch as your claim, PR into `main`
  ([`docs/process/LANES.md`](docs/process/LANES.md)).
- **Testing:** `./test` is the one entry point (fast tier, < ~20s). `./test all` runs every tier.
- **Design:** [`SPEC.md`](SPEC.md) is the plan. Its §9 open questions are the user's call.
