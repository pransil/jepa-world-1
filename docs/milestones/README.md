# Milestone summaries

SPEC §8: "Each milestone ends with passing tests and a short written summary." That summary is
`docs/milestones/M<n>.md`, added in the PR that completes the milestone (`/builder` Step J).

## Template

```markdown
# M<n>: <name>

- **Completed:** YYYY-MM-DD   **PRs:** #…, #…

## Built
Modules and what each does (one line each). Link the DESIGN.md files.

## Proven by
| SPEC §8 test | test name(s) | tier |
|---|---|---|

## Measured
Numbers that matter for later milestones (e.g. L1 head step time on MPS, memory per batch,
dataset size and generation time). Each number cites its run dir or command.

## Open issues / deferred
Anything left for a later milestone, with the reason.
```
