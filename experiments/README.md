# Experiments

One file per experiment: `experiments/<YYYY-MM-DD>-<slug>.md`. Write them with `/experiment`
(`.claude/skills/experiment/SKILL.md`). Never edit another experiment's note. To revisit a
question, write a new note that links the old one.

Run artifacts (checkpoints, metrics, TensorBoard) live outside git, in
`$JEPA_ARTIFACTS/runs/<run-dir>/`. A note records the run dir name and the commit sha, so its
numbers can always be traced to the code and config that produced them.

## Template

```markdown
# <question, as a title>

- **Date:** YYYY-MM-DD   **Lane/branch:** worktree-wt-…   **Commit:** <sha> (clean | dirty)
- **Config:** configs/<slug>.yaml (+ `--set` overrides, if any)
- **Run dirs:** <run-dir names>, one per seed/setting

## Hypothesis
One sentence. **Deciding metric:** <metric, split, horizon>. **Refuted if:** <outcome>.

## Results
| setting | seed | <deciding metric> | ratio to persistence | physics baseline | notes |
|---|---|---|---|---|---|

Training sanity: converged? collapse (embedding std per level)? loss terms sane?

## Conclusion
**supported / refuted / inconclusive**, with the number that decides it. Dev-set numbers are
labelled as dev.

## Next
What this suggests trying next, if anything.
```
