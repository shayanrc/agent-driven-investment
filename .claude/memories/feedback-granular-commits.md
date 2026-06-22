---
name: feedback-granular-commits
description: Always make granular commits — one focused, logical change per commit, not one lump
metadata:
  type: feedback
---

**Rule:** make **granular commits** — each commit is one focused, self-contained
logical change with its own clear message. Do not lump unrelated edits (a feature + a
doc fix + a refactor) into a single commit.

**Why:** granular history is reviewable, revertible, and bisectable — each commit can be
understood, cherry-picked, or rolled back on its own. The user asked for this explicitly
on 2026-06-22.

**How to apply:**
- Split a body of work by concern: e.g. migrating three memory rules → three commits,
  one per rule (memory file + its index pointer + its CLAUDE.md line together, since
  those *are* one logical change).
- Stage selectively (`git add <paths>`, or `git add -p` for hunks) so each commit
  contains only its change. `git add -i` is not available in this environment.
- Keep tooling/infra changes separate from the results/docs they enable when they're
  distinct concerns.
- Still one branch per plan ([[feedback-branch-retention]] + the per-plan branch rule);
  granularity is about *commits within* the branch, not more branches.
- Merge method is unchanged: rebase-and-merge ([[feedback-rebase-merge]]), never
  `--squash` (which would collapse the granular history), never `--delete-branch`.
