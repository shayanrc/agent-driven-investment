---
name: feedback-branch-per-plan
description: Each new plan doc (V<N>_PLAN, V<N>_EXPERIMENTS_PLAN, ABLATION_STUDIES_PLAN, etc.) gets its own git branch — don't commit to main
metadata:
  type: feedback
---

When starting a new plan doc (`V<N>_PLAN.md`, `V<N>_EXPERIMENTS_PLAN.md`, `ABLATION_STUDIES_PLAN.md`, etc.) and the experiments it specifies, create a dedicated branch and develop the plan + its implementation there. Don't stream commits to main.

**Why:** Stated by user 2026-05-20 after merging v3 work directly to main. Wants future plans isolated for reviewability and easy rollback — each plan + its experimental code + reports is a logical unit that should land as a PR, not as a stream of commits to main. The v3 work landed on main as a one-time exception.

**How to apply:**
- Before starting implementation of a new plan, run `git checkout -b <plan-slug>` (e.g. `v4-experiments`).
- Commit incrementally to that branch as experiments land.
- Push and let the user open / merge the PR when the plan is complete.
- Pure refactors of already-merged work (path renames, doc reorganisation) may go on main — the rule is per-*plan*, not per-*change*.
