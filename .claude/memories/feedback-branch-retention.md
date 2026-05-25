---
name: feedback-branch-retention
description: Keep merged plan/feature branches around (local + remote) after merging — don't propose deletion.
metadata:
  type: feedback
---

After a plan/feature branch is merged to its parent (typically `main`), the branch stays around — both local and remote. Don't offer to delete it.

**Why:** The user explicitly answered "let's keep them for reference" when offered post-merge branch cleanup of `data-pipelines-v1` and `data-pipelines-v1.7-nse` (2026-05-24, after PR #1 merged). The reason isn't accidental: the branch shape (per-plan branch + `--no-ff` merge) is itself part of the history they want preserved — git log shows which work belonged together as a coherent slice, and the branch refs are the way to navigate back to "the v1.7 NSE work" without having to remember commit SHAs.

**How to apply:**
- Don't suggest `git branch -d` / `git push origin --delete` after a merge, even when the branch is "fully merged and safe to delete."
- Don't suggest it as an "optional cleanup" either — that was the framing that prompted the rule.
- When listing post-merge actions in a summary, omit branch deletion entirely. If the user wants it deleted they'll ask.
- Applies to: plan branches (`V<N>_PLAN.md` branches), feature branches (`data-pipelines-v1.7-nse`-style), and the intermediate dev branches (`data-pipelines-v1`).
- Exception: branches the user explicitly says were a mistake (e.g., accidental checkout, dead spike) can be deleted on request.

See `[[project-overview]]` for the multi-module structure that produces these branches.
