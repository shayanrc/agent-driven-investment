---
name: feedback-rebase-merge
description: Merge PRs with `gh pr merge <num> --rebase` (rebase-and-merge), NEVER `--squash`. Standing user directive 2026-05-28
metadata:
  type: feedback
---

Merge PRs in this repo with **`gh pr merge <num> --rebase`** (rebase-and-merge) — **not** `--squash`.

**Why:** the repository owner gave the standing directive "always rebase and merge" (2026-05-28, when approving PR #61). Rebase-and-merge preserves the individual commits and keeps history linear rather than collapsing a branch into one squashed commit. This overrides the older `--squash` instruction that was originally in CLAUDE.md; the codified rule moved into CLAUDE.md in PR #63 (merged 2026-05-28, `7435e9c`).

**How to apply:**
- Every merge — both merges performed directly by the assistant and the instructions baked into autonomous review/merge sub-agent briefs (tell them `--rebase`, not `--squash`).
- Unchanged guardrails still hold: **NEVER `--delete-branch`** (branch-retention policy, see `[[feedback-branch-retention]]`); approve via `gh pr comment`, **never** `gh pr review --approve` (author self-approval is blocked).
- See `[[feedback-auto-fire-review-merge]]` for when to auto-fire the review/merge cycle in the first place.

**The method directive is NOT a blanket merge authorization.** It sets the *method* used when merging, not whether to auto-merge a given PR. The HOLD-for-user categories (CLAUDE.md / spec / SKILL.md / policy / memories) still wait for explicit user approval — the self-merge of PR #63 was initially classifier-blocked (self-authored policy doc) and only went through after explicit user approval. Don't conflate "use rebase" with "merge it now."
