---
name: feedback-auto-fire-review-merge
description: Default behavior — always fire the autonomous review/merge agent on PRs the assistant opens; do not pre-ask for permission per PR
metadata:
  type: feedback
---

When the assistant opens a PR in this project, the default is to **immediately fire the autonomous review/merge background agent for it** without asking first. Do not pause to confirm "want me to fire the reviewer, or hold for your eyes?" per PR.

**Why:** the user explicitly stated this preference after Sweep #1 PR (#27) on 2026-05-27. They want the PR pipeline to run end-to-end on autopilot. Pre-asking per PR is friction; the user will redirect if they want a hold. CLAUDE.md § Environment codifies the headline rule ("Autonomous PR review/merge pipeline is the default for this project"); this memory holds the *why* + the exception list.

**How to apply:**
- For every PR the assistant opens via `gh pr create`, immediately spawn a background review/merge sub-agent.
- Skip ONLY when the user explicitly asks to hold a specific PR (e.g., "don't auto-merge this one — I want to read it first" or "hold PR #X").
- The autonomous-merge authorization stands across the whole session — it is the per-project override of the per-machine `gh-cli-installed` note that says "merging is still a user action".
- Still respect non-negotiables: `--rebase` not `--squash` (see `[[feedback-rebase-merge]]`); NEVER `--delete-branch` (see `[[feedback-branch-retention]]`); approve via `gh pr comment`, NEVER `gh pr review --approve` (PR author = gh CLI user blocks self-approval); no AI attribution in commits/PR bodies.

**Exception cases** where pre-asking IS still appropriate (the HOLD-for-user categories):
- PRs that touch CLAUDE.md / `.claude/memories/` / `.claude/skills/SKILL.md` / spec docs / other project-wide policy.
- Spec changes or architecture decisions that warrant user judgment before landing.
- PRs that introduce a NEW dependency, NEW module, or NEW external API surface.
- When the assistant has a substantive concern about the PR content itself (e.g., spec amendment, methodology change).

See `[[feedback-rebase-merge]]` for the merge-method directive, `[[feedback-branch-retention]]` for the branch-deletion guardrail, and `[[feedback-hourly-wakeup-mandate]]` for the hourly-cron protocol that also auto-fires review/merge on stale PRs.
