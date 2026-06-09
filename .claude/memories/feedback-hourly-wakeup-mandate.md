---
name: feedback-hourly-wakeup-mandate
description: Hourly wakeup (task #94) must run a health-check + triage pass and ACT on the recommended course of action — never stall waiting for user input
metadata:
  type: feedback
---

At each hourly wakeup (the recurring cron, task #94), run a brief **health-check + triage pass and then ACT**. **Always include the current wall-clock time at the top of the response** — run `date '+%Y-%m-%d %H:%M %Z'` and surface it, so elapsed-time between wakeups is concrete (not estimated from heartbeat counters).

- Check open PRs, background agents/tasks, running experiments/sweeps (e.g. #107), and disk (≥ 10 GB free).
- **For any running sweep, surface the best-cell-so-far** per `[[feedback-sweep-best-cell-reporting]]` (cell name + R-Precision@10 test lift + base rate, NOT just `N/M done`). If 0 cells done: "0 done — no best yet."
- **Cache health-checks MUST target the scratch cache** with `--data-root ${SCRATCH_CACHE}` (e.g. `uv run python -m scripts.data_pipelines.skill_runner health --data-root ${SCRATCH_CACHE}`). `${SCRATCH_CACHE}` is per-machine — substitute the literal from per-user memory `scratch-cache-path`. The main checkout's `data/processed.db-wal` is filesystem-corrupted (see `[[project-nse-data-quirks]]` § 3) — never query or poke the main checkout's `data/`.
- Fire the autonomous review/fix/merge sub-agent for any assistant-authored PR not already in a review cycle — except the HOLD-for-user categories (CLAUDE.md / spec / policy / SKILL.md / memories). See `[[feedback-auto-fire-review-merge]]`.
- Troubleshoot stalled runs (dead process, stale heartbeat, no new cell in > 1 h) or low disk per the disk-wedge + sweep-recovery playbooks (`[[feedback-disk-wedge-pattern]]`).
- Reconcile + triage the todo list; render the dependency graph to PNG when sequencing (`[[feedback-task-priority-graph]]`).
- **If a prior turn ended waiting on a user decision, take the recommended course of action instead of stalling.**
- If all healthy and nothing actionable: one-line "all green" and stop (still include the best-cell line for any running sweep).

**Why:** the user explicitly wants the wakeup to keep work moving autonomously while they're away (2026-05-28 directive: "if you've been waiting for user input at the hourly wakeup, take the recommended course of action"). A wakeup that just re-asks a pending question wastes the cycle. This is consistent with the project's autonomous PR review/merge default in CLAUDE.md.

**How to apply:** fires hourly while the REPL is idle. Keep output brief; never duplicate work already in flight. Respect the standing constraints even when acting autonomously: `--rebase` not `--squash` (`[[feedback-rebase-merge]]`); no `--delete-branch` (`[[feedback-branch-retention]]`); no `gh pr review --approve` (use `gh pr comment`); hold policy/spec/SKILL/memory-doc PRs for the user; no `pkill -f` in sub-agents (`[[feedback-agent-pkill-antipattern]]`).

**Operational note:** recurring crons in this environment auto-expire after 7 days — re-arm the cron weekly.
