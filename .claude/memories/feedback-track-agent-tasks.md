---
name: feedback-track-agent-tasks
description: Every sub-agent launch gets a TaskCreate immediately and is set to in_progress for the duration of the agent's work. Tasks are the source of truth for what's in flight.
metadata:
  type: feedback
---

When launching a sub-agent (any `Agent()` call, foreground or background), create a corresponding task with `TaskCreate` in the same turn AND mark it `in_progress` while the agent is running. Mark it `completed` (or `deleted` if superseded) when the agent's notification reports it's done.

**Why:** User requested this 2026-05-25 after observing several rounds of "agent launched but no task entry" + "task stuck at `pending` while the agent is actively running." The TaskList is the durable handoff between turns and across session compactions; if it doesn't reflect what's in flight, future turns lose visibility. This was reinforced as a *standing* rule, not a one-off.

**How to apply:**

1. **At launch:** every `Agent()` call → immediately `TaskCreate` with a subject naming the agent's job + `activeForm` describing what it's actively doing. Default status is `pending`; promote to `in_progress` via `TaskUpdate` in the same turn (TaskCreate doesn't accept `status` directly).
2. **While running:** the task stays `in_progress`. If multiple agents are working on the same conceptual deliverable, prefer separate tasks (one per agent) over one shared task — easier to track per-agent state.
3. **On completion notification:** mark `completed`. If the agent terminated with uncommitted work and the parent took over, mark the task `completed` only after the parent's follow-up actually closed out the work.
4. **For long-running background agents** (seed runs, walk-forward tunes), the task can stay `in_progress` across many notification messages. Update the description if scope shifts mid-flight (e.g., "agent pivoted to tighter window after retry-storm diagnosis").
5. **Refactor stale tasks:** if a task's description doesn't reflect reality anymore (because the agent's plan changed, or because the parent's understanding of state was wrong), update the description in place. Don't leave drift.

**Don't apply when:**
- Synchronous foreground actions you (the parent) perform directly — those aren't agent launches. Tasks are for delegation, not self-tracking of every step.
- Short bash commands run in the parent — same reason.

See `[[feedback-experiment-agent-loop]]` for what to put in the launch prompt itself when the work is long-running.
