---
name: feedback-experiment-agent-loop
description: When launching a sub-agent for an experiment that may exceed a single session, include explicit loop-pattern guidance in the launch prompt up front.
metadata:
  type: feedback
---

When spinning up a sub-agent for analog_mc (or any module's) experiment work that may exceed a single agent session — most notably canonical walk-forward runs (~hours of compute) — the launch prompt must include explicit guidance on the long-running pattern. I cannot send the agent a follow-up message: the `SendMessage` capability referenced in the Agent tool docs is not exposed in this environment.

**Why:** User requested this 2026-05-24 after observing that V5.A.2 was launched without loop guidance. V5.A.2 itself is bounded (cached forecasts, no walk-forward) so it survived without it, but V5.B (~12h compute) would not. The rule is to set the pattern at launch, not retroactively.

**How to apply (paste into the agent's launch prompt):**

> For any compute that may take more than ~30 minutes, do not block your single tool call waiting for it. Instead:
>
> 1. Launch the long shell with `Bash run_in_background=true` and a stable log path (e.g. `logs/<exp_id>_<UTC>.log`).
> 2. Use `Monitor` with a filtered `tail -f <log> | grep -E --line-buffered "elapsed|fold=|Error|Traceback|completed"` so progress events stream in without flooding context.
> 3. If the compute will likely outlast your session (rule of thumb: anything ≥2h), use `ScheduleWakeup` to self-pace — schedule a wakeup at 1200–1800s with a self-contained prompt that re-checks progress, decides whether to schedule the next wake, and ends when the job completes.
> 4. Never block one tool call for hours — that wastes the session and prevents you from being interruptible.

Also include in the launch prompt the reminder that I (the parent) will check in hourly via `ScheduleWakeup` and that the agent's auto-completion notification reaches me directly — so the agent doesn't need to do anything special to "signal done."

**Don't apply when:** the experiment is genuinely bounded to a single session (V5.A.2-style: a few hours of pure-Python compute that the agent can wait through synchronously). Adding the pattern unnecessarily adds prompt complexity for no value.

See `[[project-overview]]` for the analog_mc / data_pipelines module split; `[[feedback-branch-retention]]` for the post-merge branch policy that goes with this workflow.
