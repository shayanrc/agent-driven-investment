---
name: feedback-experiment-agent-loop
description: When launching a sub-agent for an experiment that may exceed a single session, include explicit loop-pattern guidance in the launch prompt up front.
metadata:
  type: feedback
---

When spinning up a sub-agent for any module's experiment work that may exceed a single agent session — most notably canonical walk-forward runs (~hours of compute) or bulk data seeds — the launch prompt must include explicit guidance on the long-running pattern. I cannot send the agent a follow-up message: the `SendMessage` capability referenced in the Agent tool docs is not exposed in this environment.

**Why:** User requested this 2026-05-24 after observing that V5.A.2 was launched without loop guidance. V5.A.2 itself is bounded (cached forecasts, no walk-forward) so it survived without it, but V5.B (~12h compute) would not. The rule is to set the pattern at launch, not retroactively. Reinforced 2026-05-25 when the NIFTY total-market seed agent had to figure out its own pacing (4-worker parallel + 60s hard-kill per subprocess) mid-flight rather than receiving it up front.

**How to apply (paste into the agent's launch prompt):**

> For any compute that may take more than ~30 minutes, do not block your single tool call waiting for it. Instead:
>
> 1. Launch the long shell with `Bash run_in_background=true` and a stable log path (e.g. `logs/<exp_id>_<UTC>.log`).
> 2. Use `Monitor` with a filtered `tail -f <log> | grep -E --line-buffered "elapsed|fold=|Error|Traceback|completed"` so progress events stream in without flooding context.
> 3. If the compute will likely outlast your session (rule of thumb: anything ≥2h), use `ScheduleWakeup` to self-pace — schedule a wakeup at 1200–1800s with a self-contained prompt that re-checks progress, decides whether to schedule the next wake, and ends when the job completes.
> 4. Never block one tool call for hours — that wastes the session and prevents you from being interruptible.
> 5. For bulk operations over many items (seeds, fold runs), prefer a parallel-worker approach (4 workers is a sane default for most CPU- or network-bound work) and add a per-item hard timeout (e.g. 60s subprocess kill) so one stuck item doesn't block the queue.

Also include in the launch prompt the reminder that I (the parent) will check in periodically and that the agent's auto-completion notification reaches me directly — so the agent doesn't need to do anything special to "signal done."

**Don't apply when:** the experiment is genuinely bounded to a single session (V5.A.2-style: a few hours of pure-Python compute that the agent can wait through synchronously). Adding the pattern unnecessarily adds prompt complexity for no value.

See `[[project-overview]]` for the module split; `[[feedback-branch-retention]]` for the post-merge branch policy that goes with this workflow.
