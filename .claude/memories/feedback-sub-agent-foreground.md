---
name: feedback-sub-agent-foreground
description: Sub-agents that arm Monitor + exit prematurely orphan background processes. For sub-30-min runs, sub-agent should run the command in the FOREGROUND with `timeout`, not background+Monitor.
metadata:
  type: feedback
---

When launching a sub-agent to run a bounded long-ish task (15-60 min), do NOT instruct it to `Bash run_in_background=true` + `Monitor`. The agent's session quota burns through the Monitor wait, the agent exits "completed" with a "Monitor armed; waiting" status, and the background process keeps running with no agent watching it. Result: no completion notification, no artifact verification, orphan procs.

**Why:** First observed 2026-05-26 during the v1 pilot GBDT experiment relaunches (post-disk-wedge recovery). Three consecutive Exp 1 / 2 / 3 launches all hit this — agents armed Monitor, sessions terminated as "completed" while the actual experiment process was still running unattended. The pattern is sticky: even with explicit "do not arm Monitor" instructions in the prompt, agents fall back to it when "watching a long-running task." Reinforced when Exp 3 (midcap150) agent wrote a custom fetch script that takes ~3 hours, armed Monitor on it, then exited — orphaning the fetch with no chain back to running the actual experiment.

**How to apply:**

1. **Sub-30-min tasks:** instruct the sub-agent to run the command in the FOREGROUND with `timeout` as a hard cap:
   ```bash
   timeout 1800 uv run python -m gbdt.experiment <spec.yaml> 2>&1 | tee logs/exp.log
   ```
   The sub-agent stays in its session through the wait. When `timeout` returns, the agent reads the artifact + reports. Single notification, single source of truth.
2. **Multi-hour tasks (≥2h):** background + Monitor IS appropriate, but add a chain step — explicitly tell the sub-agent that when Monitor signals completion, it must finalize (read report, verify artifact, post to PR). Or split into a "launch" agent + a "finalizer" agent driven by parent's `ScheduleWakeup`.
3. **Parent-side polling cadence for orphaned background work:** if a sub-agent exits with "Monitor armed; waiting" but the underlying process is alive, parent should `ScheduleWakeup` every 1800s to check process state + artifact appearance. This is the "I'll periodically check on you" fallback when the agent itself drops the watch.
4. **Pre-flight in skill prompts:** SKILL.md for long-running skills should explicitly call out the foreground-with-timeout pattern as default; background+Monitor is for >2h compute only, with a documented finalization step.
5. **Detection symptom:** sub-agent session result includes "Monitor armed" + tool_uses ≤ 15 + duration ≤ 5 min → it almost certainly exited prematurely. Don't trust its "completed" status; verify the underlying work.

**Don't apply when:** the task genuinely is a few seconds (pre-flight check, file inspection) — synchronous Bash call is fine. Or when the task is truly long-running (≥2h walk-forward tune) — then background + Monitor + ScheduleWakeup chain is the right pattern per `[[feedback-experiment-agent-loop]]`.

See `[[feedback-experiment-agent-loop]]` for the long-running pattern and `[[feedback-track-agent-tasks]]` for how tasks should reflect actual agent state (don't mark "completed" if the underlying work is still running).
