---
name: feedback-longrun-heartbeat
description: Always set up an hourly heartbeat (cron) to monitor long-running work — diagnose issues, report status+ETA in IST proactively — plus the false-stall diagnostics (check the real worker RSS, not the uv wrapper).
metadata:
  type: feedback
---

Whenever a long-running task is in flight (gbdt sweeps, back-extends, panel builds,
finetunes, multi-hour fetches), ALWAYS set up an **hourly heartbeat** (a recurring cron,
e.g. `7 * * * *`) that checks the running work, **diagnoses issues** (failed/rc≠0 cells,
stalls, OOM, disk pressure, currency-check refusals), and **reports status + ETA
proactively** — don't wait to be asked. (Migrated from per-user memory 2026-07-15.)

**Why:** the user runs many multi-hour sweeps and wants proactive visibility + early
problem detection, not silence until completion. Event notifications only fire on task
*completion*; the heartbeat catches mid-run problems and ETA drift.

**How to apply:** on starting any long-running work, create the hourly heartbeat
immediately. Parse the run logs (DONE/SKIP/rc≠0 counts, pace, current cell), check box
health (mem/disk/load), post a concise status+ETA in **IST** per
[[feedback-report-time-in-ist]], and surface the best-cell-so-far for sweeps per
[[feedback-sweep-best-cell-reporting]]. Tear it down (CronDelete) when the chain is idle.
The recurring health-check pass itself is specified in [[feedback-hourly-wakeup-mandate]].

**Avoid false-stall alarms** (the assistant cried "wedged" 2–3× in one session, all wrong):
- **Check the REAL worker, not the `uv run` wrapper.** `uv run python -m …` spawns a
  child; the wrapper holds a tiny RSS (~38 MB) while the child holds the GB-scale RSS.
  Resolve the child PID (`pgrep -f .venv/bin/python`) and read ITS rss/state/CPU.
- **A frozen log + full CPU + no output is usually WORK, not a stall.** Before alarming,
  confirm CPU is *accumulating* (state `R`, `utime` rising over a few seconds) and RSS is
  GB-scale. A busy-`R` process pegging a core is computing, not sleep-hung.
- **`/daily-predictions`' final self-check looks exactly like a stall.** The
  `[validate] reproducing predictions/test.csv` phase re-runs full inference — CPU-heavy,
  logging only at start + end, so the log freezes for minutes near the end. Normal.
- **`&`-backgrounded fits: the launcher returns while the fit runs on.** A Bash task
  showing "completed exit 0" means the *launcher* finished, not the fit. Stdout tails
  buffer/truncate — trust process-alive + the run's own progress log. See
  [[feedback-process-death-verification]] before declaring anything dead.
- If genuinely unsure, FLAG with the diagnosis + recommendation and let the user decide —
  don't unilaterally kill a long job ([[feedback-unrequested-actions]]).
