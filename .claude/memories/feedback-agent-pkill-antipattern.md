---
name: feedback-agent-pkill-antipattern
description: Never use `pkill -f <pattern>` in bash wrappers — it catches ALL matching processes (sub-agent variant kills the parent's concurrent work; parent variant self-kills the Bash tool's own shell → exit 144). Kill by PID; prefer resumable runners.
metadata:
  type: feedback
---

When a sub-agent needs to enforce a timeout on a subprocess (e.g. a smoketest experiment), it MUST NOT wrap the launch with patterns like:

```bash
bash -c 'timeout 280 uv run python -m gbdt.experiment ... & sleep 270 && pkill -f "gbdt.experiment"; wait'
```

The `pkill -f "gbdt.experiment"` is the bug. It matches the substring `gbdt.experiment` in ANY process's command line on the system, including:
- Other gbdt experiments the parent launched concurrently (e.g. a 2.5h cell-replication run)
- Reviewer agents running their own smoketests
- The same parent's prior unrelated invocations still in flight

**Why:** Discovered 2026-05-27 — the A+B logging sub-agent (PR #39) embedded this pattern as belt-and-suspenders timeout enforcement. Its `sleep 270 && pkill -f "gbdt.experiment"` fired at the configured 270s mark and killed BOTH its own smoketest AND the parent's concurrent sp500 H=25 cell-B run (PID 95877, ~5 min in). The sp500 run died with exit code 144 (signal 16 — SIGSTKFLT on Linux, the kernel-reported version of certain SIGTERM-from-unprivileged-userspace deliveries on x86), looking exactly like the *first* sp500 failure from earlier the same day. Two sp500 attempts lost, ~30 min of debugging overhead to identify the cross-process kill.

The agent had been given an explicit prompt: "use `timeout 300 ...`" — which is sufficient on its own (timeout(1) only kills the subprocess it manages). The pattern-kill was over-engineering that turned into a cross-process foot-gun.

**How to apply (paste into the agent launch prompt, especially for any agent running gbdt smoketests, data_pipelines fetches, or anything else that has multiple potentially-concurrent instances):**

> **CRITICAL — DO NOT USE `pkill -f <pattern>` IN ANY BASH WRAPPER.** Pattern-based kills catch ALL matching processes on the system, not just yours. If you need to kill a subprocess you launched, target it by exact PID:
>
> ```bash
> # GOOD: targets only what you launched
> PYPID=$!
> sleep 270 && kill -TERM $PYPID
> ```
>
> Or just use `timeout(1)` alone — it kills only the subprocess it manages:
>
> ```bash
> # GOOD: timeout enforces deadline cleanly
> timeout 300 uv run python -m gbdt.experiment ...
> ```
>
> Do NOT wrap timeout in a belt-and-suspenders pattern that uses pkill -f. If the implementer's timeout doesn't fire (e.g. Python ignored SIGTERM somehow), the next steps are to escalate to the parent agent, NOT to broadcast pkill.

**Also applies to:**
- `killall <name>` — same cross-process scope.
- `pgrep -f <pattern> | xargs kill` — same issue.
- Any subshell that does `for p in $(pgrep ...); do kill $p; done`.

**The self-kill variant (bites the PARENT session too — merged from per-user memory 2026-07-15):**
`pkill -f "<pattern>"` matches against the FULL command line of every process — including
the Bash tool's **own shell**, whose command string contains the very pattern being passed
to pkill. The pkill kills its own parent shell and the command aborts with **exit 144**
before any later step (spec-gen, relaunch) runs. This bit 3+ times in one session: it
killed running gbdt sweeps mid-flight and repeatedly aborted "kill + regenerate + relaunch"
compound commands at the pkill line, leaving specs unwritten and waiters stuck. Each
failure looked like a mysterious `exit 144`.

- To stop a background job you launched, **kill by PID** — capture `pid=$!` at launch (or
  `pgrep` into a var), then `kill $pid`. Never `pkill -f` a pattern that is a substring of
  the command doing the killing.
- If a pattern-kill is truly needed (e.g. cleanup of stuck nselib network hangs), isolate
  it in its OWN Bash call with a pattern that CANNOT appear in that call's text, and expect
  a non-zero rc (`|| true`). Only the parent session — with full visibility into what's
  safe to kill — may do even this; sub-agents never.
- Better: make runners **resumable/idempotent** (skip a cell if its output exists, e.g.
  `[ -f eval.csv ] && continue`) so recovery is "relaunch", not "kill + restart".
- A `&`-backgrounded fit keeps running after the launcher returns — the Bash task showing
  "completed" means the launcher finished, NOT the fit. Verify process-alive + progress
  log per [[feedback-process-death-verification]] / [[feedback-longrun-heartbeat]].

See also `[[feedback-sub-agent-foreground.md]]` for the related antipattern of background+Monitor for short tasks, and `[[feedback-experiment-agent-loop.md]]` for the loop guidance that should be in long-running agent prompts.
