---
name: project-agent-loop-wrapper
description: scripts/gbdt/run_agent_loop_resumable.sh is the standard launcher for long-running gbdt experiments — setsid detachment + auto-restart on crash (when checkpoint exists) + heartbeat-stall watchdog + idempotency
metadata:
  type: project
---

`scripts/gbdt/run_agent_loop_resumable.sh` (landed 2026-05-31 as task #191 / PR #92) is the **standard launcher** for any long-running gbdt experiment — sweep cells, agent-driven FS+HP loop runs, anything expected to take ≥ 30 minutes. Bare `uv run python -m gbdt experiment ...` is fine for sub-30-min foreground tasks; everything beyond that goes through the wrapper.

**What the wrapper provides:**

- **Hard detachment from the calling shell**: child runs under `setsid nohup` with stdin redirected from `/dev/null`. A SIGHUP delivered to the parent (e.g. a wrapping sub-agent terminated by a user-tier rate-limit, or an interactive shell that exits) does NOT cascade to the python process. This is the specific failure mode that lost the 2026-05-31 r1k validation trio.
- **Auto-restart-on-death via `--resume`**: when the python child exits non-zero, the wrapper checks for `<out-dir>/loop/checkpoint.json` (the V1.1 Phase 2 exit-and-resume location) and, if present, relaunches with `--resume <run_id>`. Capped at `--max-retries` attempts (default 3).
- **Heartbeat-stall watchdog**: if `<out-dir>/loop/progress.log` mtime goes stale by more than `--heartbeat-stall-secs` (default 1800 = 30 min), the wrapper SIGTERMs the process group via the captured PGID, waits 30s, SIGKILLs, then takes the restart path. 30 min covers the slowest legitimate single phase.
- **Idempotency guard**: refuses to double-launch against the same `--out-dir` while a prior PGID (recorded in `<out-dir>/.wrapper/pid`) is still alive. Prevents the duplicate-launch foot-gun in `[[feedback-process-death-verification]]`.
- **Atomic 8-state status JSON at `<out-dir>/.wrapper/status.json`**: `starting | running | restarting | heartbeat_stalled_killed | exited_ok | exited_failed | max_retries_hit | paused_awaiting_decision`. These are the 8 distinct values written by `write_status` callsites in `scripts/gbdt/run_agent_loop_resumable.sh` (verifiable via `grep -n 'write_status' scripts/gbdt/run_agent_loop_resumable.sh`). Written via temp+rename so a partial read never returns junk; safe to poll from a Monitor loop. `paused_awaiting_decision` (post-#193 bug 3) is what the wrapper writes when the V1.1 `agent_file_protocol` runner pauses at iter N awaiting an agent decision — both `exited_ok` and `paused_awaiting_decision` produce wrapper exit 0, but only `exited_ok` means the pipeline is complete.
- **Sidecar dotfile namespace** (post-#193 bug 2): all wrapper-owned state lives under `<out-dir>/.wrapper/` (`pid`, `status.json`, `log`). The runner's emptiness check filters dotfile entries, so wrapper state does not force callers to pass `--overwrite` on first launch. Pre-#193 installs wrote `<out-dir>/wrapper.{pid,status,log}` at out-dir top level — those leftovers are harmless but the new idempotency guard won't see them.
- **Agent-loop pause/resume contract** (post-#193 bug 3): with `--callback-mode agent_file_protocol`, the runner exits 0 after writing `loop/iter_N_request.json` (a pause, not completion). The wrapper detects pending requests without matching `loop/iter_N_decision.json`, writes `paused_awaiting_decision`, and exits clean. The orchestrator (Claude session, sub-agent, chain waiter) inspects the request, writes the decision, and re-invokes the wrapper with `--resume <run_id>`. Each iter is a separate wrapper invocation; the wrapper does NOT poll for decisions or auto-resume across pauses.

**When to use:**

- Single long-running gbdt experiments expected to run ≥ 30 minutes (cold-features build alone is ~3 h on r1k).
- Agent-driven FS+HP loop runs invoked from a sub-agent session — the sub-agent's lifetime should never gate the python's lifetime.

*Sweep scripts currently call `gbdt.experiment` directly (`run_nasdaq100_sweep.sh`, `run_russell1000_sweep.sh`); integrating the wrapper into the sweep loop so every cell is launched through it is a possible follow-up — out of scope for this memory.*

**Honest caveat — cold-features build is unprotected.** The checkpoint is written by the V1.1 loop *after* iteration 0 completes. If the process dies during the cold-features build (the ~3 h phase BEFORE iteration 0), no checkpoint exists, `--resume` isn't possible, and the wrapper exits `exited_failed` after the first crash. Closing this gap requires chunked features build (write intermediate state every family, resume from last completed family) — a separate, larger change. For now: treat the cold-features build as a single atomic phase, accept that an OOM / disk wedge during it costs the full build.

**Auto-restart amplifies environmental failures.** Auto-restart does NOT diagnose root cause. If the failure mode is OOM, disk wedge, or any environmental issue, the wrapper will reproduce the failure on each retry — burning N× the CPU/RAM. When the failure is environmental rather than transient, set `--max-retries 0` until the underlying capacity issue is fixed. Pre-flight only runs once at startup, not before each restart.

**Wrapper-itself-dies case.** If the wrapper bash process is killed (OOM, signal, user error), the detached child keeps running but loses its watchdog. Recovery: `kill -TERM -$(cat <out-dir>/.wrapper/pid)` to send SIGTERM to the child's process group; clean up `<out-dir>/.wrapper/status.json` (will be frozen in `running`); then relaunch the wrapper. The next launch's idempotency guard will refuse if the original PGID is still alive — manually verify with `ps -p <pgid>` first.

**Relationship to other launch patterns.** Supersedes `[[feedback-sub-agent-foreground]]`'s "background+Monitor+ScheduleWakeup chain" guidance for `≥ 30 min` gbdt experiments where a checkpoint exists post iter-0. For sub-30-min sub-agent tasks (typical: review/merge agents, doc-fixup agents), continue with foreground+`timeout` per `[[feedback-sub-agent-foreground]]`. The wrapper does NOT replace `[[feedback-experiment-agent-loop]]`'s loop-pattern guidance for the agent driving the experiment — it supervises the child python on the operator side.

**Reference invocation:**

```bash
bash scripts/gbdt/run_agent_loop_resumable.sh \
  --spec configs/gbdt/experiments/<cell>.yaml \
  --out-dir results/gbdt/experiments/<cell> \
  --data-root ${SCRATCH_CACHE} \
  [--max-retries 3] \
  [--heartbeat-stall-secs 1800]
```

**Why this exists (canonical incident):** 2026-05-31 r1k agent-loop trio loss. Three cells (nasdaq100/sp500/r1k at H=25) all launched as foreground `uv run python -m gbdt experiment ...` from sub-agent sessions. The sub-agent sessions terminated (rate-limit + watchdog) and — what I incorrectly believed at the time — SIGHUP'd their children. Combined with my misdiagnosis cascade (described in `[[feedback-process-death-verification]]`), the trio was rerun unnecessarily, costing ~15 h of cold-rebuild CPU. PRs #188 (post-mortem) and #191 (this wrapper) hardened against the SIGHUP-cascade and the dying-mid-run failure modes. The other half of the lesson — *verify before relaunching* — is `[[feedback-process-death-verification]]`.

**How to apply:** when writing a sub-agent brief for long-running gbdt work, instruct the sub-agent to invoke through this wrapper — never bare `uv run python -m gbdt experiment ...`. State the `--out-dir` explicitly so the wrapper's idempotency guard can do its job. For status checks, parse `<out-dir>/.wrapper/status.json` via `[[feedback-subagent-transcript-parsing]]`-style bounded queries; for liveness checks before any relaunch decision, follow `[[feedback-process-death-verification]]`. See `[[project-gbdt-tuning-playbook]]` for the loop semantics this wrapper hosts and `[[feedback-experiment-agent-loop]]` for the launch-side patterns.

**Source of truth:** the wrapper itself landed in commit for task #191 / PR #92 (`scripts/gbdt/run_agent_loop_resumable.sh`); consult that file for the live state-write semantics this memory summarizes.
