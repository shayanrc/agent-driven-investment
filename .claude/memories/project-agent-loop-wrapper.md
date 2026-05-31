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
- **Idempotency guard**: refuses to double-launch against the same `--out-dir` while a prior PGID (recorded in `<out-dir>/wrapper.pid`) is still alive. Prevents the duplicate-launch foot-gun in `[[feedback-process-death-verification]]`.
- **Atomic 7-state `wrapper.status` JSON**: `starting | running | resuming | exited_ok | exited_failed | stalled_killed | gave_up_retries`. Written via temp+rename so a partial read never returns junk; safe to poll from a Monitor loop.

**When to use:**

- Any gbdt experiment expected to run ≥ 30 minutes (cold-features build alone is ~3 h on r1k).
- Every sweep cell — sweep script can still chain cells sequentially, but each cell is launched through the wrapper for that cell's `--out-dir`.
- Any manual experiment kicked off from a sub-agent session — the sub-agent's lifetime should never gate the python's lifetime.

**Honest caveat — cold-features build is unprotected.** The checkpoint is written by the V1.1 loop *after* iteration 0 completes. If the process dies during the cold-features build (the ~3 h phase BEFORE iteration 0), no checkpoint exists, `--resume` isn't possible, and the wrapper exits `exited_failed` after the first crash. Closing this gap requires chunked features build (write intermediate state every family, resume from last completed family) — a separate, larger change. For now: treat the cold-features build as a single atomic phase, accept that an OOM / disk wedge during it costs the full build.

**Reference invocation:**

```bash
bash scripts/gbdt/run_agent_loop_resumable.sh \
  --spec configs/gbdt/experiments/<cell>.yaml \
  --out-dir results/gbdt/experiments/<cell> \
  --data-root /mnt/122CEE982CEE765F/cache_data \
  [--max-retries 3] \
  [--heartbeat-stall-secs 1800]
```

**Why this exists (canonical incident):** 2026-05-31 r1k agent-loop trio loss. Three cells (nasdaq100/sp500/r1k at H=25) all launched as foreground `uv run python -m gbdt experiment ...` from sub-agent sessions. The sub-agent sessions terminated (rate-limit + watchdog) and — what I incorrectly believed at the time — SIGHUP'd their children. Combined with my misdiagnosis cascade (described in `[[feedback-process-death-verification]]`), the trio was rerun unnecessarily, costing ~15 h of cold-rebuild CPU. PRs #188 (post-mortem) and #191 (this wrapper) hardened against the SIGHUP-cascade and the dying-mid-run failure modes. The other half of the lesson — *verify before relaunching* — is `[[feedback-process-death-verification]]`.

**How to apply:** when writing a sub-agent brief for long-running gbdt work, instruct the sub-agent to invoke through this wrapper — never bare `uv run python -m gbdt experiment ...`. State the `--out-dir` explicitly so the wrapper's idempotency guard can do its job. For status checks, parse `wrapper.status` via `[[feedback-subagent-transcript-parsing]]`-style bounded queries; for liveness checks before any relaunch decision, follow `[[feedback-process-death-verification]]`. See `[[project-gbdt-tuning-playbook]]` for the loop semantics this wrapper hosts and `[[feedback-experiment-agent-loop]]` for the launch-side patterns.
