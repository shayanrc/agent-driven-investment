---
name: feedback-process-death-verification
description: When a wrapping sub-agent's session ends, the child python process often survives — verify with `ps -ef` before declaring it dead and relaunching
metadata:
  type: feedback
---

Before declaring a long-running child process dead and relaunching it, **verify it's actually dead with `ps -ef | grep <command-pattern>` or `pgrep -af <pattern>`**. Sub-agent session termination is NOT a reliable signal that the child process died.

**Why:** 2026-05-31 sp500 sweep. The wrapping sub-agent's monitoring session ended (rate-limit + watchdog combined) and I concluded — incorrectly — that the python sweep process had died with it. I launched a duplicate via `setsid` from the main session, then discovered the original (PID 139260) was still running at 98% CPU, untouched. Had to identify and kill the duplicate (PIDs 139923 / 139936 / 139939) and clean its partial cell 2 directory. Cost ~5 minutes of wall-clock plus the wasted duplicate-build CPU. The earlier r1k agent-loop trio death the same day (different mechanism — covered in `[[project-agent-loop-wrapper]]`) had seeded the false intuition that sub-agent death = child death.

**How to apply:** at any moment you suspect a long-running gbdt or sub-agent-launched process has died:

1. **Get ground truth first**: `ps -ef | grep -E "python.*gbdt|run_.*sweep" | grep -v grep` (or `pgrep -af gbdt`). This is the only authoritative signal.
2. **Check progress / heartbeat secondarily**: `<out-dir>/loop/progress.log` mtime, `wrapper.status` JSON, recent rows in the predictions parquet — these confirm the process is also *making progress*, not just alive-but-stuck. (A process with PID 139260 holding 98% CPU is definitely alive; one at 0% CPU with stale heartbeat is alive-but-stalled and is the watchdog's job, not yours.)
3. **ONLY if confirmed dead by step 1**, consider relaunch. Use `[[project-agent-loop-wrapper]]` (the wrapper) for the relaunch — its idempotency guard would have rejected the duplicate launch in the canonical incident if it had been in use on the original launch. The wrapper plus `wrapper.status` plus `wrapper.pid` files together make this verification step unambiguous.
4. **Mechanism note — be skeptical of cascading-SIGHUP theories.** The intuition that "sub-agent dies → SIGHUPs its python child → python dies" is WEAKER than it feels. Sub-agents in this environment do NOT routinely SIGHUP their children when their session ends (otherwise the sp500 case above couldn't have happened — the sub-agent ended and the python kept running). The r1k trio death was more likely an OOM (three parallel feature rebuilds at ~6.4 GB RSS each on finite memory) or a transient disk-side issue, not a SIGHUP cascade. Don't generalize from one incident; check `dmesg`, `journalctl -k`, and the OOM-killer log before settling on a cause story.

**Don't apply when:** the process you're asking about is genuinely short-lived and you've already verified termination via its expected exit notification (e.g. the harness telling you a background bash task completed, with a real exit code). In that case the notification IS the ground truth.

See `[[project-agent-loop-wrapper]]` for the wrapper that makes this verification step easier (and that you should use for the relaunch when one is genuinely needed), `[[feedback-agent-pkill-antipattern]]` for why `pkill -f <pattern>` is the wrong tool for the cleanup half, and `[[feedback-disk-wedge-pattern]]` for one of the real mechanisms that can kill a long-running process (cascading IO wedge, not SIGHUP).
