#!/usr/bin/env bash
# run_agent_loop_resumable.sh (task #191)
#
# Bash wrapper for `uv run python -m gbdt experiment <spec>` long-running
# agent-driven FS+HP loop runs. Two surgical guarantees on top of the bare
# `uv run python -m gbdt ...` invocation:
#
#   (a) SIGHUP detachment — the python child runs under `setsid nohup`, so a
#       SIGHUP delivered to the calling shell (e.g. when a wrapping sub-agent
#       is terminated by a user-tier rate-limit) does NOT cascade to the
#       python process. This is the failure that lost the #188 r1k validation
#       trio on 2026-05-31.
#
#   (b) auto-restart-on-death via --resume — when the python child exits
#       non-zero, the wrapper checks for a checkpoint at
#       `<out-dir>/loop/checkpoint.json` (the V1.1 Phase 2 exit-and-resume
#       location) and, if present, relaunches with `--resume <run_id>`.
#       Capped at --max-retries attempts; falls through to fail-fast when
#       no checkpoint exists (resume isn't possible from a cold-features
#       crash).
#
# Plus a heartbeat-stall watchdog: if `<out-dir>/loop/progress.log` mtime
# goes stale by > --heartbeat-stall-secs, the wrapper SIGTERMs the process
# group (recoverable from the setsid pgid), waits 30s, SIGKILLs, then takes
# the restart path. Default 30 min covers the slowest legitimate phase
# (cold-features build on r1k ~ 3h is one heartbeat-emitting phase but
# heartbeat fires regardless of progress, so a 30-min stall is unambiguous).
#
# State is exposed via:
#   <out-dir>/wrapper.pid     — child PGID (kill -TERM -$(cat ...) targets it)
#   <out-dir>/wrapper.status  — atomic-written JSON with current state
#   <out-dir>/wrapper.log     — append-only wrapper + child stdout/stderr
#
# Idempotent on relaunch: if a prior wrapper for the same --out-dir is still
# alive, the second invocation refuses with a clear error.

set -euo pipefail

# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: run_agent_loop_resumable.sh --spec <path> --out-dir <path> [OPTIONS] [-- EXTRA_ARGS]

Required:
  --spec <path>                  Path to the gbdt experiment YAML spec.
  --out-dir <path>               Artifact dir for this cell. The wrapper writes
                                 wrapper.pid / wrapper.status / wrapper.log here
                                 alongside the experiment's loop/, predictions/,
                                 metrics.json, etc.

Optional:
  --data-root <path>             Passed through to `python -m gbdt` as the
                                 env var GBDT_DATA_ROOT (currently informational
                                 only — the runner reads `data/` from cwd).
  --callback-mode <mode>         Passed through as --callback-mode <mode>.
                                 Typically "agent_file_protocol" for the
                                 exit-and-resume loop this wrapper exists for.
  --max-retries N                Max auto-restart attempts after a non-zero
                                 exit + checkpoint present. Default 3.
                                 N=0 disables auto-restart entirely (useful
                                 for fail-fast debugging — a single failed
                                 attempt immediately exits non-zero).
  --heartbeat-stall-secs N       Kill + restart if progress.log mtime is
                                 older than N seconds. Default 1800 (30min).
                                 Set 0 to disable the watchdog.
  --log-file <path>              Wrapper + child output log. Default
                                 <out-dir>/wrapper.log.
  --run-id <id>                  Override the run-id used for --resume.
                                 Default is basename(spec without .yaml).
  --no-overwrite                 Do NOT pass --overwrite to the runner on the
                                 first launch. Default: also do not pass it
                                 (the runner's default is no-overwrite); this
                                 flag is accepted for documentation symmetry.
  --overwrite                    Pass --overwrite to the runner on the first
                                 launch (resume launches NEVER carry it).
  -h, --help                     Show this help and exit 0.

Extra args after `--` are passed verbatim to `python -m gbdt experiment <spec>`.

Examples:
  scripts/gbdt/run_agent_loop_resumable.sh \
      --spec configs/gbdt/experiments/r1k_up_3pct_25d_dd5pct.yaml \
      --out-dir results/gbdt/experiments/r1k_up_3pct_25d_dd5pct \
      --callback-mode agent_file_protocol \
      --max-retries 5

  # Fail-fast (no auto-restart, surfaces errors immediately):
  scripts/gbdt/run_agent_loop_resumable.sh \
      --spec ... --out-dir ... --max-retries 0
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

SPEC=""
OUT_DIR=""
DATA_ROOT=""
CALLBACK_MODE=""
MAX_RETRIES=3
HEARTBEAT_STALL_SECS=1800
LOG_FILE=""
RUN_ID_OVERRIDE=""
OVERWRITE_FLAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spec)                 SPEC="$2";                 shift 2 ;;
        --out-dir)              OUT_DIR="$2";              shift 2 ;;
        --data-root)            DATA_ROOT="$2";            shift 2 ;;
        --callback-mode)        CALLBACK_MODE="$2";        shift 2 ;;
        --max-retries)          MAX_RETRIES="$2";          shift 2 ;;
        --heartbeat-stall-secs) HEARTBEAT_STALL_SECS="$2"; shift 2 ;;
        --log-file)             LOG_FILE="$2";             shift 2 ;;
        --run-id)               RUN_ID_OVERRIDE="$2";      shift 2 ;;
        --no-overwrite)         OVERWRITE_FLAG="";         shift 1 ;;
        --overwrite)            OVERWRITE_FLAG="--overwrite"; shift 1 ;;
        -h|--help)              usage; exit 0 ;;
        --)                     shift; while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done ;;
        *)                      echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$SPEC" || -z "$OUT_DIR" ]]; then
    echo "ERROR: --spec and --out-dir are required" >&2
    usage >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

if [[ ! -f "$SPEC" ]]; then
    echo "ERROR: --spec path does not exist or is not a file: $SPEC" >&2
    exit 2
fi

OUT_DIR_PARENT="$(dirname "$OUT_DIR")"
if [[ ! -d "$OUT_DIR_PARENT" ]]; then
    echo "ERROR: --out-dir parent does not exist: $OUT_DIR_PARENT" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

# Disk pre-flight (CLAUDE.md feedback-disk-wedge-pattern): wedge risk ≥ 95% full.
# 10 GB free is the project-wide minimum for long-running gbdt runs.
# WRAPPER_TEST_SKIP_DISK_CHECK=1 bypasses (for tests on small tmpfs partitions).
if [[ -z "${WRAPPER_TEST_SKIP_DISK_CHECK:-}" ]]; then
    DISK_AVAIL_K="$(df --output=avail "$OUT_DIR" | tail -1 | tr -d ' ')"
    DISK_AVAIL_GB=$(( DISK_AVAIL_K / 1024 / 1024 ))
    if [[ "$DISK_AVAIL_GB" -lt 10 ]]; then
        echo "ERROR: disk pre-flight failed: only ${DISK_AVAIL_GB} GB free at $OUT_DIR (need >= 10 GB)" >&2
        exit 2
    fi
fi

if [[ -z "${WRAPPER_TEST_STUB_CMD:-}" ]] && ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv not on PATH — install uv or activate the project's environment" >&2
    exit 2
fi

if ! command -v setsid >/dev/null 2>&1; then
    echo "ERROR: setsid not on PATH — required for SIGHUP detachment" >&2
    exit 2
fi

# Validate MAX_RETRIES is a non-negative integer.
if ! [[ "$MAX_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --max-retries must be a non-negative integer, got: $MAX_RETRIES" >&2
    exit 2
fi
if ! [[ "$HEARTBEAT_STALL_SECS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --heartbeat-stall-secs must be a non-negative integer, got: $HEARTBEAT_STALL_SECS" >&2
    exit 2
fi

# Defaults that depend on parsed values.
if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="$OUT_DIR/wrapper.log"
fi

# RUN_ID derives from the spec filename stem (matches gbdt.__main__'s
# `name = spec_path.stem`, which is what --resume expects).
if [[ -n "$RUN_ID_OVERRIDE" ]]; then
    RUN_ID="$RUN_ID_OVERRIDE"
else
    RUN_ID="$(basename "$SPEC")"
    RUN_ID="${RUN_ID%.yaml}"
    RUN_ID="${RUN_ID%.yml}"
fi

PID_FILE="$OUT_DIR/wrapper.pid"
STATUS_FILE="$OUT_DIR/wrapper.status"
PROGRESS_LOG="$OUT_DIR/loop/progress.log"
CHECKPOINT_PRIMARY="$OUT_DIR/loop/checkpoint.json"
CHECKPOINT_FALLBACK="$OUT_DIR/checkpoint.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

iso_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

log_line() {
    # Append a timestamped line to the wrapper log. Never crash.
    local msg="$1"
    printf '[%s] [wrapper] %s\n' "$(iso_now)" "$msg" >> "$LOG_FILE" 2>/dev/null || true
}

# Read run_id from checkpoint.json without jq — minimal grep+cut. Falls back
# to the spec-stem default if the checkpoint does not carry a run_id field.
read_run_id_from_checkpoint() {
    local ckpt
    if [[ -f "$CHECKPOINT_PRIMARY" ]]; then
        ckpt="$CHECKPOINT_PRIMARY"
    elif [[ -f "$CHECKPOINT_FALLBACK" ]]; then
        ckpt="$CHECKPOINT_FALLBACK"
    else
        return 1
    fi
    local id
    id="$(grep -oE '"run_id"[[:space:]]*:[[:space:]]*"[^"]+"' "$ckpt" 2>/dev/null \
            | head -1 | cut -d'"' -f4)"
    if [[ -n "$id" ]]; then
        printf '%s\n' "$id"
        return 0
    fi
    return 1
}

checkpoint_exists() {
    [[ -f "$CHECKPOINT_PRIMARY" ]] || [[ -f "$CHECKPOINT_FALLBACK" ]]
}

# Atomic status write — temp + mv (rename). Survives partial writes; readers
# either see the prior state or the new one, never a half-flushed JSON. No jq
# dependency; we hand-build the JSON.
write_status() {
    local state="$1"
    local pid="$2"
    local attempt="$3"
    local started_at="$4"
    local last_event_at
    last_event_at="$(iso_now)"
    local tmp="${STATUS_FILE}.tmp.$$"
    cat > "$tmp" <<EOF
{
  "state": "${state}",
  "pid": ${pid},
  "attempt": ${attempt},
  "max_retries": ${MAX_RETRIES},
  "started_at": "${started_at}",
  "last_event_at": "${last_event_at}",
  "out_dir": "${OUT_DIR}",
  "spec": "${SPEC}",
  "run_id": "${RUN_ID}"
}
EOF
    mv -f "$tmp" "$STATUS_FILE"
    log_line "state=${state} pid=${pid} attempt=${attempt}/${MAX_RETRIES}"
}

# Idempotency guard: refuse to start a second wrapper for the same out-dir
# if a prior wrapper's recorded PGID is alive AND its status is "running"
# (or "starting" / "restarting"). The user can force-kill via
# `kill -TERM -$(cat .../wrapper.pid)`.
check_idempotent() {
    if [[ ! -f "$STATUS_FILE" ]] || [[ ! -f "$PID_FILE" ]]; then
        return 0
    fi
    local prior_state prior_pid
    prior_state="$(grep -oE '"state"[[:space:]]*:[[:space:]]*"[^"]+"' "$STATUS_FILE" 2>/dev/null \
                    | head -1 | cut -d'"' -f4)"
    prior_pid="$(cat "$PID_FILE" 2>/dev/null | tr -d ' \n')"
    case "$prior_state" in
        starting|running|restarting)
            if [[ -n "$prior_pid" ]] && kill -0 "$prior_pid" 2>/dev/null; then
                cat >&2 <<EOF
ERROR: wrapper already running for out_dir=${OUT_DIR}
  prior state: ${prior_state}
  prior PID (PGID): ${prior_pid}
  status file: ${STATUS_FILE}
Nothing done. To stop the prior run, send SIGTERM to its process group:
  kill -TERM -${prior_pid}
EOF
                exit 3
            fi
            ;;
    esac
    return 0
}

# Launch the python child under setsid+nohup. Sets the global LAUNCHED_PID
# to the new process's PID/PGID. NOT invoked via $(...) because background
# jobs spawned in a command-substitution subshell don't appear in the outer
# shell's jobs table, so `wait $pid` in the outer shell would fail to harvest
# the exit code.
launch_child() {
    local resume_flag="$1"  # empty for fresh launch, "--resume <run_id>" otherwise

    # Build the command line as an array so quoting survives.
    # Test hook: WRAPPER_TEST_STUB_CMD lets tests/gbdt/test_run_agent_loop_resumable.sh
    # substitute a tiny mock child for `uv run python -m gbdt experiment`. The
    # stub receives the spec path as $1 and any subsequent flags verbatim,
    # mimicking the real CLI surface. Production runs leave WRAPPER_TEST_STUB_CMD
    # unset and get the real command.
    local -a cmd
    if [[ -n "${WRAPPER_TEST_STUB_CMD:-}" ]]; then
        # Word-split the stub command string into argv; preserves quoting only
        # for simple cases (good enough for tests, which pass a single script
        # path with no spaces).
        # shellcheck disable=SC2206
        local -a stub_arr=(${WRAPPER_TEST_STUB_CMD})
        cmd=("${stub_arr[@]}" "$SPEC")
    else
        cmd=(uv run python -m gbdt experiment "$SPEC")
    fi
    if [[ -n "$OVERWRITE_FLAG" && -z "$resume_flag" ]]; then
        cmd+=("$OVERWRITE_FLAG")
    fi
    if [[ -n "$CALLBACK_MODE" ]]; then
        cmd+=(--callback-mode "$CALLBACK_MODE")
    fi
    if [[ -n "$resume_flag" ]]; then
        # resume_flag is "--resume <run_id>" — split safely.
        # shellcheck disable=SC2206
        local -a resume_arr=($resume_flag)
        cmd+=("${resume_arr[@]}")
    fi
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    log_line "launching: ${cmd[*]}"
    # Optional env passthrough (informational; runner currently reads ./data).
    local -a env_prefix
    if [[ -n "$DATA_ROOT" ]]; then
        env_prefix=(env "GBDT_DATA_ROOT=$DATA_ROOT")
    else
        env_prefix=()
    fi

    # setsid + nohup detaches from the parent's session AND HUP signal. `&`
    # backgrounds; $! is the PID of `setsid`, which (because setsid makes a
    # new session) is also the PGID. Targeting -PGID kills the whole tree.
    setsid nohup "${env_prefix[@]}" "${cmd[@]}" >> "$LOG_FILE" 2>&1 &
    LAUNCHED_PID=$!
    echo "$LAUNCHED_PID" > "$PID_FILE"
}
LAUNCHED_PID=0

# Send SIGTERM to the process group, wait up to 30s, then SIGKILL if alive.
kill_process_group() {
    local pid="$1"
    if ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    log_line "SIGTERM -${pid} (process group)"
    kill -TERM -"$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    log_line "SIGKILL -${pid} (group did not respond to SIGTERM in 30s)"
    kill -KILL -"$pid" 2>/dev/null || true
    sleep 1
    return 0
}

# Monitor a running PID: poll alive + heartbeat-stall every MONITOR_INTERVAL.
# Returns the child's exit code in the global MONITOR_EXIT_CODE; sets
# MONITOR_KILLED_STALLED=1 if the watchdog fired.
MONITOR_INTERVAL=${WRAPPER_MONITOR_INTERVAL_SECS:-30}
MONITOR_EXIT_CODE=0
MONITOR_KILLED_STALLED=0

monitor_child() {
    local pid="$1"
    MONITOR_EXIT_CODE=0
    MONITOR_KILLED_STALLED=0
    while true; do
        if ! kill -0 "$pid" 2>/dev/null; then
            # Child exited; harvest exit code via `wait`. `wait` only works
            # for direct children; setsid+nohup+& still makes it our direct
            # child of the wrapper shell. If wait returns 127 ("not a child")
            # we couldn't capture — record 0 to avoid spurious "non-zero".
            if wait "$pid" 2>/dev/null; then
                MONITOR_EXIT_CODE=0
            else
                MONITOR_EXIT_CODE=$?
                if [[ "$MONITOR_EXIT_CODE" -eq 127 ]]; then
                    MONITOR_EXIT_CODE=0
                fi
            fi
            return 0
        fi
        # Heartbeat-stall watchdog.
        if [[ "$HEARTBEAT_STALL_SECS" -gt 0 && -f "$PROGRESS_LOG" ]]; then
            local mtime now age
            mtime="$(stat -c %Y "$PROGRESS_LOG" 2>/dev/null || echo 0)"
            now="$(date +%s)"
            age=$(( now - mtime ))
            if [[ "$age" -gt "$HEARTBEAT_STALL_SECS" ]]; then
                log_line "heartbeat stalled: progress.log age=${age}s > threshold=${HEARTBEAT_STALL_SECS}s"
                kill_process_group "$pid"
                MONITOR_KILLED_STALLED=1
                MONITOR_EXIT_CODE=143  # 128 + SIGTERM
                return 0
            fi
        fi
        sleep "$MONITOR_INTERVAL"
    done
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

check_idempotent

mkdir -p "$(dirname "$LOG_FILE")"
: > "$LOG_FILE.startmark" 2>/dev/null || true
rm -f "$LOG_FILE.startmark"

STARTED_AT="$(iso_now)"
log_line "==== wrapper start ===="
log_line "spec=$SPEC"
log_line "out_dir=$OUT_DIR"
log_line "run_id=$RUN_ID"
log_line "max_retries=$MAX_RETRIES heartbeat_stall_secs=$HEARTBEAT_STALL_SECS"

attempt=1
exit_code=0

while true; do
    if [[ "$attempt" -eq 1 ]]; then
        write_status "starting" 0 "$attempt" "$STARTED_AT"
        launch_child ""
    else
        write_status "restarting" 0 "$attempt" "$STARTED_AT"
        # On restart, prefer the run_id from the checkpoint (if present);
        # fall back to the spec-stem default. The checkpoint's run_id is
        # canonical — it's what the paused run wrote when it created the
        # checkpoint, and what `--resume` expects.
        ckpt_run_id="$(read_run_id_from_checkpoint || true)"
        if [[ -z "$ckpt_run_id" ]]; then
            ckpt_run_id="$RUN_ID"
        fi
        log_line "restart attempt ${attempt}/${MAX_RETRIES} with --resume ${ckpt_run_id}"
        # Brief sleep so the child has time to flush + filesystem settles.
        sleep "${WRAPPER_RESTART_SLEEP_SECS:-10}"
        launch_child "--resume ${ckpt_run_id}"
    fi
    CHILD_PID="$LAUNCHED_PID"

    write_status "running" "$CHILD_PID" "$attempt" "$STARTED_AT"
    monitor_child "$CHILD_PID"
    exit_code=$MONITOR_EXIT_CODE

    if [[ "$MONITOR_KILLED_STALLED" -eq 1 ]]; then
        write_status "heartbeat_stalled_killed" "$CHILD_PID" "$attempt" "$STARTED_AT"
        # Treat as a recoverable failure: take the restart path if a
        # checkpoint is present and we have retries left.
    fi

    if [[ "$exit_code" -eq 0 ]]; then
        write_status "exited_ok" "$CHILD_PID" "$attempt" "$STARTED_AT"
        log_line "child exited cleanly (exit 0) — done"
        exit 0
    fi

    log_line "child exited non-zero (exit $exit_code) on attempt ${attempt}/${MAX_RETRIES}"

    # Auto-restart eligibility:
    #   1. attempts must remain (attempt < MAX_RETRIES)
    #      — MAX_RETRIES=0 means "never restart"; the loop exits after the
    #        first failed attempt regardless of checkpoint state.
    #   2. a checkpoint must exist (resume isn't meaningful without one;
    #      cold-features-build crashes leave no checkpoint).
    if [[ "$MAX_RETRIES" -eq 0 ]]; then
        write_status "exited_failed" "$CHILD_PID" "$attempt" "$STARTED_AT"
        log_line "max_retries=0 — no auto-restart attempted; exiting"
        exit "$exit_code"
    fi

    if ! checkpoint_exists; then
        write_status "exited_failed" "$CHILD_PID" "$attempt" "$STARTED_AT"
        log_line "no checkpoint at $CHECKPOINT_PRIMARY (or $CHECKPOINT_FALLBACK) — cannot resume; exiting"
        exit "$exit_code"
    fi

    if [[ "$attempt" -ge "$MAX_RETRIES" ]]; then
        write_status "max_retries_hit" "$CHILD_PID" "$attempt" "$STARTED_AT"
        log_line "max_retries=$MAX_RETRIES exhausted — giving up"
        exit "$exit_code"
    fi

    attempt=$(( attempt + 1 ))
    # Loop back to relaunch.
done
