#!/usr/bin/env bash
# Bash-driven tests for scripts/gbdt/run_agent_loop_resumable.sh (task #191).
#
# Drives the wrapper against a tiny mock python child (no real gbdt run —
# the wrapper exposes a WRAPPER_TEST_STUB_CMD hook for this). Each test
# isolates itself under tests/gbdt/tmp/test_<name>_<pid>/ and asserts via
# the atomic-written wrapper.status JSON.
#
# Run directly: bash tests/gbdt/test_run_agent_loop_resumable.sh
# Or via pytest: uv run pytest tests/gbdt/test_run_agent_loop_resumable.py

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WRAPPER="$REPO_ROOT/scripts/gbdt/run_agent_loop_resumable.sh"
TMP_ROOT="$REPO_ROOT/tests/gbdt/tmp"
mkdir -p "$TMP_ROOT"

FAIL_COUNT=0
PASS_COUNT=0
RESULTS=()

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

# Build a unique tmpdir for one test.
make_tmpdir() {
    local name="$1"
    local d
    d="$(mktemp -d "${TMP_ROOT}/test_${name}_XXXXXX")"
    # Ensure parent for --out-dir exists (the wrapper checks dirname(out-dir)).
    mkdir -p "$d/work"
    echo "$d"
}

# Extract a top-level JSON string field (no jq).
status_state() {
    grep -oE '"state"[[:space:]]*:[[:space:]]*"[^"]+"' "$1" 2>/dev/null \
        | head -1 | cut -d'"' -f4
}
status_attempt() {
    grep -oE '"attempt"[[:space:]]*:[[:space:]]*[0-9]+' "$1" 2>/dev/null \
        | head -1 | grep -oE '[0-9]+$'
}

assert_eq() {
    local got="$1" want="$2" label="$3"
    if [[ "$got" == "$want" ]]; then
        return 0
    fi
    echo "  ASSERT FAIL [$label]: got=$got want=$want" >&2
    return 1
}

record_result() {
    local name="$1" ok="$2" detail="$3"
    if [[ "$ok" -eq 0 ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("PASS  $name")
        echo "PASS  $name"
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("FAIL  $name -- $detail")
        echo "FAIL  $name -- $detail" >&2
    fi
}

# Make a stub script that takes ($spec [...rest]) and behaves per its mode.
# Modes:
#   exit_ok               — sleep $sleep, exit 0.
#   exit_fail             — sleep $sleep, exit 1.
#   exit_fail_with_ckpt   — write a fake checkpoint JSON, sleep $sleep, exit 1.
#                           Counts invocations via $invocations_file; max
#                           $fail_count failures then exit 0.
#   heartbeat_stall       — write initial progress.log, then sleep forever.
#   detach_marker         — sleep $sleep, write detach_marker file, exit 0.
make_stub() {
    local mode="$1"
    local tmpdir="$2"
    local stub="$tmpdir/stub_${mode}.sh"
    case "$mode" in
        exit_ok)
            cat > "$stub" <<'STUB'
#!/usr/bin/env bash
# Tail args: $1 = spec path
sleep "${STUB_SLEEP:-1}"
exit 0
STUB
            ;;
        exit_fail)
            cat > "$stub" <<'STUB'
#!/usr/bin/env bash
sleep "${STUB_SLEEP:-1}"
exit 1
STUB
            ;;
        exit_fail_with_ckpt)
            cat > "$stub" <<'STUB'
#!/usr/bin/env bash
# Writes a checkpoint, then exits non-zero up to ${STUB_FAIL_TIMES:-99} times.
# Counts invocations in $STUB_INVOCATIONS_FILE.
mkdir -p "${STUB_CKPT_DIR}"
cat > "${STUB_CKPT_DIR}/checkpoint.json" <<EOF
{
  "schema_version": "v1",
  "run_id": "${STUB_RUN_ID:-mock-run}",
  "iter_idx": 0
}
EOF
count=$(cat "${STUB_INVOCATIONS_FILE}" 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > "${STUB_INVOCATIONS_FILE}"
sleep "${STUB_SLEEP:-1}"
fail_until="${STUB_FAIL_TIMES:-99}"
if [[ "$count" -le "$fail_until" ]]; then
    exit 1
fi
exit 0
STUB
            ;;
        heartbeat_stall)
            cat > "$stub" <<'STUB'
#!/usr/bin/env bash
mkdir -p "${STUB_PROGRESS_DIR}"
printf 'initial heartbeat\n' > "${STUB_PROGRESS_DIR}/progress.log"
# Sleep forever (wrapper's heartbeat watchdog should kill us).
while true; do sleep 1; done
STUB
            ;;
        detach_marker)
            cat > "$stub" <<'STUB'
#!/usr/bin/env bash
sleep "${STUB_SLEEP:-2}"
touch "${STUB_MARKER_FILE}"
exit 0
STUB
            ;;
        *)
            echo "make_stub: unknown mode '$mode'" >&2
            return 1
            ;;
    esac
    chmod +x "$stub"
    echo "$stub"
}

cleanup_tmpdir() {
    # Aggressively kill any lingering children from the test (idempotency
    # check would otherwise persist). Best-effort.
    local d="$1"
    if [[ -f "$d/work/wrapper.pid" ]]; then
        local pgid
        pgid="$(cat "$d/work/wrapper.pid" 2>/dev/null || true)"
        if [[ -n "$pgid" ]] && kill -0 "$pgid" 2>/dev/null; then
            kill -KILL -"$pgid" 2>/dev/null || true
        fi
    fi
}

# ----------------------------------------------------------------------
# Test A — exits_ok: stub exits 0, wrapper writes exited_ok.
# ----------------------------------------------------------------------
test_A_exits_ok() {
    local name="A_exits_ok"
    local d
    d="$(make_tmpdir "$name")"
    local spec="$d/spec.yaml"
    echo "target: {universe: x}" > "$spec"
    local stub
    stub="$(make_stub exit_ok "$d")"
    local out="$d/work"
    local rc=0

    STUB_SLEEP=1 \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
        bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 2 \
            --heartbeat-stall-secs 0 \
        || rc=$?

    local fail=0 detail=""
    if [[ "$rc" -ne 0 ]]; then
        fail=1; detail="wrapper exit=$rc, expected 0"
    elif [[ ! -f "$out/wrapper.status" ]]; then
        fail=1; detail="wrapper.status not written"
    else
        local st; st="$(status_state "$out/wrapper.status")"
        if [[ "$st" != "exited_ok" ]]; then
            fail=1; detail="state=$st, expected exited_ok"
        fi
    fi
    cleanup_tmpdir "$d"
    record_result "$name" "$fail" "$detail"
}

# ----------------------------------------------------------------------
# Test B — exits_fail_no_checkpoint: stub exits 1, no checkpoint, no restart.
# ----------------------------------------------------------------------
test_B_exits_fail_no_checkpoint() {
    local name="B_exits_fail_no_checkpoint"
    local d
    d="$(make_tmpdir "$name")"
    local spec="$d/spec.yaml"
    echo "target: {universe: x}" > "$spec"
    local stub
    stub="$(make_stub exit_fail "$d")"
    local out="$d/work"
    local rc=0

    STUB_SLEEP=1 \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
        bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 3 \
            --heartbeat-stall-secs 0 \
        || rc=$?

    local fail=0 detail=""
    if [[ "$rc" -eq 0 ]]; then
        fail=1; detail="wrapper exit=0, expected non-zero"
    elif [[ ! -f "$out/wrapper.status" ]]; then
        fail=1; detail="wrapper.status not written"
    else
        local st; st="$(status_state "$out/wrapper.status")"
        if [[ "$st" != "exited_failed" ]]; then
            fail=1; detail="state=$st, expected exited_failed"
        fi
        local at; at="$(status_attempt "$out/wrapper.status")"
        if [[ "$at" != "1" ]]; then
            fail=1; detail="$detail; attempt=$at, expected 1 (no restart)"
        fi
    fi
    cleanup_tmpdir "$d"
    record_result "$name" "$fail" "$detail"
}

# ----------------------------------------------------------------------
# Test C — exits_fail_with_checkpoint: max_retries=2 → restart, give up
# after attempt 2 → max_retries_hit.
# ----------------------------------------------------------------------
test_C_exits_fail_with_checkpoint() {
    local name="C_exits_fail_with_checkpoint"
    local d
    d="$(make_tmpdir "$name")"
    local spec="$d/c_spec.yaml"
    echo "target: {universe: x}" > "$spec"
    local stub
    stub="$(make_stub exit_fail_with_ckpt "$d")"
    local out="$d/work"
    local invocations="$d/invocations.count"
    : > "$invocations"
    local rc=0

    STUB_SLEEP=1 \
    STUB_CKPT_DIR="$out/loop" \
    STUB_INVOCATIONS_FILE="$invocations" \
    STUB_FAIL_TIMES=99 \
    STUB_RUN_ID="c_spec" \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
    WRAPPER_RESTART_SLEEP_SECS=1 \
        bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 2 \
            --heartbeat-stall-secs 0 \
        || rc=$?

    local fail=0 detail=""
    if [[ "$rc" -eq 0 ]]; then
        fail=1; detail="wrapper exit=0, expected non-zero"
    elif [[ ! -f "$out/wrapper.status" ]]; then
        fail=1; detail="wrapper.status not written"
    else
        local st; st="$(status_state "$out/wrapper.status")"
        if [[ "$st" != "max_retries_hit" ]]; then
            fail=1; detail="state=$st, expected max_retries_hit"
        fi
        local at; at="$(status_attempt "$out/wrapper.status")"
        if [[ "$at" != "2" ]]; then
            fail=1; detail="$detail; final attempt=$at, expected 2"
        fi
        local n_calls; n_calls="$(cat "$invocations")"
        if [[ "$n_calls" != "2" ]]; then
            fail=1; detail="$detail; stub called $n_calls times, expected 2"
        fi
    fi
    cleanup_tmpdir "$d"
    record_result "$name" "$fail" "$detail"
}

# ----------------------------------------------------------------------
# Test D — heartbeat_stall: progress.log frozen → wrapper kills + restarts.
# Use max_retries=1 so the wrapper does the kill, sees no retry slot,
# emits heartbeat_stalled_killed then exited_failed (no checkpoint here).
# Verify the heartbeat_stalled_killed state appears in wrapper.log even
# if the final state is different (it's a transient state).
# ----------------------------------------------------------------------
test_D_heartbeat_stall() {
    local name="D_heartbeat_stall"
    local d
    d="$(make_tmpdir "$name")"
    local spec="$d/spec.yaml"
    echo "target: {universe: x}" > "$spec"
    local stub
    stub="$(make_stub heartbeat_stall "$d")"
    local out="$d/work"
    local rc=0

    # Background the wrapper so we can enforce an outer timeout if it gets
    # stuck. heartbeat-stall=3s + monitor interval 1s + 30s SIGTERM grace =
    # comfortably under 60s.
    STUB_PROGRESS_DIR="$out/loop" \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
        timeout 60 bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 0 \
            --heartbeat-stall-secs 3 \
        || rc=$?

    local fail=0 detail=""
    if [[ "$rc" -eq 0 ]]; then
        fail=1; detail="wrapper exit=0, expected non-zero (was stall-killed)"
    elif [[ ! -f "$out/wrapper.log" ]]; then
        fail=1; detail="wrapper.log not written"
    else
        if ! grep -q "heartbeat stalled" "$out/wrapper.log"; then
            fail=1; detail="wrapper.log lacks 'heartbeat stalled' line"
        fi
        # Final state should be exited_failed (max_retries=0 → no restart).
        local st; st="$(status_state "$out/wrapper.status")"
        if [[ "$st" != "exited_failed" ]]; then
            fail=1; detail="$detail; state=$st, expected exited_failed"
        fi
    fi
    cleanup_tmpdir "$d"
    record_result "$name" "$fail" "$detail"
}

# ----------------------------------------------------------------------
# Test E — idempotent_double_launch: first wrapper running, second refuses.
# ----------------------------------------------------------------------
test_E_idempotent_double_launch() {
    local name="E_idempotent_double_launch"
    local d
    d="$(make_tmpdir "$name")"
    local spec="$d/spec.yaml"
    echo "target: {universe: x}" > "$spec"
    local stub
    stub="$(make_stub exit_ok "$d")"   # exits 0 after STUB_SLEEP
    local out="$d/work"

    # Launch wrapper #1 in the background with a longish stub sleep.
    STUB_SLEEP=15 \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
        bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 0 \
            --heartbeat-stall-secs 0 \
        >/dev/null 2>&1 &
    local first_wrapper_pid=$!

    # Wait for wrapper.pid to appear and the recorded PID to be alive.
    local waited=0
    while [[ "$waited" -lt 20 ]]; do
        if [[ -f "$out/wrapper.pid" ]]; then
            local child_pid
            child_pid="$(cat "$out/wrapper.pid" 2>/dev/null || true)"
            if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
                break
            fi
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # Now attempt wrapper #2 — should refuse with exit 3.
    local second_rc=0
    STUB_SLEEP=1 \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
        bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 0 \
            --heartbeat-stall-secs 0 \
            >/dev/null 2>&1 || second_rc=$?

    # Wait for wrapper #1 to finish naturally.
    wait "$first_wrapper_pid" 2>/dev/null || true

    local fail=0 detail=""
    if [[ "$second_rc" -ne 3 ]]; then
        fail=1; detail="second wrapper exit=$second_rc, expected 3 (idempotency refusal)"
    fi
    cleanup_tmpdir "$d"
    record_result "$name" "$fail" "$detail"
}

# ----------------------------------------------------------------------
# Test F — setsid_actually_detaches: outer shell exits, child keeps running.
# Use detach_marker stub: sleeps for STUB_SLEEP=3s then touches a marker.
# Outer shell launches wrapper, exits immediately; the marker should appear
# after the outer shell is gone.
# ----------------------------------------------------------------------
test_F_setsid_actually_detaches() {
    local name="F_setsid_actually_detaches"
    local d
    d="$(make_tmpdir "$name")"
    local spec="$d/spec.yaml"
    echo "target: {universe: x}" > "$spec"
    local stub
    stub="$(make_stub detach_marker "$d")"
    local out="$d/work"
    local marker="$d/detach_marker.touched"

    # Outer subshell: launches wrapper in background, then exits. setsid
    # makes the child its own session leader; the wrapper itself is a
    # foreground process here, but the python child it spawns is the one we
    # want to verify survives a parent SIGHUP. To exercise the actual SIGHUP
    # cascade, we'd want to kill the wrapper itself with SIGHUP and confirm
    # the child stub keeps running.
    #
    # So: launch wrapper, capture wrapper PID, SIGHUP it, sleep, then check
    # the stub's marker file was created.

    STUB_SLEEP=4 \
    STUB_MARKER_FILE="$marker" \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
        setsid bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 0 \
            --heartbeat-stall-secs 0 \
            >/dev/null 2>&1 &
    local wrapper_pid=$!

    # Wait for the wrapper to launch its child.
    local waited=0
    while [[ "$waited" -lt 10 ]]; do
        if [[ -f "$out/wrapper.pid" ]]; then
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # SIGHUP the wrapper process (and its session, since we used setsid for
    # the wrapper itself in this test). The python child stub, launched
    # with its OWN setsid, should be in a separate session and survive.
    if kill -0 "$wrapper_pid" 2>/dev/null; then
        kill -HUP -"$wrapper_pid" 2>/dev/null || true
    fi

    # Wait up to 8s for the marker file to be created by the child stub.
    waited=0
    local marker_seen=0
    while [[ "$waited" -lt 8 ]]; do
        if [[ -f "$marker" ]]; then
            marker_seen=1
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # Cleanup any survivors from the child (the wrapper itself should be
    # dead from SIGHUP, but the detached stub keeps running until exit).
    cleanup_tmpdir "$d"

    local fail=0 detail=""
    if [[ "$marker_seen" -ne 1 ]]; then
        fail=1; detail="detach marker not created after parent SIGHUP — child died with parent"
    fi
    record_result "$name" "$fail" "$detail"
}

# ----------------------------------------------------------------------
# Test G — stale_progress_log_not_treated_as_stall: pre-existing progress.log
# from a prior run has mtime far in the past. The watchdog must floor mtime
# at wrapper-start so a stub that exits cleanly (and never writes a heartbeat
# of its own) is NOT mis-killed. Regression test for #193 bug 1.
# ----------------------------------------------------------------------
test_G_stale_progress_log_not_killed() {
    local name="G_stale_progress_log_not_killed"
    local d
    d="$(make_tmpdir "$name")"
    local spec="$d/spec.yaml"
    echo "target: {universe: x}" > "$spec"
    local stub
    stub="$(make_stub exit_ok "$d")"
    local out="$d/work"
    mkdir -p "$out/loop"
    # Pre-existing heartbeat file with mtime ~1 hour ago (well past the 3s
    # stall threshold below). The stub never touches it.
    printf 'stale heartbeat\n' > "$out/loop/progress.log"
    touch -d '1 hour ago' "$out/loop/progress.log"
    local rc=0

    # Stub sleeps 5s then exits 0. heartbeat-stall=10s + monitor=1s means
    # WITHOUT the bug-1 fix the watchdog would SIGTERM within ~1-2s (age was
    # ~3600s vs threshold 10s) and the wrapper would record
    # heartbeat_stalled_killed. With the fix, mtime is floored at
    # wrapper-start so age never exceeds 5s; child exits cleanly.
    STUB_SLEEP=5 \
    WRAPPER_TEST_STUB_CMD="$stub" \
    WRAPPER_TEST_SKIP_DISK_CHECK=1 \
    WRAPPER_MONITOR_INTERVAL_SECS=1 \
        timeout 30 bash "$WRAPPER" \
            --spec "$spec" \
            --out-dir "$out" \
            --max-retries 0 \
            --heartbeat-stall-secs 10 \
        || rc=$?

    local fail=0 detail=""
    if [[ "$rc" -ne 0 ]]; then
        fail=1; detail="wrapper exit=$rc, expected 0 (child should have exited cleanly)"
    elif [[ ! -f "$out/wrapper.log" ]]; then
        fail=1; detail="wrapper.log not written"
    elif grep -q "heartbeat stalled" "$out/wrapper.log"; then
        fail=1; detail="wrapper killed child on stale-mtime — bug-1 fix regressed"
    else
        local st; st="$(status_state "$out/wrapper.status")"
        if [[ "$st" != "exited_ok" ]]; then
            fail=1; detail="state=$st, expected exited_ok"
        fi
    fi
    cleanup_tmpdir "$d"
    record_result "$name" "$fail" "$detail"
}

# ----------------------------------------------------------------------
# Run all tests
# ----------------------------------------------------------------------

echo "==== run_agent_loop_resumable.sh tests ===="
echo "Wrapper: $WRAPPER"
echo "Tmp root: $TMP_ROOT"
echo

test_A_exits_ok
test_B_exits_fail_no_checkpoint
test_C_exits_fail_with_checkpoint
test_D_heartbeat_stall
test_E_idempotent_double_launch
test_F_setsid_actually_detaches
test_G_stale_progress_log_not_killed

echo
echo "==== Summary ===="
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"

if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
fi
exit 0
