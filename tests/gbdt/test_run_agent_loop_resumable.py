"""Pytest entry that delegates to the bash test driver.

The actual assertions live in ``tests/gbdt/test_run_agent_loop_resumable.sh``
because the wrapper under test is itself a bash script and the test surface
(signal handling, process groups, file mtimes, atomic JSON writes) is easier
to exercise from bash than to mock from Python. This entry just shells out
so ``uv run pytest`` picks the suite up alongside everything else.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SCRIPT = REPO_ROOT / "tests" / "gbdt" / "test_run_agent_loop_resumable.sh"
WRAPPER = REPO_ROOT / "scripts" / "gbdt" / "run_agent_loop_resumable.sh"


@pytest.mark.skipif(
    shutil.which("setsid") is None,
    reason="setsid not on PATH — wrapper requires it",
)
def test_run_agent_loop_resumable_bash_suite():
    """Run the bash test driver; pass iff all six tests pass."""
    assert TEST_SCRIPT.exists(), f"missing test script: {TEST_SCRIPT}"
    assert WRAPPER.exists(), f"missing wrapper script: {WRAPPER}"
    proc = subprocess.run(
        ["bash", str(TEST_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Print captured streams so pytest -s / a failure shows the per-test result.
    print("---- bash stdout ----")
    print(proc.stdout)
    print("---- bash stderr ----")
    print(proc.stderr)
    assert proc.returncode == 0, (
        f"bash test driver failed with exit {proc.returncode}; "
        f"see captured output above"
    )
