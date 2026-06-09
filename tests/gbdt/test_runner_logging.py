"""Runner-logging tests:

A) Throttled features-stage progress lines emitted from
   :func:`gbdt.features.build_feature_matrix`.
B) Pre-flight cache + code fingerprint emitted from
   :func:`gbdt.__main__.run_experiment` and persisted into
   ``metrics.json::preflight``.

Both improvements live in a single PR; this module covers both so the
test plan in the PR body maps to one file.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gbdt import features as F
from gbdt.__main__ import (
    _collect_preflight,
    _format_preflight_line,
    _sanitize_path_for_emission,
    _sanitize_preflight_for_emission,
)
from gbdt.leakage_harness import make_synthetic_panel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _synth_panel_with_index(n_rows: int = 250, n_tickers: int = 2, seed: int = 0):
    panel = make_synthetic_panel(n_rows, n_tickers, seed=seed)
    rng = np.random.default_rng(seed + 99)
    dates = panel.index.get_level_values("date").unique()
    rets = rng.normal(0, 0.008, size=len(dates))
    close = 1000.0 * np.exp(np.cumsum(rets))
    high = close * (1.0 + rng.uniform(0.0, 0.005, len(dates)))
    low = close * (1.0 - rng.uniform(0.0, 0.005, len(dates)))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, len(dates)))
    index_df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": np.ones(len(dates))},
        index=pd.Index(dates, name="date"),
    )
    return panel, index_df


# ---------------------------------------------------------------------------
# A) features-stage progress logging
# ---------------------------------------------------------------------------


_PROGRESS_RE = re.compile(
    r"^\[features\] family=(\S+) step=(\d+)/(\d+) elapsed=([\d.]+)s$"
)


def _extract_progress_lines(captured_stderr: str) -> list[tuple[str, int, int, float]]:
    out = []
    for line in captured_stderr.splitlines():
        m = _PROGRESS_RE.match(line.strip())
        if m:
            out.append((m.group(1), int(m.group(2)), int(m.group(3)),
                        float(m.group(4))))
    return out


def test_A1_features_logger_emits_at_least_one_line(capsys):
    """A1: every run produces at least one progress line, even for
    short/fast feature builds (smoketest panels included)."""
    panel, index_df = _synth_panel_with_index(n_rows=220, n_tickers=2)
    # Subset of families so the build is fast; logging contract holds
    # regardless of the family count.
    X = F.build_feature_matrix(
        panel, index_df,
        lookbacks=(5, 10),
        families=["F2", "F4", "F12"],
    )
    assert X.shape[0] == len(panel)
    captured = capsys.readouterr()
    lines = _extract_progress_lines(captured.err)
    assert len(lines) >= 1, f"expected ≥1 progress line, got: {captured.err!r}"
    # The single forced-final line should reference the last family.
    fam, step, total, _elapsed = lines[-1]
    assert step == total, "final line should report step == total"
    assert total == 3, f"plan had 3 families, got total={total}"


def test_A2_features_logger_throttles_under_30s(capsys):
    """A2: when every family completes instantaneously the logger emits
    AT MOST one line — the forced-final emit — never multiple lines."""
    panel, index_df = _synth_panel_with_index(n_rows=200, n_tickers=2)
    # Use the full default family pool so the throttle has many
    # opportunities to fire; on a small synthetic panel everything runs
    # in well under 30s.
    X = F.build_feature_matrix(
        panel, index_df,
        lookbacks=(5, 10),
        families=["F2", "F4", "F8", "F12", "F13", "F15"],
    )
    assert X.shape[0] == len(panel)
    captured = capsys.readouterr()
    lines = _extract_progress_lines(captured.err)
    # Throttle bound: at most one line per 30s, plus a single forced
    # final emit when no line fired yet. On a synthetic ≤1s build that
    # means exactly one line.
    assert len(lines) == 1, (
        f"throttle violated: expected exactly 1 line on instant build, "
        f"got {len(lines)} lines: {captured.err!r}"
    )


def test_A2b_throttle_constant_is_30s():
    """Sanity check: the throttle interval constant is the documented
    30-second value (not silently changed)."""
    assert F._FEATURES_PROGRESS_THROTTLE_SEC == 30.0


# ---------------------------------------------------------------------------
# B) preflight log line + metrics.json persistence
# ---------------------------------------------------------------------------


_PREFLIGHT_FIELDS = (
    "cache_db", "cache_db_size", "cache_db_mtime",
    "data_root", "code_commit", "code_dirty",
)


def test_B1_preflight_log_format(tmp_path, capsys):
    """B1: log line emits all six fields with the documented key=value
    format. (Order matters for downstream log scrapers.)"""
    # Create a fake data root with a processed.db so the size/mtime
    # fields are populated.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db = data_dir / "processed.db"
    db.write_bytes(b"\x00" * 4096)

    pf = _collect_preflight(tmp_path)
    line = _format_preflight_line(pf, tmp_path)
    print(line)  # exercise the print path
    captured = capsys.readouterr()
    assert line in captured.out
    # Paths in the emitted line are sanitized to repo-relative (no leak of
    # the absolute realpath into committed log files).
    assert "data_root=data" in line
    assert "cache_db=data/processed.db" in line

    # Documented key order (matches the runner's log line).
    expected_keys = list(_PREFLIGHT_FIELDS)
    found_keys = [m.group(1) for m in re.finditer(r"(\w+)=", line)]
    assert found_keys[: len(expected_keys)] == expected_keys, (
        f"preflight line key order drift: got {found_keys}"
    )
    assert line.startswith("[preflight] ")


def test_B2_preflight_dict_has_all_six_fields(tmp_path):
    """B2: ``metrics.json::preflight`` is populated with all six fields."""
    pf = _collect_preflight(tmp_path)
    assert set(pf.keys()) == set(_PREFLIGHT_FIELDS)
    # JSON-roundtrippable — what gets persisted.
    blob = json.dumps(pf)
    rt = json.loads(blob)
    assert rt == pf


def test_B3_preflight_git_unavailable_graceful_fallback(tmp_path):
    """B3: when ``git`` returns non-zero (no .git dir / git missing)
    the preflight still emits with ``code_commit='unknown'`` and
    ``code_dirty=False``."""
    # tmp_path is not a git repo. Running git there returns non-zero.
    assert not (tmp_path / ".git").exists()
    pf = _collect_preflight(tmp_path)
    assert pf["code_commit"] == "unknown"
    assert pf["code_dirty"] is False


def test_B3b_preflight_git_binary_missing_graceful_fallback(tmp_path):
    """B3 variant: subprocess.run raising ``FileNotFoundError`` (git
    binary missing on PATH) also falls back to the unknown/False
    defaults without crashing the run."""
    with patch("gbdt.__main__.subprocess.run",
                side_effect=FileNotFoundError("git: command not found")):
        pf = _collect_preflight(tmp_path)
    assert pf["code_commit"] == "unknown"
    assert pf["code_dirty"] is False
    # All six fields still present.
    assert set(pf.keys()) == set(_PREFLIGHT_FIELDS)


def test_B4_preflight_resolves_symlinked_cache_db(tmp_path):
    """B4: when ``data/processed.db`` is a symlink to another path the
    preflight resolves it (records the symlink target, not the link)."""
    # Real DB lives outside the repo root (mimics /tmp/exp_data wipe pattern).
    real_data = tmp_path / "real_cache"
    real_data.mkdir()
    real_db = real_data / "processed.db"
    real_db.write_bytes(b"\xff" * 8192)

    # Repo's data/ is a symlink to the real cache.
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "data").symlink_to(real_data)

    pf = _collect_preflight(repo)
    assert pf["cache_db"] == os.path.realpath(real_db)
    assert pf["cache_db_size"] == 8192
    assert pf["cache_db_mtime"]  # ISO timestamp populated
    # data_root resolves through the directory symlink too.
    assert pf["data_root"] == os.path.realpath(real_data)


def test_B5_preflight_missing_db_returns_empty_string(tmp_path):
    """Edge: no processed.db on disk -> empty path string + 0 size.
    The runner emits the line anyway (fingerprint of an empty-cache run
    is still useful)."""
    pf = _collect_preflight(tmp_path)
    assert pf["cache_db"] == ""
    assert pf["cache_db_size"] == 0
    assert pf["cache_db_mtime"] == ""
    # data_root is always populated (resolved even if the dir doesn't exist).
    assert pf["data_root"]


def test_B6_preflight_with_real_git_repo(tmp_path):
    """When the repo IS a git repo, code_commit is the 40-char SHA and
    code_dirty reflects working-tree state. Uses a real (tiny) git repo
    in tmp_path so we don't depend on the surrounding repo."""
    # Skip if git not on PATH (CI sandbox may lack it).
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True,
                        timeout=5)
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        pytest.skip("git binary unavailable")

    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ,
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True,
                    env=env)
    (repo / "f.txt").write_text("hi\n")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"],
                    cwd=repo, check=True, env=env)

    pf_clean = _collect_preflight(repo)
    assert re.fullmatch(r"[0-9a-f]{40}", pf_clean["code_commit"])
    assert pf_clean["code_dirty"] is False

    # Dirty the working tree.
    (repo / "f.txt").write_text("changed\n")
    pf_dirty = _collect_preflight(repo)
    assert pf_dirty["code_commit"] == pf_clean["code_commit"]
    assert pf_dirty["code_dirty"] is True


def test_B7_sanitize_preflight_for_emission_relativizes_paths(tmp_path):
    """B7: ``_sanitize_preflight_for_emission`` rewrites in-repo paths
    relative to the repo root and collapses external paths to
    ``<external>/<basename>``. Prevents the host's absolute partition path
    from leaking into committed ``metrics.json`` files and run logs while
    leaving the raw preflight dict (consumed by the universe + FS-prefit
    caches) untouched."""
    # In-repo case: data/ symlink to a directory inside tmp_path's repo.
    repo = tmp_path / "repo"
    repo.mkdir()
    data_dir = repo / "data"
    data_dir.mkdir()
    db = data_dir / "processed.db"
    db.write_bytes(b"\x00" * 256)

    pf = _collect_preflight(repo)
    # Raw fields untouched — downstream consumers see realpaths.
    assert pf["cache_db"] == os.path.realpath(db)
    assert pf["data_root"] == os.path.realpath(data_dir)

    san = _sanitize_preflight_for_emission(pf, repo)
    # Emitted view is repo-relative.
    assert san["cache_db"] == "data/processed.db"
    assert san["data_root"] == "data"
    # Non-path fields unchanged.
    assert san["cache_db_size"] == pf["cache_db_size"]
    assert san["cache_db_mtime"] == pf["cache_db_mtime"]
    assert san["code_commit"] == pf["code_commit"]
    assert san["code_dirty"] == pf["code_dirty"]

    # External case: data root realpath outside the repo (mirrors the
    # production worktree-symlink contract: data/ → /mnt/<uuid>/cache_data).
    external = tmp_path / "external_cache"
    external.mkdir()
    ext_db = external / "processed.db"
    ext_db.write_bytes(b"\xff" * 128)
    pf_ext = {
        "cache_db": str(ext_db),
        "cache_db_size": 128,
        "cache_db_mtime": "2026-06-09T00:00:00+00:00",
        "data_root": str(external),
        "code_commit": "deadbeef" * 5,
        "code_dirty": False,
    }
    san_ext = _sanitize_preflight_for_emission(pf_ext, repo)
    assert san_ext["cache_db"] == "<external>/processed.db"
    assert san_ext["data_root"] == "<external>/external_cache"

    # Empty-string case: missing DB. Sanitizer must not crash.
    pf_empty = {**pf_ext, "cache_db": "", "cache_db_size": 0,
                "cache_db_mtime": ""}
    san_empty = _sanitize_preflight_for_emission(pf_empty, repo)
    assert san_empty["cache_db"] == ""


def test_B8_sanitize_path_for_emission_covers_progress_log_sites(tmp_path):
    """B8: ``_sanitize_path_for_emission`` covers the standalone path-emission
    sites — the four ``_milestone(...)`` calls in ``run_experiment`` that
    interpolate paths into ``loop/progress.log`` (a committed artifact).

    Verifies the four representative path shapes the runner emits:
    in-repo ``out_dir``, in-repo ``sweep_csv_path``, external cache root,
    and empty-string graceful handling.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    # `Path` input — the runner usually passes Path objects to _milestone.
    out_dir = repo / "results/gbdt/experiments/test_cell"
    assert _sanitize_path_for_emission(out_dir, repo) == (
        "results/gbdt/experiments/test_cell"
    )

    # `str` input — the sweep CSV path is built as a string.
    sweep_csv = str(repo / "results/gbdt/data/r_precision_at_k.csv")
    assert _sanitize_path_for_emission(sweep_csv, repo) == (
        "results/gbdt/data/r_precision_at_k.csv"
    )

    # External cache path — collapses to <external>/<basename>.
    external = tmp_path / "external_cache" / "gbdt_feature_cache"
    assert _sanitize_path_for_emission(external, repo) == (
        "<external>/gbdt_feature_cache"
    )

    # Empty/None graceful handling — no crash, returns "".
    assert _sanitize_path_for_emission("", repo) == ""
    assert _sanitize_path_for_emission(None, repo) == ""
