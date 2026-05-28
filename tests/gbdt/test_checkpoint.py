"""V1.1 Phase 1 — checkpoint/resume scaffolding round-trip.

Covers (plan § 11 Phase 1 + § 12 R6):
  - ``write_checkpoint`` then ``read_checkpoint`` returns the same dict, with a
    ``schema_version`` field stamped on.
  - ``read_checkpoint`` on a missing path returns ``None``.
  - The checkpoint path co-locates under ``<artifact_dir>/loop/checkpoint.json``
    (plan § 0.6).

Pure unit tests on the JSON round-trip; no runner / training involved.
"""

from __future__ import annotations

from gbdt.checkpoint import (
    SCHEMA_VERSION,
    checkpoint_path,
    read_checkpoint,
    write_checkpoint,
)


def test_checkpoint_path_co_located_under_loop_subdir(tmp_path):
    out_dir = tmp_path / "results" / "gbdt" / "experiments" / "cell_x"
    p = checkpoint_path(out_dir)
    assert p == out_dir / "loop" / "checkpoint.json"


def test_read_missing_returns_none(tmp_path):
    assert read_checkpoint(tmp_path / "no_such_run") is None


def test_write_then_read_round_trip(tmp_path):
    out_dir = tmp_path / "cell_x"
    state = {
        "iter": 3,
        "max_iterations": 8,
        "current_features": ["index_return_5", "realized_vol_200"],
        "current_hp": {"depth": 6, "learning_rate": 0.05},
        "val_briers": [0.1685, 0.1660, 0.1642],
    }
    path = write_checkpoint(out_dir, state)
    assert path.exists()
    assert path == checkpoint_path(out_dir)

    loaded = read_checkpoint(out_dir)
    assert loaded is not None
    # Schema version is stamped on the persisted dict (plan § 12 R6).
    assert loaded["schema_version"] == SCHEMA_VERSION
    # All original keys survive the round-trip unchanged.
    for k, v in state.items():
        assert loaded[k] == v


def test_write_does_not_mutate_input_state(tmp_path):
    state = {"iter": 0}
    write_checkpoint(tmp_path / "cell", state)
    assert "schema_version" not in state  # input untouched


def test_write_creates_parent_dirs(tmp_path):
    # The loop/ subdir does not exist yet; write_checkpoint must create it.
    out_dir = tmp_path / "deeply" / "nested" / "cell"
    assert not (out_dir / "loop").exists()
    write_checkpoint(out_dir, {"iter": 1})
    assert (out_dir / "loop" / "checkpoint.json").exists()


def test_round_trip_via_string_path(tmp_path):
    # Accepts a path-like string as well as a Path.
    out_dir = str(tmp_path / "cell_str")
    write_checkpoint(out_dir, {"iter": 2})
    loaded = read_checkpoint(out_dir)
    assert loaded is not None and loaded["iter"] == 2
