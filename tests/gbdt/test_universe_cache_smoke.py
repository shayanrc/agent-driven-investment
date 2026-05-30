"""task #183 — runner-integration smoke for the shared universe-level cache.

The unit tests in ``test_universe_feature_cache.py`` cover the cache module in
isolation (key composition, round-trip, miss paths). This smoke wires it
through the *real* ``run_experiment(...)`` runner — proving that the two-level
flow in ``__main__.py`` actually works end-to-end:

  1. **Cell A** (target tuple #1) runs cold → builds → writes the universe
     cache (under our tmp ``data_root``).
  2. **Cell B** (target tuple #2 — same universe, DIFFERENT target) runs and
     the universe cache HITS — no rebuild. The matrix Cell B uses is
     bit-identical to what it would have built standalone.

That last assertion is the load-bearing one: it certifies that sharing the
matrix across sibling cells produces the EXACT SAME ``X`` the runner would
have produced on a cold build, so results / determinism / the finalization
contract are unchanged.

Harness mirrors ``test_phase4_smoke.py``: a tiny synthetic panel injected at
the ``gbdt.data.load_panel`` seam, with the rest of the runner running for
real. ``_collect_preflight`` is monkeypatched to point ``data_root`` at
``tmp_path`` (so the universe cache lands somewhere we control and the test
is hermetic — no writes to the real shared cache).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

import gbdt.data as gbdt_data
import gbdt.__main__ as gbdt_main
from gbdt import feature_cache as per_cell_cache
from gbdt import features as gbdt_features
from gbdt import universe_feature_cache as ufc
from gbdt.__main__ import run_experiment
from gbdt.data import TickerStatus, UniversePanel


# ---------------------------------------------------------------------------
# Synthetic universe panel (same shape as test_phase4_smoke)
# ---------------------------------------------------------------------------


def _synthetic_panel(n_per_ticker: int = 360, n_tickers: int = 4,
                     seed: int = 3) -> UniversePanel:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_per_ticker, freq="B")
    frames, statuses = [], []
    for i in range(n_tickers):
        rets = rng.normal(0.0003, 0.012, n_per_ticker)
        c = 100.0 * np.exp(np.cumsum(rets))
        ticker = f"NSE:T{i}"
        frames.append(pd.DataFrame({
            "date": dates, "ticker": ticker,
            "open": c,
            "high": c * (1 + np.abs(rng.normal(0, 0.004, n_per_ticker))),
            "low": c * (1 - np.abs(rng.normal(0, 0.004, n_per_ticker))),
            "close": c, "adj_close": c,
            "volume": rng.integers(100_000, 500_000, n_per_ticker),
        }))
        statuses.append(TickerStatus(
            ticker=ticker, rows=n_per_ticker, kept=True, reason="",
            cache_last_date="2016-05-01", cache_age_days=1, is_stale=False,
        ))
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()

    ic = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, n_per_ticker)))
    index_series = pd.DataFrame({
        "date": dates, "open": ic, "high": ic * 1.003, "low": ic * 0.997,
        "close": ic, "adj_close": ic,
        "volume": rng.integers(1_000_000, 5_000_000, n_per_ticker),
    }).set_index("date")

    return UniversePanel(
        universe="smoke_share",
        panel=panel,
        index_series=index_series,
        annualization_factor=250,
        statuses=statuses,
        stale_tickers=[],
        staleness_days_threshold=gbdt_data.DEFAULT_STALENESS_DAYS,
    )


def _write_sibling_spec(
    out_dir, artifact_dir, *,
    stem: str, threshold_pct: int, horizon_days: int, max_drawdown,
) -> object:
    """Write a runner spec that only differs in the target tuple — sibling cells
    for the universe-share scenario."""
    spec = {
        "target": {
            "universe": "smoke_share",
            "direction": "up",
            "threshold_pct": threshold_pct,
            "horizon_days": horizon_days,
            "max_drawdown": max_drawdown,
        },
        "split": {
            "train_rows": 180, "val_rows": 90, "eval_rows": 50, "test_rows": 30,
            "min_rows_per_ticker": 350,
        },
        "features": {
            "candidates": ["F2", "F4"],          # 6 cols total — fast build
            "lookback_windows": [5, 10, 20],
            "exclude": [],
        },
        "backend": {
            "library": "catboost",
            "calibration_method": "conditional_isotonic",
            # NOT agent_file_protocol — let the default algorithmic loop run
            # to completion so the smoke exercises a real end-to-end run, not
            # a paused one. With max_iterations=1 the loop finishes in seconds.
            "fs_hp_loop": {"max_iterations": 1, "callback_mode": "default"},
            "hp_starting": {
                "iterations": 15, "depth": 2, "learning_rate": 0.1,
                "l2_leaf_reg": 3.0, "boosting_type": "Plain",
                "early_stopping_rounds": 10,
            },
        },
        "artifacts": {"experiment_dir": str(artifact_dir)},
        "random_seed": 42,
    }
    spec_path = out_dir / f"{stem}.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return spec_path


@pytest.fixture()
def share_env(tmp_path, monkeypatch):
    """Patch load_panel + redirect preflight.data_root to tmp_path.

    Keeping the universe cache writes under ``tmp_path/cache_data`` keeps the
    test hermetic — nothing lands in the real shared cache at
    ``/mnt/.../cache_data/gbdt_feature_cache/``.
    """
    monkeypatch.setenv("GBDT_HEARTBEAT_INTERVAL", "0")
    monkeypatch.setattr(
        gbdt_data, "load_panel",
        lambda universe, *a, **kw: _synthetic_panel(),
    )

    fake_data_root = tmp_path / "cache_data"
    fake_data_root.mkdir()
    real_preflight = gbdt_main._collect_preflight

    def _patched_preflight(repo_root):
        pf = real_preflight(repo_root)
        pf["data_root"] = str(fake_data_root)
        return pf

    monkeypatch.setattr(gbdt_main, "_collect_preflight", _patched_preflight)

    art_dir = tmp_path / "artifacts"
    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    return spec_dir, art_dir, fake_data_root


# ---------------------------------------------------------------------------
# Smoke: cell A populates the universe cache, cell B hits it.
# ---------------------------------------------------------------------------


def test_sibling_cells_share_universe_feature_cache(share_env, capsys):
    spec_dir, art_dir, fake_data_root = share_env

    # Cell A — target (up, +5%, 10d, no drawdown). Cold: builds + writes
    # both caches.
    spec_a = _write_sibling_spec(
        spec_dir, art_dir, stem="cellA",
        threshold_pct=5, horizon_days=10, max_drawdown=None,
    )
    run_experiment(spec_a, resume=None)

    # The universe cache directory now contains exactly one parquet + sidecar.
    cache_dir = fake_data_root / ufc.DEFAULT_CACHE_SUBDIR
    assert cache_dir.is_dir(), "universe cache dir must be created on cold build"
    parquets = sorted(cache_dir.glob("*.parquet"))
    sidecars = sorted(cache_dir.glob("*.key.json"))
    assert len(parquets) == 1, f"expected one cached matrix, got {len(parquets)}"
    assert len(sidecars) == 1, f"expected one sidecar, got {len(sidecars)}"

    # Per-cell cache for cell A landed too (artifact dir local).
    cellA_dir = art_dir / "cellA"
    assert (cellA_dir / "_feature_matrix_cache.parquet").exists()

    # Snapshot Cell A's built matrix for the golden compare in step 3.
    X_a = pd.read_parquet(cellA_dir / "_feature_matrix_cache.parquet")

    capsys.readouterr()  # discard cell A's log so we can pin cell B's logs

    # Cell B — DIFFERENT target tuple (up, +10%, 20d, dd 5%), same universe +
    # split + features + seed ⇒ same universe key, different per-cell key.
    spec_b = _write_sibling_spec(
        spec_dir, art_dir, stem="cellB",
        threshold_pct=10, horizon_days=20, max_drawdown=0.05,
    )
    run_experiment(spec_b, resume=None)

    out = capsys.readouterr().out
    # The runner log must mark the universe-cache HIT for cell B (NOT a build,
    # NOT a per-cell hit — cell B is a fresh cell with no prior per-cell cache).
    assert "loaded from universe cache" in out, (
        f"expected cell B to hit the universe cache; runner log was:\n{out}"
    )
    # And the runner MUST NOT have rebuilt (i.e. the "[features] start (no cache
    # hit — building)" line is absent).
    assert "no cache hit — building" not in out, (
        f"cell B should not have rebuilt; runner log was:\n{out}"
    )

    # Still exactly ONE artifact in the shared cache — sibling cells share it.
    parquets_after = sorted(cache_dir.glob("*.parquet"))
    assert len(parquets_after) == 1, (
        f"sibling cells must share the same universe-cache entry; got "
        f"{len(parquets_after)} after cell B"
    )

    # Cell B's per-cell cache landed too (mirrored from the universe layer).
    cellB_dir = art_dir / "cellB"
    cellB_cell_cache = cellB_dir / "_feature_matrix_cache.parquet"
    assert cellB_cell_cache.exists(), (
        "per-cell cache must be mirrored from the universe layer so a future "
        "--resume of cell B hits the cheaper local layer"
    )

    # Golden snapshot: cell B's actual feature matrix == cell A's feature
    # matrix == what build_feature_matrix returns directly on the same panel.
    X_b = pd.read_parquet(cellB_cell_cache)
    pd.testing.assert_frame_equal(X_b, X_a, check_exact=True)

    # And both equal the standalone direct-build (the load-bearing invariant
    # that proves the cache layer doesn't pollute results).
    panel_obj = _synthetic_panel()
    X_direct = gbdt_features.build_feature_matrix(
        panel_obj.panel, panel_obj.index_series,
        lookbacks=(5, 10, 20),
        annualization=250,
        families=["F2", "F4"],
        exclude=[],
    ).dropna(axis=1, how="all")
    pd.testing.assert_frame_equal(X_b, X_direct, check_exact=True)


# ---------------------------------------------------------------------------
# Smoke: a fresh cell with NO prior cache produces a matrix identical to one
# built without the cache layer at all — guards against any silent side-effect
# of writing the universe cache during build.
# ---------------------------------------------------------------------------


def test_cold_build_through_runner_matches_direct_build(share_env):
    spec_dir, art_dir, fake_data_root = share_env
    spec = _write_sibling_spec(
        spec_dir, art_dir, stem="cold",
        threshold_pct=5, horizon_days=10, max_drawdown=None,
    )
    run_experiment(spec, resume=None)

    cold_dir = art_dir / "cold"
    X_runner = pd.read_parquet(cold_dir / "_feature_matrix_cache.parquet")

    # Direct build on the same panel — no cache layer involved.
    panel_obj = _synthetic_panel()
    X_direct = gbdt_features.build_feature_matrix(
        panel_obj.panel, panel_obj.index_series,
        lookbacks=(5, 10, 20),
        annualization=250,
        families=["F2", "F4"],
        exclude=[],
    ).dropna(axis=1, how="all")
    pd.testing.assert_frame_equal(X_runner, X_direct, check_exact=True)


# ---------------------------------------------------------------------------
# Smoke: cell B with a DIFFERENT split MUST miss the universe cache —
# split is part of the cache key (preserves the C6 walk-forward boundary
# discipline).
# ---------------------------------------------------------------------------


def test_different_split_misses_universe_cache(share_env, capsys):
    spec_dir, art_dir, fake_data_root = share_env

    spec_a = _write_sibling_spec(
        spec_dir, art_dir, stem="cellA",
        threshold_pct=5, horizon_days=10, max_drawdown=None,
    )
    run_experiment(spec_a, resume=None)
    capsys.readouterr()  # discard cell A's output

    # Cell C — same universe + features, DIFFERENT split. Must rebuild.
    spec_c = {
        "target": {
            "universe": "smoke_share",
            "direction": "up", "threshold_pct": 5, "horizon_days": 10,
            "max_drawdown": None,
        },
        "split": {
            "train_rows": 200, "val_rows": 70, "eval_rows": 50, "test_rows": 30,
            "min_rows_per_ticker": 350,
        },
        "features": {
            "candidates": ["F2", "F4"],
            "lookback_windows": [5, 10, 20],
            "exclude": [],
        },
        "backend": {
            "library": "catboost",
            "calibration_method": "conditional_isotonic",
            "fs_hp_loop": {"max_iterations": 1, "callback_mode": "default"},
            "hp_starting": {
                "iterations": 15, "depth": 2, "learning_rate": 0.1,
                "l2_leaf_reg": 3.0, "boosting_type": "Plain",
                "early_stopping_rounds": 10,
            },
        },
        "artifacts": {"experiment_dir": str(art_dir)},
        "random_seed": 42,
    }
    spec_c_path = spec_dir / "cellC.yaml"
    spec_c_path.write_text(yaml.safe_dump(spec_c, sort_keys=False))
    run_experiment(spec_c_path, resume=None)

    out = capsys.readouterr().out
    assert "no cache hit — building" in out, (
        "cell with a different split MUST miss the universe cache (C6 "
        "walk-forward boundary discipline)"
    )
    # Two distinct universe-cache entries now exist (one per split).
    cache_dir = fake_data_root / ufc.DEFAULT_CACHE_SUBDIR
    parquets = sorted(cache_dir.glob("*.parquet"))
    assert len(parquets) == 2, (
        f"a different split must produce a different universe key; expected 2 "
        f"cache entries, got {len(parquets)}"
    )
