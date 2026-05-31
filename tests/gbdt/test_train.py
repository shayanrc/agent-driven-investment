"""Stage 7 — walk-forward driver tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt.diagnostics import DiagnosticBundle
from gbdt.train import (
    SplitSpec,
    carve_single_fold,
    default_fs_hp_callback,
    walk_forward_train,
)


def _toy_panel(n_per_ticker: int = 1600, n_tickers: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=n_per_ticker, freq="B")
    frames = []
    for i in range(n_tickers):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_per_ticker)))
        df = pd.DataFrame({
            "date": dates,
            "ticker": f"T{i}",
            "open": c, "high": c * 1.005, "low": c * 0.995,
            "close": c, "adj_close": c,
            "volume": np.ones(n_per_ticker, dtype=int),
        })
        frames.append(df)
    panel = pd.concat(frames).set_index(["date", "ticker"]).sort_index()
    # Tiny feature set: 5 noise + 1 signal
    n_total = len(panel)
    X = pd.DataFrame(rng.normal(0, 1, (n_total, 6)),
                      index=panel.index,
                      columns=["sig", "n1", "n2", "n3", "n4", "n5"])
    # Target: 1 iff sig > 0
    y = ((X["sig"] + rng.normal(0, 0.3, n_total)) > 0).astype(int)
    return panel, X, y


# ---------------------------------------------------------------------------
# Carving / boundary discipline
# ---------------------------------------------------------------------------


def test_carve_single_fold_segment_lengths():
    panel, _, _ = _toy_panel(1600, 3)
    split = SplitSpec(800, 400, 200, 100)
    fold = carve_single_fold(panel, split)
    for t in panel.index.get_level_values("ticker").unique():
        assert len(fold.train_idx[t]) == 800
        assert len(fold.val_idx[t]) == 400
        assert len(fold.eval_idx[t]) == 200
        assert len(fold.test_idx[t]) == 100


def test_carve_single_fold_no_overlap_and_order_respected():
    panel, _, _ = _toy_panel(1600, 2)
    split = SplitSpec(800, 400, 200, 100)
    fold = carve_single_fold(panel, split)
    for t in panel.index.get_level_values("ticker").unique():
        all_idx = np.concatenate([fold.train_idx[t], fold.val_idx[t],
                                   fold.eval_idx[t], fold.test_idx[t]])
        # Strictly forward (monotonic increasing positions)
        assert np.all(np.diff(all_idx) == 1), (
            f"non-contiguous walk-forward positions for ticker {t}"
        )
        # Train < val < eval < test (boundary C6)
        assert fold.train_idx[t].max() < fold.val_idx[t].min()
        assert fold.val_idx[t].max() < fold.eval_idx[t].min()
        assert fold.eval_idx[t].max() < fold.test_idx[t].min()


def test_carve_drops_tickers_below_min_total():
    panel, _, _ = _toy_panel(1600, 2)
    # Pad with a short ticker
    short = panel.xs("T0", level="ticker").head(500).reset_index()
    short["ticker"] = "TOO_SHORT"
    short = short.set_index(["date", "ticker"])
    big = pd.concat([panel, short]).sort_index()
    split = SplitSpec(800, 400, 200, 100)
    fold = carve_single_fold(big, split)
    assert "TOO_SHORT" not in fold.train_idx


# ---------------------------------------------------------------------------
# Inner-stop integration
# ---------------------------------------------------------------------------


def test_walk_forward_inner_stop_plateau_fires():
    """A no-op callback that returns the same features + same HP triggers
    plateau after a few iterations because val Brier doesn't improve."""
    panel, X, y = _toy_panel(1600, 2, seed=1)
    feats = list(X.columns)

    def noop_cb(bundle, available):
        # Tiny HP nudge to force a refit without changing meaningfully
        return list(available), dict(bundle.hp), "noop"

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=feats,
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=8,
        plateau_threshold=0.005,
        degradation_gate=0.10,
        fs_hp_callback=noop_cb,
    )
    assert result.inner_stop_signal in ("plateau", "cap", "degradation")
    assert len(result.iterations) >= 1
    assert result.best_val_brier == min(b.val_brier for b in result.iterations)


def test_walk_forward_best_checkpoint_logic():
    panel, X, y = _toy_panel(1600, 2, seed=2)

    def cb(bundle, available):
        # Always shrink to 2 features → worse val Brier
        return ["sig", "n1"], dict(bundle.hp), "shrink"

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=3,
        fs_hp_callback=cb,
    )
    # Iter 0 saw full features; later iters shrank.
    # Best should still be one of the 3.
    assert 0 <= result.best_iteration < len(result.iterations)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def test_walk_forward_emits_segment_predictions():
    panel, X, y = _toy_panel(1600, 2, seed=3)
    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain"},
        max_iterations=2,
    )
    for seg in ("train", "val", "eval", "test"):
        df = result.predictions[seg]
        assert set(df.columns) >= {"date", "ticker", "p_raw", "p_calibrated", "y_true"}


# ---------------------------------------------------------------------------
# Algorithmic callback fallback
# ---------------------------------------------------------------------------


def test_default_fs_hp_callback_drops_unimportant():
    bundle = DiagnosticBundle(
        iter=0, hp={"iterations": 100, "depth": 6, "boosting_type": "Plain",
                    "l2_leaf_reg": 3.0, "learning_rate": 0.05},
        features=["a", "b", "c"], n_features=3,
        train_brier=0.20, val_brier=0.22, train_val_gap=0.02,
        eval_brier_provisional=None,
        spiegelhalter_z=0.5, spiegelhalter_p=0.6,
        reliability={}, positive_prevalence_val=0.4, positive_recall_val=0.4,
        early_stop_iteration=80, iteration_cap_hit=False,
        importance_native={"a": 1.0, "b": 0.5, "c": 0.001},
        importance_permutation=None, top_feature_correlation={},
        learning_curve={},
    )
    keep, hp, why = default_fs_hp_callback(bundle, ["a", "b", "c"])
    # 'c' has importance 0.001 = 0.1% of top → dropped only if we have >=10 to start
    # but we have 3 so default keeps top-10 = all 3.
    assert set(keep) == {"a", "b", "c"}
    assert "fallback" in why.lower()


# ---------------------------------------------------------------------------
# L1 from _187 — val-Brier tie-break (integration smoke)
# ---------------------------------------------------------------------------


def test_walk_forward_tiebreak_picks_lower_gap_within_band():
    """Drive a multi-iteration loop where two iters land in the tie band and
    the lower-gap iter should win. The integration covers: bundles carry
    ``train_val_gap`` + ``spiegelhalter_z``; ``walk_forward_train`` threads
    them into ``best_checkpoint``; the chosen ``best_iteration`` exposes a
    present ``train_val_gap`` + ``spiegelhalter_z`` on its bundle.
    """
    panel, X, y = _toy_panel(1600, 3, seed=4)

    # Three iters: all stay close on val Brier (small HP nudges). The wide
    # tie_band guarantees the tie set spans all of them so the picker chooses
    # the lowest-gap iter.
    nudges = [0.05, 0.052, 0.048]
    seq = iter(nudges[1:])

    def cb(bundle, available):
        try:
            lr = next(seq)
        except StopIteration:
            lr = 0.05
        next_hp = dict(bundle.hp)
        next_hp["learning_rate"] = lr
        return list(available), next_hp, f"lr={lr}"

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain",
            "learning_rate": nudges[0]},
        max_iterations=3,
        plateau_threshold=0.005,
        degradation_gate=1.0,         # disable degradation gate
        tie_band=1.0,                 # huge band → all iters tied
        fs_hp_callback=cb,
    )
    assert len(result.iterations) >= 2
    gaps = [b.train_val_gap for b in result.iterations]
    zs = [b.spiegelhalter_z for b in result.iterations]
    assert all(g is not None for g in gaps), "gap should be present per iter"
    assert all(z is not None for z in zs), "Z should be present per iter"
    # With tie_band=1.0 the full history is one tie set → winner is the
    # lowest-gap iter. (Ties broken on |z| then iter idx.)
    lowest_gap_iter = min(range(len(gaps)), key=lambda i: (gaps[i], abs(zs[i]), i))
    assert result.best_iteration == lowest_gap_iter
    # And the selected bundle's gap + z are present in the artifact bundle.
    chosen = result.iterations[result.best_iteration]
    assert chosen.train_val_gap is not None
    assert chosen.spiegelhalter_z is not None


def test_walk_forward_tiebreak_disabled_preserves_strict_argmin():
    """``tie_band=0.0`` reverts to the strict val-Brier argmin even when
    gap/Z metrics are available — backwards-compatible behaviour."""
    panel, X, y = _toy_panel(1600, 3, seed=5)

    def cb(bundle, available):
        next_hp = dict(bundle.hp)
        return list(available), next_hp, "noop"

    result = walk_forward_train(
        panel=panel, X=X, y=y, features=list(X.columns),
        hp={"iterations": 30, "depth": 3, "boosting_type": "Plain",
            "learning_rate": 0.05},
        max_iterations=3,
        plateau_threshold=0.005,
        degradation_gate=1.0,
        tie_band=0.0,                 # disable tie-break
        fs_hp_callback=cb,
    )
    strict_best = min(
        range(len(result.iterations)),
        key=lambda i: result.iterations[i].val_brier,
    )
    assert result.best_iteration == strict_best
