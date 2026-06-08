"""Regression tests for canonical R-Precision@K tie-break + macro form
in the runner's segment_diagnostics path.

Issue #252 (PR #139 review M1): the runner's
``segment_diagnostics.<seg>.top_k_metrics.per_day.p_at_k`` field uses
**micro** aggregation
(``sum_d(positives_in_top_k(d)) / sum_d(min(R(d), k))``) and was the
source of memo-author confusion when its values disagreed with the
canonical CSV ``results/gbdt/data/r_precision_at_k.csv`` (macro,
``(1/Q) · Σ r_q / min(K, R_q)``). Tie-break ordering inside each day
is identical in both paths (stable mergesort by
``(p_calibrated desc, ticker asc)``); the divergence is in the
aggregation, not the tie-break.

Fix (this PR): the runner's segment_diagnostics bundle now also carries
an ``r_precision_at_k`` block — canonical macro form, byte-identical to
the regenerate script. These tests pin that contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbdt import topk_diagnostics


# ---------------------------------------------------------------------------
# Helper — port of the canonical regenerate-script computation
# ---------------------------------------------------------------------------


def _canonical_r_precision_from_regenerate_script(
    df: pd.DataFrame, k_values: tuple[int, ...]
) -> dict[int, float]:
    """Re-implement the regenerate script's per-cell computation.

    Mirrors ``scripts/gbdt/regenerate_r_precision_at_k_csv.py::compute_row``
    body (the part that builds ``R_precision_at_K`` columns). Kept here
    intentionally so a future drift in either path triggers this test.
    """
    by_day = [
        (
            d,
            g.sort_values(
                by=["p_calibrated", "ticker"],
                ascending=[False, True],
                kind="mergesort",
            ),
        )
        for d, g in df.groupby("date")
    ]
    out: dict[int, float] = {}
    for K in k_values:
        ratios = []
        for _d, g in by_day:
            R_q = int(g["y_true"].sum())
            if R_q == 0:
                continue
            r_q = int(g.head(K)["y_true"].sum())
            ratios.append(r_q / min(K, R_q))
        out[K] = float(np.mean(ratios)) if ratios else float("nan")
    return out


# ---------------------------------------------------------------------------
# 1. Tied-prob fixture: same data, both paths produce equal top-K picks
# ---------------------------------------------------------------------------


def _tied_prob_fixture() -> pd.DataFrame:
    """1 day, 5 tickers with several tied ``p_calibrated`` values.

    Tickers (alphabetical order, ascending tie-break for ties):
      - A: p=0.7 (tied with C, E)
      - B: p=0.7 (tied with C, E)  -- wait, vary it
      - C: p=0.7 (tied)
      - D: p=0.3
      - E: p=0.7 (tied)

    Layout that makes the tie-break matter:
      - A: p=0.9, y=1   (unambiguous #1)
      - B: p=0.5, y=0   (tied with D, E at 0.5)
      - C: p=0.5, y=1   (tied)
      - D: p=0.5, y=1   (tied)
      - E: p=0.5, y=0   (tied)

    With canonical tie-break ``(p desc, ticker asc)`` the day's sorted
    order is: A, B, C, D, E. Top-3 picks = {A, B, C}; positives caught
    = 2 (A + C). R_q = 3 (A + C + D). With ``min(3, 3) = 3`` the
    canonical R-Precision@3 = 2/3 ≈ 0.6667.

    Under any other tie-break (e.g. unstable quicksort on p only, which
    could yield A, D, C, B, E), positives caught could be 3 (A + D + C)
    and R-Precision@3 = 3/3 = 1.0 — measurably different.
    """
    rows = [
        {"date": "2024-01-01", "ticker": "A", "p_calibrated": 0.9, "y_true": 1},
        {"date": "2024-01-01", "ticker": "B", "p_calibrated": 0.5, "y_true": 0},
        {"date": "2024-01-01", "ticker": "C", "p_calibrated": 0.5, "y_true": 1},
        {"date": "2024-01-01", "ticker": "D", "p_calibrated": 0.5, "y_true": 1},
        {"date": "2024-01-01", "ticker": "E", "p_calibrated": 0.5, "y_true": 0},
    ]
    df = pd.DataFrame(rows)
    df["p_raw"] = df["p_calibrated"]
    df["sample_weight"] = 1.0
    return df


def test_per_day_p_at_k_matches_canonical_tie_break():
    """Synthetic 1-day, 5-ticker fixture with tied ``p_calibrated`` values.

    Computes K=3 R-Precision via both
    (a) the runner's new ``compute_r_precision_at_k`` path, and
    (b) the inline canonical-regenerate-script reference,
    and asserts byte equality on the K=3 value. The expected value
    derives from the canonical tie-break ``(p desc, ticker asc)`` —
    sorted-day = [A, B, C, D, E], top-3 = {A, B, C}, r_q = 2, R_q = 3,
    min(K, R_q) = 3, ratio = 2/3.
    """
    df = _tied_prob_fixture()
    runner_block = topk_diagnostics.compute_r_precision_at_k(
        df, k_values=(3,)
    )
    canonical = _canonical_r_precision_from_regenerate_script(
        df, k_values=(3,)
    )

    expected = 2.0 / 3.0
    assert canonical[3] == pytest.approx(expected)
    assert (
        runner_block["by_k"]["3"]["r_precision_at_k"] == pytest.approx(expected)
    )
    # Byte-equality of runner vs canonical reference.
    assert (
        runner_block["by_k"]["3"]["r_precision_at_k"]
        == pytest.approx(canonical[3])
    )


def test_micro_vs_macro_divergence_known_case():
    """Micro vs macro produce measurably different values on staggered
    panels. This is the original source of memo-author confusion
    (PR #139 review M1).

    Construction:
      Day 1: 3 tickers, R=1, model catches it in top-3. r/min(K,R) = 1/1 = 1.0.
      Day 2: 6 tickers, R=3, model catches 0 in top-3. r/min(K,R) = 0/3 = 0.0.

      Macro: (1 + 0) / 2 = 0.5
      Micro: sum_r=1, sum_denom=1+3=4 → 0.25
    """
    rows = [
        # Day 1 — top-3 picks (sorted by p desc): A, B, C; A is positive.
        {"date": "2024-01-01", "ticker": "A", "p_calibrated": 0.9, "y_true": 1},
        {"date": "2024-01-01", "ticker": "B", "p_calibrated": 0.5, "y_true": 0},
        {"date": "2024-01-01", "ticker": "C", "p_calibrated": 0.3, "y_true": 0},
        # Day 2 — top-3 picks (sorted by p desc): A, B, C; none positive.
        {"date": "2024-01-02", "ticker": "A", "p_calibrated": 0.9, "y_true": 0},
        {"date": "2024-01-02", "ticker": "B", "p_calibrated": 0.8, "y_true": 0},
        {"date": "2024-01-02", "ticker": "C", "p_calibrated": 0.7, "y_true": 0},
        {"date": "2024-01-02", "ticker": "D", "p_calibrated": 0.1, "y_true": 1},
        {"date": "2024-01-02", "ticker": "E", "p_calibrated": 0.05, "y_true": 1},
        {"date": "2024-01-02", "ticker": "F", "p_calibrated": 0.01, "y_true": 1},
    ]
    df = pd.DataFrame(rows)
    df["p_raw"] = df["p_calibrated"]
    df["sample_weight"] = 1.0

    # Micro (runner legacy)
    micro = topk_diagnostics.compute_top_k_metrics(df, k_values=(3,))
    assert micro["per_day"]["3"]["p_at_k"] == pytest.approx(0.25)

    # Macro (canonical)
    macro = topk_diagnostics.compute_r_precision_at_k(df, k_values=(3,))
    assert macro["by_k"]["3"]["r_precision_at_k"] == pytest.approx(0.5)

    # And the canonical-script reference matches macro.
    canonical = _canonical_r_precision_from_regenerate_script(
        df, k_values=(3,)
    )
    assert canonical[3] == pytest.approx(0.5)


def test_canonical_macro_matches_regenerate_script_on_full_K_set():
    """On a non-trivial multi-day panel, the runner's
    ``compute_r_precision_at_k`` matches the inline regenerate-script
    reference at every K in the standard {1, 3, 5, 10, 20} set.
    """
    # Build a 5-day, 8-ticker panel with mixed prevalence + ties.
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
    rng = np.random.default_rng(0)
    rows = []
    for day_i in range(5):
        date = f"2024-01-{day_i + 1:02d}"
        # Probabilities: give 2 ties per day to exercise the tie-break.
        probs = rng.uniform(0.1, 0.9, size=len(tickers))
        # Force a tie between tickers C and F.
        probs[2] = 0.5
        probs[5] = 0.5
        # Force a 3-way tie between B, D, G.
        probs[1] = 0.7
        probs[3] = 0.7
        probs[6] = 0.7
        ys = rng.integers(0, 2, size=len(tickers))
        for t, p, y in zip(tickers, probs, ys):
            rows.append({
                "date": date, "ticker": t,
                "p_calibrated": float(p), "p_raw": float(p),
                "y_true": int(y), "sample_weight": 1.0,
            })
    df = pd.DataFrame(rows)

    K_set = (1, 3, 5, 10, 20)
    runner = topk_diagnostics.compute_r_precision_at_k(df, k_values=K_set)
    canonical = _canonical_r_precision_from_regenerate_script(df, k_values=K_set)

    for k in K_set:
        runner_val = runner["by_k"][str(k)]["r_precision_at_k"]
        canon_val = canonical[k]
        if np.isnan(canon_val):
            assert runner_val is None or np.isnan(runner_val)
        else:
            assert runner_val == pytest.approx(canon_val), (
                f"K={k}: runner={runner_val} canonical={canon_val}"
            )


def test_canonical_block_shape_and_provenance_fields():
    """The runner's ``r_precision_at_k`` block carries the provenance
    fields memo-authors need to identify which formula it is."""
    df = _tied_prob_fixture()
    block = topk_diagnostics.compute_r_precision_at_k(df, k_values=(1, 3, 5))
    assert block["formula_version"] == "macro_per_day_fixed_k"
    assert (
        block["tie_break"]
        == "(p_calibrated desc, ticker asc) mergesort"
    )
    assert block["n_rows"] == 5
    assert block["Q_days"] == 1
    assert block["base_rate"] == pytest.approx(3 / 5)
    assert set(block["by_k"].keys()) == {"1", "3", "5"}
    for k in ("1", "3", "5"):
        assert "r_precision_at_k" in block["by_k"][k]
        assert "n_qualifying_days" in block["by_k"][k]


def test_canonical_block_handles_empty_segment():
    """Empty df → all r_precision_at_k values None, Q_days=0."""
    empty = pd.DataFrame(
        columns=["date", "ticker", "p_calibrated", "p_raw", "y_true",
                 "sample_weight"]
    )
    block = topk_diagnostics.compute_r_precision_at_k(empty)
    assert block["n_rows"] == 0
    assert block["Q_days"] == 0
    assert block["base_rate"] is None
    for k in ("1", "3", "5", "10", "20"):
        assert block["by_k"][k]["r_precision_at_k"] is None
        assert block["by_k"][k]["n_qualifying_days"] == 0


def test_canonical_block_handles_no_positive_days():
    """All-negative segment → Q_days=0, every K → None."""
    rows = [
        {"date": "2024-01-01", "ticker": "A", "p_calibrated": 0.5, "y_true": 0},
        {"date": "2024-01-01", "ticker": "B", "p_calibrated": 0.3, "y_true": 0},
        {"date": "2024-01-02", "ticker": "A", "p_calibrated": 0.5, "y_true": 0},
        {"date": "2024-01-02", "ticker": "B", "p_calibrated": 0.3, "y_true": 0},
    ]
    df = pd.DataFrame(rows)
    df["p_raw"] = df["p_calibrated"]
    df["sample_weight"] = 1.0
    block = topk_diagnostics.compute_r_precision_at_k(df, k_values=(1, 3))
    assert block["n_rows"] == 4
    assert block["Q_days"] == 0
    assert block["by_k"]["1"]["r_precision_at_k"] is None
    assert block["by_k"]["3"]["r_precision_at_k"] is None


def test_canonical_block_determinism():
    """Row-order permutation of the input must not change output."""
    df = _tied_prob_fixture()
    # Build a multi-day richer fixture so determinism actually exercises
    # the cross-day grouping path.
    rows = []
    for d in ("2024-01-01", "2024-01-02", "2024-01-03"):
        for t, p, y in [
            ("A", 0.9, 1), ("B", 0.5, 0), ("C", 0.5, 1),
            ("D", 0.5, 1), ("E", 0.3, 0),
        ]:
            rows.append({
                "date": d, "ticker": t, "p_calibrated": p,
                "p_raw": p, "y_true": y, "sample_weight": 1.0,
            })
    df = pd.DataFrame(rows)
    a = topk_diagnostics.compute_r_precision_at_k(df, k_values=(1, 3, 5))
    shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
    b = topk_diagnostics.compute_r_precision_at_k(shuffled, k_values=(1, 3, 5))
    assert a == b


def test_compute_all_includes_r_precision_block():
    """``compute_all`` wires the canonical block into the bundle the
    runner persists into ``metrics.json::segment_diagnostics``."""
    df = _tied_prob_fixture()
    bundle = topk_diagnostics.compute_all(df)
    assert "r_precision_at_k" in bundle
    assert bundle["r_precision_at_k"]["formula_version"] == "macro_per_day_fixed_k"
    # Default K set per ``project-r-precision-methodology.md``.
    assert set(bundle["r_precision_at_k"]["by_k"].keys()) == {
        "1", "3", "5", "10", "20"
    }
