"""Tests for the four post-prediction segment diagnostics
(``gbdt.topk_diagnostics`` + the report-layer renderer).

Hand-checkable fixtures only — every expected value is computed by hand
in the docstring of the test, not pulled from a reference run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gbdt import topk_diagnostics
from gbdt.report import _render_segment_diagnostics, compute_segment_diagnostics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tiny_df():
    """10 days × 5 tickers = 50 rows, hand-checkable.

    Construction:
      - Tickers A, B, C, D, E. Dates D0..D9 (business days).
      - p_calibrated = 0.9 for ticker A, 0.7 for B, 0.5 for C, 0.3 for D,
        0.1 for E — constant across days. This makes the per-day top-1
        always {A}, top-5 always {A,B,C,D,E}.
      - y_true: ticker A is positive on every day (so per-day P@1 = 1.0).
        Tickers B/C/D/E alternate: B positive on D0,D2,..., C positive on
        D1,D3,..., D never positive, E never positive. Pattern below:
        y[(date_idx, ticker)] in {0,1}.
    """
    tickers = ["A", "B", "C", "D", "E"]
    p_map = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    rows = []
    for di, dt in enumerate(dates):
        for tk in tickers:
            if tk == "A":
                y = 1
            elif tk == "B":
                y = 1 if di % 2 == 0 else 0
            elif tk == "C":
                y = 1 if di % 2 == 1 else 0
            else:
                y = 0
            rows.append({
                "date": dt, "ticker": tk,
                "p_raw": p_map[tk], "p_calibrated": p_map[tk],
                "y_true": y, "sample_weight": 1.0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# compute_top_k_metrics
# ---------------------------------------------------------------------------


def test_top_k_per_day_hand_checkable():
    """P@1 = 1.0 (A always picked, always positive).
    P@5 = positives / 50. Positives: A=10, B=5, C=5 → 20/50 = 0.4.
    base_rate = 20/50 = 0.4. So lift @5 = 1.0.
    Lift @1 = 1.0/0.4 = 2.5.
    """
    df = _tiny_df()
    out = topk_diagnostics.compute_top_k_metrics(df, k_values=(1, 5))
    assert out["n_rows"] == 50
    assert out["base_rate"] == pytest.approx(0.4)
    p1 = out["per_day"]["1"]
    assert p1["p_at_k"] == pytest.approx(1.0)
    assert p1["n_picks_total"] == 10
    assert p1["n_positives_in_picks"] == 10
    assert p1["n_days_full_k"] == 10
    assert p1["n_days_total"] == 10
    assert p1["lift"] == pytest.approx(2.5)
    p5 = out["per_day"]["5"]
    assert p5["p_at_k"] == pytest.approx(0.4)
    assert p5["n_picks_total"] == 50
    assert p5["n_positives_in_picks"] == 20
    assert p5["lift"] == pytest.approx(1.0)


def test_top_k_global_hand_checkable():
    """Global top-5 by score across the whole panel — ties on (A, day0)
    through (A, day9) get broken by ticker asc and the score is 0.9 for
    all of them. So the top-5 by score is {(A, D0)..(A, D4)} — all
    positives. P@5_global = 1.0, lift = 2.5.
    """
    df = _tiny_df()
    out = topk_diagnostics.compute_top_k_metrics(df, k_values=(1, 5))
    g5 = out["global"]["5"]
    assert g5["n_picks"] == 5
    assert g5["n_positives_in_picks"] == 5
    assert g5["p_at_k"] == pytest.approx(1.0)
    assert g5["lift"] == pytest.approx(2.5)


def test_top_k_empty_segment():
    df = pd.DataFrame(columns=["date", "ticker", "p_raw", "p_calibrated",
                                 "y_true", "sample_weight"])
    out = topk_diagnostics.compute_top_k_metrics(df)
    assert out["n_rows"] == 0
    assert out["base_rate"] is None
    for k in ("1", "5", "10"):
        assert out["per_day"][k]["p_at_k"] is None
        assert out["per_day"][k]["n_picks_total"] == 0
        assert out["global"][k]["p_at_k"] is None


def test_top_k_days_with_fewer_than_k_tickers():
    """First day has 2 tickers, second day has 5. k=5: day1 contributes 2
    picks, day2 contributes 5; n_picks_total = 7, n_days_full_k = 1."""
    rows = []
    # Day 0: 2 tickers
    for tk, p, y in [("A", 0.9, 1), ("B", 0.5, 0)]:
        rows.append({"date": "2024-01-01", "ticker": tk,
                      "p_raw": p, "p_calibrated": p, "y_true": y,
                      "sample_weight": 1.0})
    # Day 1: 5 tickers
    for tk, p, y in [("A", 0.9, 1), ("B", 0.7, 1), ("C", 0.5, 0),
                      ("D", 0.3, 0), ("E", 0.1, 0)]:
        rows.append({"date": "2024-01-02", "ticker": tk,
                      "p_raw": p, "p_calibrated": p, "y_true": y,
                      "sample_weight": 1.0})
    df = pd.DataFrame(rows)
    out = topk_diagnostics.compute_top_k_metrics(df, k_values=(5,))
    pd5 = out["per_day"]["5"]
    assert pd5["n_picks_total"] == 7
    assert pd5["n_days_full_k"] == 1
    assert pd5["n_days_total"] == 2
    assert pd5["n_positives_in_picks"] == 3


# ---------------------------------------------------------------------------
# compute_per_ticker_hit_rate
# ---------------------------------------------------------------------------


def test_per_ticker_hit_rate_a_always_picked():
    """A is picked every day (top-5 includes everyone). A's hit_rate =
    10/10 = 1.0. B's hit_rate = 5/10 = 0.5. D's = 0/10 = 0.
    Sorted by n_picks desc then ticker asc: all 5 tickers tied at 10
    picks → alpha-sorted A,B,C,D,E.
    """
    df = _tiny_df()
    out = topk_diagnostics.compute_per_ticker_hit_rate(df, k=5)
    assert out["k"] == 5
    rows = out["rows"]
    assert len(rows) == 5
    assert [r["ticker"] for r in rows] == ["A", "B", "C", "D", "E"]
    assert all(r["n_picks"] == 10 for r in rows)
    by_t = {r["ticker"]: r for r in rows}
    assert by_t["A"]["hit_rate"] == pytest.approx(1.0)
    assert by_t["B"]["hit_rate"] == pytest.approx(0.5)
    assert by_t["C"]["hit_rate"] == pytest.approx(0.5)
    assert by_t["D"]["hit_rate"] == pytest.approx(0.0)
    assert by_t["E"]["hit_rate"] == pytest.approx(0.0)


def test_per_ticker_hit_rate_k1_isolates_a():
    """k=1 picks only A every day → A:10, others:0 picks (and absent
    from rows since they had 0 picks)."""
    df = _tiny_df()
    out = topk_diagnostics.compute_per_ticker_hit_rate(df, k=1)
    assert len(out["rows"]) == 1
    assert out["rows"][0]["ticker"] == "A"
    assert out["rows"][0]["n_picks"] == 10
    assert out["rows"][0]["hit_rate"] == pytest.approx(1.0)


def test_per_ticker_hit_rate_empty():
    df = pd.DataFrame(columns=["date", "ticker", "p_calibrated", "y_true"])
    out = topk_diagnostics.compute_per_ticker_hit_rate(df)
    assert out == {"k": 5, "rows": []}


# ---------------------------------------------------------------------------
# compute_per_quarter_p_k
# ---------------------------------------------------------------------------


def test_per_quarter_p_k_two_quarters():
    """10 days all in 2024Q1 (Jan), so just one quarter. Build a longer
    fixture that spans Q1 + Q2."""
    rows = []
    # 5 days in Q1 (Jan), A always picked + positive.
    # 5 days in Q2 (Apr), A picked but never positive (regime collapse).
    q1_dates = pd.date_range("2024-01-01", periods=5, freq="B")
    q2_dates = pd.date_range("2024-04-01", periods=5, freq="B")
    for dt in q1_dates:
        rows.append({"date": dt, "ticker": "A", "p_calibrated": 0.9,
                      "p_raw": 0.9, "y_true": 1, "sample_weight": 1.0})
        rows.append({"date": dt, "ticker": "B", "p_calibrated": 0.5,
                      "p_raw": 0.5, "y_true": 0, "sample_weight": 1.0})
    for dt in q2_dates:
        rows.append({"date": dt, "ticker": "A", "p_calibrated": 0.9,
                      "p_raw": 0.9, "y_true": 0, "sample_weight": 1.0})
        rows.append({"date": dt, "ticker": "B", "p_calibrated": 0.5,
                      "p_raw": 0.5, "y_true": 0, "sample_weight": 1.0})
    df = pd.DataFrame(rows)
    out = topk_diagnostics.compute_per_quarter_p_k(df, k=1)
    # k=1: Q1 picks A 5x, all positive → P@1=1.0. Q2 picks A 5x, all
    # negative → P@1=0.0.
    assert len(out["rows"]) == 2
    by_q = {r["quarter"]: r for r in out["rows"]}
    assert "2024Q1" in by_q
    assert "2024Q2" in by_q
    assert by_q["2024Q1"]["p_at_k"] == pytest.approx(1.0)
    assert by_q["2024Q2"]["p_at_k"] == pytest.approx(0.0)
    # base_rate = 5/20 = 0.25.
    assert by_q["2024Q1"]["base_rate"] == pytest.approx(0.25)
    assert by_q["2024Q1"]["lift"] == pytest.approx(4.0)
    assert by_q["2024Q2"]["lift"] == pytest.approx(0.0)


def test_per_quarter_p_k_empty():
    df = pd.DataFrame(columns=["date", "ticker", "p_calibrated", "y_true"])
    out = topk_diagnostics.compute_per_quarter_p_k(df)
    assert out == {"k": 5, "rows": []}


# ---------------------------------------------------------------------------
# compute_prediction_range
# ---------------------------------------------------------------------------


def test_prediction_range_flag_fires_when_low():
    """Tight predictions clustered around 0.5 → std small → flag True."""
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 10, "ticker": list("ABCDEFGHIJ"),
        "p_raw": [0.5] * 10,
        "p_calibrated": [0.48, 0.49, 0.50, 0.51, 0.52, 0.48, 0.49, 0.50, 0.51, 0.52],
        "y_true": [0] * 10, "sample_weight": [1.0] * 10,
    })
    out = topk_diagnostics.compute_prediction_range(df)
    assert out["flag_low_separation"] is True
    assert out["std"] < 0.05


def test_prediction_range_flag_not_fires_when_high():
    """Predictions spread from 0.1 to 0.9 → std large → flag False."""
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 5, "ticker": list("ABCDE"),
        "p_raw": [0.1, 0.3, 0.5, 0.7, 0.9],
        "p_calibrated": [0.1, 0.3, 0.5, 0.7, 0.9],
        "y_true": [0, 0, 1, 1, 1], "sample_weight": [1.0] * 5,
    })
    out = topk_diagnostics.compute_prediction_range(df)
    assert out["flag_low_separation"] is False
    assert out["std"] >= 0.05
    assert out["min"] == pytest.approx(0.1)
    assert out["max"] == pytest.approx(0.9)


def test_prediction_range_custom_threshold():
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 5, "ticker": list("ABCDE"),
        "p_raw": [0.4, 0.45, 0.5, 0.55, 0.6],
        "p_calibrated": [0.4, 0.45, 0.5, 0.55, 0.6],
        "y_true": [0, 0, 0, 1, 1], "sample_weight": [1.0] * 5,
    })
    # std ≈ 0.07; flag False at default 0.05 but True at custom 0.10.
    out = topk_diagnostics.compute_prediction_range(df, low_separation_threshold=0.10)
    assert out["flag_low_separation"] is True


def test_prediction_range_empty():
    df = pd.DataFrame(columns=["date", "ticker", "p_calibrated", "y_true"])
    out = topk_diagnostics.compute_prediction_range(df)
    assert out["flag_low_separation"] is False
    assert out["n_rows"] == 0


# ---------------------------------------------------------------------------
# Bundle + determinism
# ---------------------------------------------------------------------------


def test_compute_all_returns_full_bundle():
    df = _tiny_df()
    bundle = topk_diagnostics.compute_all(df)
    assert set(bundle) == {
        "top_k_metrics", "per_ticker_hit_rate",
        "per_quarter_p_k", "prediction_range",
    }


def test_determinism_same_input_same_output():
    df = _tiny_df()
    a = topk_diagnostics.compute_all(df)
    # Permute the input rows; result must be unchanged.
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    b = topk_diagnostics.compute_all(shuffled)
    assert a == b


def test_json_serializable():
    """Every value must round-trip through json — no numpy / Timestamp /
    Period leaks."""
    import json
    df = _tiny_df()
    bundle = topk_diagnostics.compute_all(df)
    s = json.dumps(bundle)
    parsed = json.loads(s)
    assert parsed["top_k_metrics"]["base_rate"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Report-layer wiring
# ---------------------------------------------------------------------------


def test_compute_segment_diagnostics_handles_missing_segment():
    preds = {"eval": _tiny_df()}  # no 'test' segment
    out = compute_segment_diagnostics(preds)
    assert "eval" in out and "test" in out
    assert out["eval"]["top_k_metrics"]["n_rows"] == 50
    assert out["test"]["top_k_metrics"]["n_rows"] == 0


def test_render_segment_diagnostics_appends_sections():
    df = _tiny_df()
    metrics = {"segment_diagnostics": compute_segment_diagnostics({"eval": df})}
    lines: list[str] = []
    _render_segment_diagnostics(lines, metrics)
    md = "\n".join(lines)
    assert "## Top-K precision" in md
    assert "## Per-ticker hit-rate" in md
    assert "## Per-quarter P@5 stability" in md
    assert "## Prediction-range diagnostics" in md
    # Spot-check that the per-day P@1 = 1.0 row made it in.
    assert "1.0000" in md


def test_render_segment_diagnostics_noop_when_absent():
    lines: list[str] = []
    _render_segment_diagnostics(lines, {})
    assert lines == []
