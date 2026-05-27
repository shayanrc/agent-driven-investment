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
    """Per-day P@k under the corrected ``min(R(d), k)`` formula.

    Fixture: each day has R(d) = 2 positives (A always; B on even days,
    C on odd days). 10 days, 5 tickers/day, 20 positives total.
    base_rate = 20/50 = 0.4.

    k=1: per-day denom = sum_d(min(R(d), 1)) = 10*1 = 10. Hits = 10
      (A picked every day, always positive). P@1 = 10/10 = 1.0;
      lift = 2.5. n_days_R_lt_k = 0 (every day has R >= 1).
    k=5: per-day denom = sum_d(min(R(d), 5)) = 10*2 = 20. Hits = 20
      (top-5 each day captures all positives since k=5 == n_tickers).
      P@5 = 20/20 = 1.0; lift = 2.5. n_days_R_lt_k = 10 (R=2 < 5 every
      day). Under the old buggy formula this was 20/50 = 0.4.

    Formula spec: ``.claude/memories/project-r-precision-methodology.md``.
    """
    df = _tiny_df()
    out = topk_diagnostics.compute_top_k_metrics(df, k_values=(1, 5))
    assert out["formula_version"] == "v2_min_R_d_k"
    assert out["n_rows"] == 50
    assert out["base_rate"] == pytest.approx(0.4)
    p1 = out["per_day"]["1"]
    assert p1["p_at_k"] == pytest.approx(1.0)  # 10/10
    assert p1["n_picks_total"] == 10
    assert p1["n_positives_in_picks"] == 10
    assert p1["n_denom"] == 10  # sum_d min(R(d), 1) = sum_d min(2, 1) = 10
    assert p1["n_days_R_lt_k"] == 0  # R(d) = 2 >= 1 every day
    assert p1["n_days_full_k"] == 10
    assert p1["n_days_total"] == 10
    assert p1["lift"] == pytest.approx(2.5)
    p5 = out["per_day"]["5"]
    assert p5["p_at_k"] == pytest.approx(1.0)  # 20/20 (new) vs 20/50 = 0.4 (old)
    assert p5["n_picks_total"] == 50
    assert p5["n_positives_in_picks"] == 20
    assert p5["n_denom"] == 20  # sum_d min(R(d), 5) = sum_d min(2, 5) = 20
    assert p5["n_days_R_lt_k"] == 10  # R(d) = 2 < 5 every day
    assert p5["lift"] == pytest.approx(2.5)  # 1.0 / 0.4


def test_top_k_global_hand_checkable():
    """Global top-5 by score across the whole panel.

    All ten (A, D*) rows share p_calibrated = 0.9. ``_sorted_by_score``
    sorts by (p_calibrated desc, ticker asc) with stable mergesort, so
    the first five rows of the sort are (A, D0)..(A, D4) — all positives.

    Global P@5 denominator under the corrected formula:
    ``min(k=5, total_positives=20) = 5``. Hits = 5. P@5 = 5/5 = 1.0;
    lift = 1.0 / 0.4 = 2.5. (For this fixture the corrected and original
    formulas agree because total_positives >= k.)
    """
    df = _tiny_df()
    out = topk_diagnostics.compute_top_k_metrics(df, k_values=(1, 5))
    g5 = out["global"]["5"]
    assert g5["n_picks"] == 5
    assert g5["n_positives_in_picks"] == 5
    assert g5["n_denom"] == 5  # min(5, 20 total positives)
    assert g5["p_at_k"] == pytest.approx(1.0)
    assert g5["lift"] == pytest.approx(2.5)


def test_top_k_empty_segment():
    df = pd.DataFrame(columns=["date", "ticker", "p_raw", "p_calibrated",
                                 "y_true", "sample_weight"])
    out = topk_diagnostics.compute_top_k_metrics(df)
    assert out["formula_version"] == "v2_min_R_d_k"
    assert out["n_rows"] == 0
    assert out["base_rate"] is None
    for k in ("1", "5", "10"):
        assert out["per_day"][k]["p_at_k"] is None
        assert out["per_day"][k]["n_picks_total"] == 0
        assert out["per_day"][k]["n_denom"] == 0
        assert out["per_day"][k]["n_days_R_lt_k"] == 0
        assert out["global"][k]["p_at_k"] is None
        assert out["global"][k]["n_denom"] == 0


def test_top_k_days_with_fewer_than_k_tickers():
    """Staggered panel with R(d) < k — exercises the corrected formula.

    Day 0: 2 tickers (A pos, B neg). R(0) = 1.
    Day 1: 5 tickers (A pos, B pos, C neg, D neg, E neg). R(1) = 2.

    k=5:
      Picks-made = 2 + 5 = 7 (the OLD buggy denominator).
      Corrected denom = sum_d(min(R(d), 5)) = min(1, 5) + min(2, 5) = 3.
      Hits: day 0 top-5 = {A, B}, 1 positive. Day 1 top-5 = all rows,
        2 positives. Total hits = 3.
      P@5 = 3 / 3 = 1.0 under the corrected formula.
      (OLD formula would have given 3 / 7 = 0.4286 — both days had
       R < k, so the staggered-panel under-count is maximal here.)
      n_days_R_lt_k = 2 (both days). n_days_full_k = 1 (day 1 only).
    """
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
    assert pd5["n_denom"] == 3  # min(1,5) + min(2,5)
    assert pd5["n_days_R_lt_k"] == 2  # both days have R < k
    assert pd5["p_at_k"] == pytest.approx(1.0)  # 3/3 (new) vs 3/7 (old buggy)


def test_top_k_formula_version_field_present():
    """``formula_version == "v2_min_R_d_k"`` is the schema marker that
    distinguishes post-fix (issue #45) artifacts from pre-fix ones.

    Pre-fix ``metrics.json::segment_diagnostics::top_k_metrics`` blocks
    have NO ``formula_version`` field — absence = v1 (the buggy
    ``min(k, n_tickers_in_day)`` denominator). Post-fix blocks always
    carry the field, on populated and empty segments alike.
    """
    df = _tiny_df()
    out = topk_diagnostics.compute_top_k_metrics(df, k_values=(1, 5))
    assert out["formula_version"] == "v2_min_R_d_k"
    # And on an empty segment.
    empty_df = pd.DataFrame(
        columns=["date", "ticker", "p_raw", "p_calibrated",
                 "y_true", "sample_weight"]
    )
    empty_out = topk_diagnostics.compute_top_k_metrics(empty_df)
    assert empty_out["formula_version"] == "v2_min_R_d_k"


def test_top_k_staggered_panel_new_formula_higher_than_old():
    """On a staggered panel with R(d) << k, the corrected formula yields
    a STRICTLY HIGHER P@k than the old picks-made denominator (assuming
    the model has any signal at all).

    Construction: 4 days, each with a different number of tickers (R(d)
    is 1 every day — only ticker A is positive). The model ranks A on
    top each day. Picks-made denom = 2+3+4+5 = 14; achievable-positives
    denom = min(1, k=5) * 4 = 4. Hits = 4 (A picked + positive every
    day).

    Corrected P@5 = 4/4 = 1.0
    Old buggy P@5 = 4/14 ≈ 0.286

    This is the cross-market memo's NSE pattern in miniature.
    """
    rows = []
    panels = [
        ("2024-01-01", ["A", "B"]),
        ("2024-01-02", ["A", "B", "C"]),
        ("2024-01-03", ["A", "B", "C", "D"]),
        ("2024-01-04", ["A", "B", "C", "D", "E"]),
    ]
    score_map = {"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1}
    for date_str, tickers_day in panels:
        for tk in tickers_day:
            rows.append({
                "date": date_str, "ticker": tk,
                "p_raw": score_map[tk], "p_calibrated": score_map[tk],
                "y_true": 1 if tk == "A" else 0,
                "sample_weight": 1.0,
            })
    df = pd.DataFrame(rows)
    out = topk_diagnostics.compute_top_k_metrics(df, k_values=(5,))
    pd5 = out["per_day"]["5"]

    # New formula: corrected denominator
    assert pd5["n_picks_total"] == 14  # 2 + 3 + 4 + 5
    assert pd5["n_denom"] == 4  # sum_d min(R(d)=1, 5) = 4
    assert pd5["n_positives_in_picks"] == 4  # A picked + positive every day
    assert pd5["n_days_R_lt_k"] == 4  # every day has R=1 < 5
    new_pk = pd5["p_at_k"]
    assert new_pk == pytest.approx(1.0)  # 4 / 4

    # Verify that the new formula is strictly higher than the old buggy
    # denominator would have given (4 / 14 ≈ 0.286).
    old_buggy_pk = pd5["n_positives_in_picks"] / pd5["n_picks_total"]
    assert old_buggy_pk == pytest.approx(4 / 14)
    assert new_pk > old_buggy_pk


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
