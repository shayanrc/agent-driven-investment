"""QuarterEndCalendar: reporting-lag-aware quarterly grid."""

from __future__ import annotations

from datetime import date

from data_pipelines.domains.us_fundamentals.calendar import QuarterEndCalendar


def _cal(today: date, lag_days: int = 45) -> QuarterEndCalendar:
    return QuarterEndCalendar(lag_days=lag_days, today_fn=lambda: today)


class TestQuarterEndCalendar:
    def test_lag_gates_the_most_recent_quarter(self):
        # Today 2026-07-03: Q2's 2026-06-30 is only 3 days old (< 45) → not
        # yet demanded; Q1's 2026-03-31 (94 days) is.
        cal = _cal(date(2026, 7, 3))
        days = cal.trading_days(date(2025, 1, 1), date(2026, 7, 3))
        assert days == [
            date(2025, 3, 31), date(2025, 6, 30), date(2025, 9, 30),
            date(2025, 12, 31), date(2026, 3, 31),
        ]

    def test_grid_date_emitted_exactly_at_lag(self):
        # 2026-03-31 + 45 days = 2026-05-15: emitted that day, not before.
        window = (date(2026, 1, 1), date(2026, 12, 31))
        assert date(2026, 3, 31) not in _cal(date(2026, 5, 14)).trading_days(*window)
        assert date(2026, 3, 31) in _cal(date(2026, 5, 15)).trading_days(*window)

    def test_zero_lag_emits_current_quarter_end(self):
        cal = _cal(date(2026, 6, 30), lag_days=0)
        days = cal.trading_days(date(2026, 1, 1), date(2026, 6, 30))
        assert days == [date(2026, 3, 31), date(2026, 6, 30)]

    def test_end_clamps_before_lag_cutoff(self):
        # Requested end earlier than the lag cutoff → end wins.
        cal = _cal(date(2026, 7, 3))
        days = cal.trading_days(date(2025, 1, 1), date(2025, 6, 30))
        assert days == [date(2025, 3, 31), date(2025, 6, 30)]

    def test_empty_when_start_after_end(self):
        cal = _cal(date(2026, 7, 3))
        assert cal.trading_days(date(2026, 7, 1), date(2026, 1, 1)) == []
