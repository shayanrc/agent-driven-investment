"""Runner — cell-5 Bayesian + Kelly back-test (plan §6.6, _001).

End-to-end: fit the Bayesian calibrator on cell-5 VAL, transform test
predictions, run TopKDailyKellyLabelExit through the backtesting engine over
the test slice, compute 3 benchmarks, and persist the _001 artifact +
registry row.

    uv run python -m scripts.backtests.run_cell5_bayesian_kelly

Price basis: the cache's split-adjusted ``close`` (the column the gbdt label
used; D24 price-return basis). OHLCV is read straight from the
data_pipelines SQLite cache via ``gbdt.data._cache_read`` (no domain
registration needed; returns ``adj_close`` too, unused here per D24).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtesting.backtest import Backtest
from backtesting.strategy import run_strategy
from gbdt.data import _cache_read
from scripts.backtests import benchmarks as bm
from scripts.backtests.calibration_step import fit_calibrator, run_checkpoint
from trading_strategies.sizing import DiscreteBoundedLossKelly, VinceOptimalF
from trading_strategies.topk_daily_kelly_label_exit import (
    TopKDailyKellyLabelExit,
)

ARTIFACT = Path(
    "results/gbdt/experiments/"
    "nasdaq100_up_10pct_50d_dd5pct_agentloop_v1.3_revalidation_regen"
)
OUT = Path("results/backtests/_001_cell5_bayesian_kelly")
NDX_TICKER = "INDEX:^NDX"

# Cell-5 label geometry (D6/D11).
WIN, LOSS = 0.10, 0.05
TARGET_RETURN, STOP_DD, HORIZON = 0.10, 0.05, 50
K = 3
FRACTIONAL_C = 0.5
BREAKEVEN_P = 1.0 / (1.0 + WIN / LOSS)  # 1/3
INITIAL_CASH = 100_000.0
LOOKBACK = 5
INPUT_COL = "p_calibrated"


def _predictions_dict(df: pd.DataFrame, cal) -> dict[pd.Timestamp, list[tuple]]:
    """Calibrate a split's p and group into {date: [(ticker,mean,lo,hi)]}."""
    out = cal.transform(df[INPUT_COL].to_numpy())
    g = df.assign(p_mean=out.p_mean, p_low=out.p_low, p_high=out.p_high)
    preds: dict[pd.Timestamp, list[tuple]] = {}
    for date, sub in g.groupby("date"):
        preds[pd.Timestamp(date)] = list(
            zip(sub.ticker, sub.p_mean, sub.p_low, sub.p_high)
        )
    return preds


def _load_closes(tickers, start, end) -> dict[str, pd.Series]:
    closes = {}
    for t in tickers:
        df = _cache_read(t, start.isoformat(), end.isoformat())
        if len(df):
            closes[t] = df.set_index("date")["close"].sort_index()
    return closes


def _build_feeds(closes, start, end):
    """OHLCV feeds (open/high/low/close/volume) from the cache for the engine."""
    feeds_eq = {}
    for t, _s in closes.items():
        df = _cache_read(t, start.isoformat(), end.isoformat())
        df = df.set_index("date").sort_index()
        feeds_eq[t] = pd.DataFrame(
            {
                "open": df["open"],
                "high": df["high"],
                "low": df["low"],
                "close": df["close"],
                "volume": df["volume"].fillna(0.0),
            }
        )
    return {"equities": feeds_eq}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    (OUT / "sizer").mkdir(exist_ok=True)

    # 1-2. Stage-5 checkpoint (fits calibrator on VAL, writes artifact + fig).
    print("[1-2] fitting calibrator on VAL + checkpoint ...")
    chk = run_checkpoint(ARTIFACT, OUT)
    print(f"      checkpoint gate_triggered={chk.gate_triggered} "
          f"ece_eval={chk.ece_eval:.4f} eff_bins={chk.effective_n_bins}")
    cal, col = fit_calibrator(ARTIFACT)
    assert col == INPUT_COL

    # 3. Transform test predictions → dict.
    print("[3] transforming test predictions ...")
    test = pd.read_csv(ARTIFACT / "predictions" / "test.csv", parse_dates=["date"])
    preds = _predictions_dict(test, cal)
    tickers = sorted(test.ticker.unique())
    test_start = test.date.min()
    test_end = test.date.max()

    # 6. Fetch OHLCV. Need [test_start - lookback, test_end + horizon]; pad both.
    fetch_start = test_start - pd.Timedelta(days=20)
    fetch_end = test_end + pd.Timedelta(days=120)  # > 50 BD past test_end
    print(f"[6] reading OHLCV for {len(tickers)} tickers "
          f"[{fetch_start.date()} .. {fetch_end.date()}] ...")
    closes = _load_closes(tickers, fetch_start, fetch_end)
    # NDX index for benchmark #1 — not part of the stock roster, fetch alone.
    ndx = _load_closes([NDX_TICKER], fetch_start, fetch_end)
    closes.update(ndx)
    if NDX_TICKER not in closes:
        print(f"[warn] {NDX_TICKER} not in cache — NDX benchmark will be skipped (R4)")

    # comparison_end = 50 trading days after test_end (capped at data).
    ref_idx = max((s.index for s in closes.values()), key=len)
    after = ref_idx[ref_idx > test_end]
    comparison_end = after[min(HORIZON, len(after) - 1)] if len(after) else ref_idx[-1]
    print(f"      test [{test_start.date()} .. {test_end.date()}] "
          f"comparison_end={comparison_end.date()}")

    # 6b. Pre-flight roster coverage (R5).
    horizon_deadline = comparison_end
    delisted = []
    for t in tickers:
        s = closes.get(t)
        end_t = s.index.max() if s is not None and len(s) else None
        if end_t is None or end_t < horizon_deadline:
            delisted.append({"ticker": t, "data_end": str(end_t.date()) if end_t is not None else None})
    (OUT / "delisted_in_window.json").write_text(
        json.dumps({"horizon_deadline": str(horizon_deadline.date()),
                    "n": len(delisted), "tickers": delisted}, indent=2)
    )
    if delisted:
        print(f"[preflight] {len(delisted)} tickers end before "
              f"{horizon_deadline.date()}: {[d['ticker'] for d in delisted]}")
    else:
        print("[preflight] all tickers cover the full window (no in-window delisting)")

    # 7-8. Build feeds + engine (window through comparison_end). Roster only —
    # NDX is a benchmark, not a tradeable strategy asset.
    roster_closes = {t: s for t, s in closes.items() if t != NDX_TICKER}
    feeds = _build_feeds(roster_closes, fetch_start, comparison_end)
    bt = Backtest(
        feeds, lookback=LOOKBACK, initial_cash=INITIAL_CASH,
        fill_mode="next_open", gap_policy="ffill_zero_volume",
    )

    # 5. (Ablation) eval-replay → Vince fit on realized per-pick returns.
    print("[5] eval-replay for Vince ablation fit ...")
    vince_f = _fit_vince_on_eval(cal, tickers, fetch_start)

    # 9. Run strategy.
    print("[9] running strategy back-test ...")
    strat = TopKDailyKellyLabelExit(
        predictions=preds, K=K, target_return=TARGET_RETURN,
        stop_drawdown=STOP_DD, horizon_days=HORIZON,
        sizer=DiscreteBoundedLossKelly(), sizer_payoffs=(WIN, LOSS),
        breakeven_p=BREAKEVEN_P, fractional_c=FRACTIONAL_C,
    )
    history = run_strategy(bt, strat)
    eq_strategy = _equity_from_history(history)
    eq_strategy = eq_strategy[eq_strategy.index <= comparison_end]

    # 10. Benchmarks.
    print("[10] computing benchmarks ...")
    results = _benchmarks(closes, preds, test_start, comparison_end)

    # Strategy metrics.
    strat_m = bm.compute_metrics(eq_strategy)
    n_entries = sum(e.kind == "entry" for e in strat.events)
    n_exits = sum(e.kind == "exit" for e in strat.events)
    n_trims = sum(e.kind == "trim" for e in strat.events)
    strat_m["n_trades"] = n_entries + n_exits + n_trims
    gross_exp = _gross_exposure_series(history)
    strat_m["gross_exposure_avg"] = float(gross_exp.mean())

    # 11. Persist.
    print("[11] persisting outputs ...")
    _persist(
        eq_strategy, results, strat_m, strat, history, gross_exp,
        vince_f, chk, test_start, test_end, comparison_end,
    )
    _print_headline(strat_m, results)


def _fit_vince_on_eval(cal, tickers, fetch_start) -> dict:
    """Replay the strategy on EVAL with the Kelly sizer; fit Vince on r_i."""
    ev = pd.read_csv(ARTIFACT / "predictions" / "eval.csv", parse_dates=["date"])
    preds = _predictions_dict(ev, cal)
    ev_start, ev_end = ev.date.min(), ev.date.max()
    closes = _load_closes(tickers, ev_start - pd.Timedelta(days=20),
                          ev_end + pd.Timedelta(days=120))
    feeds = _build_feeds(closes, ev_start - pd.Timedelta(days=20),
                         ev_end + pd.Timedelta(days=120))
    bt = Backtest(feeds, lookback=LOOKBACK, initial_cash=INITIAL_CASH,
                  fill_mode="next_open", gap_policy="ffill_zero_volume")
    strat = TopKDailyKellyLabelExit(
        predictions=preds, K=K, target_return=TARGET_RETURN, stop_drawdown=STOP_DD,
        horizon_days=HORIZON, sizer=DiscreteBoundedLossKelly(),
        sizer_payoffs=(WIN, LOSS), breakeven_p=BREAKEVEN_P, fractional_c=FRACTIONAL_C,
    )
    run_strategy(bt, strat)
    # Realized per-pick returns from entry→exit anchored closes.
    rets = _realized_returns(strat)
    if len(rets) and min(rets) < 0:
        v = VinceOptimalF().fit(np.array(rets))
        out = {"f_star": v.per_position_fraction_at_risk, **v.diagnostics,
               "n_eval_picks": len(rets)}
    else:
        out = {"f_star": None, "n_eval_picks": len(rets),
               "note": "no losing eval pick; Vince f undefined"}
    (OUT / "sizer" / "fit.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def _realized_returns(strat) -> list[float]:
    """Per-position realized return = exit_close/anchor − 1 (entries→exit)."""
    anchors = {}
    rets = []
    for e in strat.events:
        if e.kind == "entry":
            anchors[e.ticker] = e.anchor_close
        elif e.kind == "exit" and e.ticker in anchors:
            a = anchors.pop(e.ticker)
            if a and e.close:
                rets.append(e.close / a - 1.0)
    return rets


def _equity_from_history(history) -> pd.Series:
    rows = {pd.Timestamp(s["timestamp"]): float(s["portfolio"]["equity"])
            for (s, _d, _i) in history}
    return pd.Series(rows).sort_index()


def _gross_exposure_series(history) -> pd.Series:
    rows = {}
    for (s, _d, _i) in history:
        pf = s["portfolio"]
        eq = float(pf["equity"])
        rows[pd.Timestamp(s["timestamp"])] = (eq - float(pf["cash"])) / eq if eq > 0 else 0.0
    return pd.Series(rows).sort_index()


def _benchmarks(closes, preds, start, end) -> dict:
    out = {}
    timeline = max((s.index for s in closes.values()), key=len)
    timeline = timeline[(timeline >= start) & (timeline <= end)]
    # 1. NDX
    if NDX_TICKER in closes:
        eq, m, _ = bm.buy_and_hold(closes[NDX_TICKER], start, end, INITIAL_CASH)
        out["ndx_bh"] = {"equity": eq, "metrics": m}
    else:
        out["ndx_bh"] = {"equity": None, "metrics": None}
    # 2. equal-weight basket (exclude the index ticker)
    basket_closes = {t: s for t, s in closes.items() if t != NDX_TICKER}
    eq, m, _ = bm.equal_weight_basket(basket_closes, start, end, INITIAL_CASH)
    out["ew_basket"] = {"equity": eq, "metrics": m}
    # 3. event-driven top-K
    eq, m, _ = bm.event_driven_topk(
        preds, basket_closes, K=K, target_return=TARGET_RETURN,
        stop_drawdown=STOP_DD, horizon_days=HORIZON, breakeven_p=BREAKEVEN_P,
        timeline=timeline, initial_cash=INITIAL_CASH,
    )
    out["ew_topk_no_kelly"] = {"equity": eq, "metrics": m}
    return out


def _persist(eq_strat, results, strat_m, strat, history, gross_exp, vince_f,
             chk, test_start, test_end, comparison_end) -> None:
    eq_strat.to_csv(OUT / "equity_curve.csv", header=["equity"])
    # picks.csv from events
    ev_rows = [asdict(e) for e in strat.events]
    pd.DataFrame(ev_rows).to_csv(OUT / "picks.csv", index=False)
    # fills.csv from history infos
    fills = []
    for (s, _d, info) in history:
        for f in info.get("fills", []):
            fills.append({"date": str(pd.Timestamp(s["timestamp"]).date()), **f})
    pd.DataFrame(fills).to_csv(OUT / "fills.csv", index=False)
    gross_exp.to_csv(OUT / "gross_exposure.csv", header=["gross_exposure"])

    # headline.csv
    def row(name, m, n):
        return {"line": name, "end_$": round(m["end"], 2),
                "total_pct": round(m["total_return"] * 100, 2),
                "cagr_pct": round(m["cagr"] * 100, 2),
                "max_dd_pct": round(m["max_dd"] * 100, 2), "n_trades": n}
    head = [row("Strategy (Bayesian+Kelly c=0.5)", strat_m, strat_m["n_trades"])]
    if results["ndx_bh"]["metrics"]:
        head.append(row("NDX buy-and-hold", results["ndx_bh"]["metrics"], 1))
    head.append(row("92-ticker equal-weight basket", results["ew_basket"]["metrics"],
                    results["ew_basket"]["metrics"]["n_trades"]))
    head.append(row("Equal-weight top-K (no Kelly)", results["ew_topk_no_kelly"]["metrics"],
                    results["ew_topk_no_kelly"]["metrics"]["n_trades"]))
    pd.DataFrame(head).to_csv(OUT / "headline.csv", index=False)

    # summary.json
    summary = {
        "window": {"test_start": str(test_start.date()), "test_end": str(test_end.date()),
                   "comparison_end": str(comparison_end.date())},
        "strategy": strat_m,
        "benchmarks": {k: v["metrics"] for k, v in results.items()},
        "turnover": {"entries": sum(e.kind == "entry" for e in strat.events),
                     "exits": sum(e.kind == "exit" for e in strat.events),
                     "trims": sum(e.kind == "trim" for e in strat.events)},
        "exit_triggers": _trigger_counts(strat),
        "vince_ablation": vince_f,
        "checkpoint": asdict(chk),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    _figs(eq_strat, results, gross_exp)


def _trigger_counts(strat) -> dict:
    from collections import Counter
    return dict(Counter(e.trigger for e in strat.events if e.kind == "exit"))


def _figs(eq_strat, results, gross_exp) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eq_strat.index, eq_strat.values, label="Strategy (Bayesian+Kelly)", lw=2)
    for key, label in [("ndx_bh", "NDX buy-and-hold"),
                       ("ew_basket", "EW basket"),
                       ("ew_topk_no_kelly", "EW top-K (no Kelly)")]:
        e = results[key]["equity"]
        if e is not None:
            ax.plot(e.index, e.values, label=label, lw=1, alpha=0.8)
    ax.axhline(INITIAL_CASH, color="gray", ls="--", lw=0.7)
    ax.set_title("Cell-5 Bayesian+Kelly vs benchmarks (gross, $100K start)")
    ax.set_ylabel("equity ($)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "equity_overlay.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(gross_exp.index, gross_exp.values, alpha=0.4)
    ax.set_title("Strategy gross exposure trajectory (Path A sawtooth)")
    ax.set_ylabel("gross exposure")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "gross_exposure.png", dpi=130)
    plt.close(fig)

    dd = eq_strat / eq_strat.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(dd.index, dd.values, color="C3", alpha=0.4)
    ax.set_title("Strategy drawdown trajectory")
    ax.set_ylabel("drawdown")
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "drawdown.png", dpi=130)
    plt.close(fig)


def _print_headline(strat_m, results) -> None:
    print("\n=== HEADLINE (gross, $100,000 start) ===")
    print(f"Strategy            end ${strat_m['end']:,.0f}  "
          f"{strat_m['total_return']*100:+.1f}%  maxDD {strat_m['max_dd']*100:.1f}%  "
          f"n_trades {strat_m['n_trades']}")
    for k, label in [("ndx_bh", "NDX buy-hold     "),
                     ("ew_basket", "EW basket        "),
                     ("ew_topk_no_kelly", "EW top-K no-Kelly")]:
        m = results[k]["metrics"]
        if m:
            print(f"{label}   end ${m['end']:,.0f}  {m['total_return']*100:+.1f}%  "
                  f"maxDD {m['max_dd']*100:.1f}%  n_trades {m['n_trades']}")


if __name__ == "__main__":
    main()
