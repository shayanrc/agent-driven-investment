"""Rolling multi-window validation of a rank/equal back-test (V1.2 gate).

`_005` showed rank/equal beats the index on each cell's single OOS window. The
honest objection: one favorable window. This runs ONE clean back-test over the
cell's *full* clean out-of-sample region — its published test.csv PLUS the fresh
post-test region (scored by inference, no retrain; faithfulness self-checked
upstream) — then reports the DISTRIBUTION of rolling H-day **excess returns**
(strategy minus index) off the equity curve.

Why rolling returns off one curve, not fresh-capital sub-windows: short
fresh-capital windows starve the strategy (2-3 trades) and the per-window feed
slicing is brittle. Rolling H-day returns over a single full-OOS equity curve use
the whole curve and answer "across all H-day holding periods, how often / by how
much does it beat the index?". Overlapping windows are autocorrelated (noted).

    uv run python -m scripts.backtests.run_rolling_validation \
        --cell <artifact> --fresh <fresh_predictions.csv> --out <dir> --name <n>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from backtesting.backtest import Backtest
from backtesting.strategy import run_strategy
from scripts.backtests import benchmarks as bm
from scripts.backtests.calibration_step import fit_calibrator
from scripts.backtests.regime_signals import compute_risk_on
from scripts.backtests.run_backtest_cell import INDEX_BY_UNIVERSE, K
from scripts.backtests.run_cell5_bayesian_kelly import (
    INITIAL_CASH,
    LOOKBACK,
    _build_feeds,
    _equity_from_history,
    _gross_exposure_series,
    _load_closes,
    _predictions_dict,
)
from trading_strategies.sizing import DiscreteBoundedLossKelly
from trading_strategies.topk_daily_kelly_label_exit import TopKDailyKellyLabelExit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--fresh", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=K,
                    help="top-K daily picks (default 3); widen to dilute low top-1 precision (_012)")
    ap.add_argument("--sizing-mode", default="equal",
                    choices=["equal", "prob_weight", "kelly", "rank_kelly"],
                    help="position sizing among the top-K (default equal); "
                         "prob_weight sizes each ∝ calibrated p (_013)")
    ap.add_argument("--prob-weight-alpha", type=float, default=1.0,
                    help="sharpen prob_weight: weight ∝ p**alpha (α=1 raw p; α>1 "
                         "concentrates on the highest-p picks) (_014)")
    ap.add_argument("--cost-bps", type=float, default=0.0,
                    help="per-side transaction cost in bps of notional (commission+"
                         "slippage), applied on every fill; round-trip = 2× (_015)")
    ap.add_argument("--regime-signal", default="sma",
                    choices=["sma", "vol", "drawdown", "breadth"],
                    help="regime-gate signal (_017 sma / _018 forward-looking). "
                         "sma=trend (price>SMA, the _017 baseline); vol=realized-vol "
                         "gate (off when vol high); drawdown=off when index >thresh "
                         "below its trailing high; breadth=off when <thresh of the "
                         "universe is above its own MA (leads the index top).")
    ap.add_argument("--regime-ma", type=int, default=0,
                    help="regime gate (_017): trend-SMA window for signal=sma (and "
                         "the per-name MA window for signal=breadth). When >0 (or any "
                         "non-sma signal), mask predictions on risk-OFF days so the "
                         "strategy makes no new entries and decays to cash via normal "
                         "label exits. 0 = off; 200 = classic 200d SMA.")
    ap.add_argument("--regime-slope", type=int, default=0,
                    help="regime gate slope condition (_017, signal=sma): also require "
                         "the SMA RISING (SMA_today > SMA_{today-D}). 0 = price>MA only.")
    ap.add_argument("--regime-window", type=int, default=0,
                    help="window (trading days) for vol (default 20) / drawdown "
                         "(default 60) / breadth per-name MA (default 50) signals. "
                         "0 = signal default.")
    ap.add_argument("--regime-thresh", type=float, default=-1.0,
                    help="threshold for vol (annualized vol, default 0.20) / drawdown "
                         "(frac below high, default 0.05) / breadth (frac of names "
                         "above MA, default 0.50). <0 = signal default.")
    ap.add_argument("--step", type=int, default=5, help="rolling-origin stride (trading days)")
    args = ap.parse_args()
    cell = Path(args.cell); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(exist_ok=True)

    spec = yaml.safe_load((cell / "spec.yaml").read_text())["target"]
    universe = spec["universe"]
    WIN = float(spec["threshold_pct"]) / 100.0
    LOSS = float(spec["max_drawdown"]); HORIZON = int(spec["horizon_days"])
    BREAKEVEN_P = LOSS / (LOSS + WIN)
    idx_ticker, idx_label = INDEX_BY_UNIVERSE[universe]

    cal, _ = fit_calibrator(cell)
    cols = ["date", "ticker", "p_calibrated"]
    frames = [pd.read_csv(cell / "predictions" / "test.csv", parse_dates=["date"])[cols]]
    if args.fresh:
        frames.append(pd.read_csv(args.fresh, parse_dates=["date"])[cols])
    full = (pd.concat(frames).drop_duplicates(["date", "ticker"])
            .sort_values(["date", "ticker"]).reset_index(drop=True))
    preds = _predictions_dict(full, cal)
    tickers = sorted(full.ticker.unique())
    oos_start, oos_end = full.date.min(), full.date.max()
    print(f"[oos] {universe} [{oos_start.date()}..{oos_end.date()}] "
          f"{full.date.nunique()} signal days, H={HORIZON}, index={idx_label}")

    # Regime gate (_017 sma / _018 forward-looking): drop predictions on risk-OFF days
    # → strategy makes no new entries in a confirmed downtrend and decays to cash via
    # its normal DD/target/horizon exits. Universe-market logic lives HERE (harness
    # preprocessing), NOT in the backend-agnostic strategy, per
    # docs/trading_strategies/goal.md ("don't bake universe-specific behavior into a
    # strategy"; "not where universe selection lives").
    n_days_total = full.date.nunique(); n_gated = 0; frac_risk_off = 0.0
    regime_active = args.regime_ma > 0 or args.regime_signal != "sma"
    if regime_active:
        sig = args.regime_signal
        warm = 450  # calendar days — covers SMA200 + vol/breadth lookbacks
        ridx = _load_closes([idx_ticker], oos_start - pd.Timedelta(days=warm),
                            oos_end + pd.Timedelta(days=5))[idx_ticker]
        kw: dict = {}
        roster_wide = None
        if sig == "sma":
            kw = {"ma": args.regime_ma, "slope": args.regime_slope}
            tag = f"SMA{args.regime_ma}" + (f"+slope{args.regime_slope}" if args.regime_slope else "")
        elif sig == "vol":
            kw = {"window": args.regime_window or 20,
                  "thresh": args.regime_thresh if args.regime_thresh >= 0 else 0.20}
            tag = f"vol(w{kw['window']},>{kw['thresh']:.2f}ann)"
        elif sig == "drawdown":
            kw = {"window": args.regime_window or 60,
                  "thresh": args.regime_thresh if args.regime_thresh >= 0 else 0.05}
            tag = f"dd(w{kw['window']},>{kw['thresh']:.0%}off-high)"
        elif sig == "breadth":
            bma = args.regime_window or args.regime_ma or 50
            kw = {"ma": bma,
                  "thresh": args.regime_thresh if args.regime_thresh >= 0 else 0.50}
            tag = f"breadth(ma{bma},<{kw['thresh']:.0%}above)"
            rc = _load_closes(tickers, oos_start - pd.Timedelta(days=int(bma * 2 + 40)),
                              oos_end + pd.Timedelta(days=5))
            roster_wide = pd.DataFrame(rc).sort_index()
        risk_on = compute_risk_on(sig, ridx, roster_closes=roster_wide, **kw)
        ro_map = (risk_on.reindex(sorted(set(risk_on.index) | {pd.Timestamp(d) for d in preds}))
                  .ffill().to_dict())
        kept = {}
        for d, v in preds.items():
            on = ro_map.get(pd.Timestamp(d), True)
            if pd.isna(on):
                on = True  # warmup (pre-OOS / insufficient history) → never gate
            (kept.__setitem__(d, v) if bool(on) else None)
            n_gated += 0 if bool(on) else 1
        preds = kept
        frac_risk_off = n_gated / n_days_total if n_days_total else 0.0
        print(f"[regime] {tag} gate: {n_gated}/{n_days_total} signal "
              f"days risk-OFF ({frac_risk_off:.0%}) → masked (no new entries)")

    closes = _load_closes(tickers, oos_start - pd.Timedelta(days=20),
                          oos_end + pd.Timedelta(days=30))
    idx = _load_closes([idx_ticker], oos_start - pd.Timedelta(days=20),
                       oos_end + pd.Timedelta(days=30))
    closes.update(idx)
    ref = max((s.index for s in closes.values()), key=len)
    comparison_end = min(ref.max(), oos_end)
    roster = {t: s for t, s in closes.items() if t != idx_ticker}

    # ONE full-OOS rank/equal back-test (late positions marked-to-market at end).
    feeds = _build_feeds(roster, oos_start - pd.Timedelta(days=20), comparison_end)
    # Transaction-cost model (_015): per-side bps of notional on every fill
    # (commission + slippage lumped). Round-trip cost = 2 × cost_bps.
    commission_fn = (
        (lambda asset, qty, price: (args.cost_bps / 1e4) * abs(qty) * price)
        if args.cost_bps > 0 else None
    )
    bt = Backtest(feeds, lookback=LOOKBACK, initial_cash=INITIAL_CASH,
                  fill_mode="next_open", gap_policy="ffill_zero_volume",
                  commission_fn=commission_fn)
    strat = TopKDailyKellyLabelExit(
        predictions=preds, K=args.k, target_return=WIN, stop_drawdown=LOSS,
        horizon_days=HORIZON, sizer=DiscreteBoundedLossKelly(), sizer_payoffs=(WIN, LOSS),
        breakeven_p=BREAKEVEN_P, fractional_c=args.c, selection_mode="rank",
        sizing_mode=args.sizing_mode, prob_weight_alpha=args.prob_weight_alpha)
    hist = run_strategy(bt, strat)
    eq = _equity_from_history(hist)
    eq = eq[(eq.index >= oos_start) & (eq.index <= comparison_end)]
    full_sm = bm.compute_metrics(eq)
    n_entry = sum(e.kind == "entry" for e in strat.events)
    avg_exp = float(_gross_exposure_series(hist).mean())

    # Index equity on the same calendar.
    ix = closes[idx_ticker]
    ix = ix[(ix.index >= oos_start) & (ix.index <= comparison_end)]
    ix_eq = INITIAL_CASH * ix / ix.iloc[0]

    # Align strat + index on common dates.
    common = eq.index.intersection(ix_eq.index)
    se = eq.reindex(common).ffill(); xe = ix_eq.reindex(common).ffill()
    dates = list(common)

    # Daily equity dump (_015): the strat + index curves on the common calendar,
    # for the block-bootstrap (honest effective-N) and bear-sub-window analysis.
    pd.DataFrame({"date": [d.date() for d in common],
                  "strat_equity": se.to_numpy(), "idx_equity": xe.to_numpy()}
                 ).to_csv(out / "daily_equity.csv", index=False)

    # Rolling H-day excess returns at stride `step`.
    recs = []
    W = HORIZON
    for i in range(0, len(dates) - W, args.step):
        s_r = se.iloc[i + W] / se.iloc[i] - 1.0
        x_r = xe.iloc[i + W] / xe.iloc[i] - 1.0
        recs.append({"origin": str(dates[i].date()), "end": str(dates[i + W].date()),
                     "strat_ret": round(float(s_r), 4), "idx_ret": round(float(x_r), 4),
                     "excess": round(float(s_r - x_r), 4)})
    r = pd.DataFrame(recs)
    r.to_csv(out / "rolling_windows.csv", index=False)

    # Guard the degenerate case: an OOS shorter than one rolling H-day window
    # yields no rows (recs == []), so r has no 'excess' column. Emit a clear
    # rolling block instead of crashing on r.excess (needs test+fresh to extend
    # the OOS past H for short cells).
    exc = r["excess"].to_numpy() if len(r) else np.empty(0)
    if len(exc):
        rolling_block = {
            "n_windows": len(r), "window_days": W, "stride": args.step,
            "frac_excess_positive": round(float((exc > 0).mean()), 3),
            "median_excess": round(float(np.median(exc)), 4),
            "mean_excess": round(float(exc.mean()), 4),
            "p25_excess": round(float(np.percentile(exc, 25)), 4),
            "p75_excess": round(float(np.percentile(exc, 75)), 4),
            "min_excess": round(float(exc.min()), 4),
            "max_excess": round(float(exc.max()), 4)}
    else:
        rolling_block = {"n_windows": 0, "window_days": W, "stride": args.step,
                         "note": "OOS shorter than one H-day window — pass --fresh to extend"}
    summary = {
        "cell": cell.name, "name": args.name,
        "geometry": {"universe": universe, "horizon": HORIZON, "index": idx_label},
        "config": {"selection_mode": "rank", "sizing_mode": args.sizing_mode,
                   "K": args.k, "prob_weight_alpha": args.prob_weight_alpha,
                   "cost_bps": args.cost_bps, "regime_signal": args.regime_signal,
                   "regime_ma": args.regime_ma, "regime_slope": args.regime_slope,
                   "regime_window": args.regime_window, "regime_thresh": args.regime_thresh,
                   "regime_days_off": n_gated, "regime_frac_off": round(frac_risk_off, 4),
                   "c": args.c, "rolling_window_days": W, "stride": args.step},
        "oos": {"start": str(oos_start.date()), "end": str(comparison_end.date()),
                "signal_days": int(full.date.nunique())},
        "full_oos": {"strat_total_return": round(full_sm["total_return"], 4),
                     "strat_max_dd": round(full_sm["max_dd"], 4),
                     "idx_total_return": round(float(xe.iloc[-1] / xe.iloc[0] - 1), 4),
                     "n_entries": n_entry, "avg_gross_exp": round(avg_exp, 4)},
        "rolling": rolling_block,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"[full-OOS] strat {full_sm['total_return']*100:+.1f}% (DD {full_sm['max_dd']*100:.1f}%) "
          f"vs {idx_label} {summary['full_oos']['idx_total_return']*100:+.1f}% | {n_entry} entries")
    if not len(r):
        print(f"[rolling {W}d] 0 windows — OOS shorter than one H-day window; pass --fresh to extend.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    x = pd.to_datetime(r["origin"])
    ax.bar(x, r.excess * 100, width=4,
           color=["#2a9d4a" if e > 0 else "#c0392b" for e in r.excess])
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel(f"{W}-day excess vs {idx_label} (%)")
    ax.set_title(f"{args.name} — rolling {W}d excess "
                 f"({summary['rolling']['frac_excess_positive']:.0%} positive, "
                 f"median {summary['rolling']['median_excess']*100:+.1f}%)")
    fig.tight_layout(); fig.savefig(out / "figs" / "rolling_excess.png", dpi=130); plt.close(fig)

    print(f"[rolling {W}d] {summary['rolling']['n_windows']} windows | "
          f"{summary['rolling']['frac_excess_positive']:.0%} beat index | "
          f"median excess {summary['rolling']['median_excess']*100:+.1f}% "
          f"(p25 {summary['rolling']['p25_excess']*100:+.1f}%, p75 {summary['rolling']['p75_excess']*100:+.1f}%, "
          f"min {summary['rolling']['min_excess']*100:+.1f}%, max {summary['rolling']['max_excess']*100:+.1f}%)")


if __name__ == "__main__":
    main()
