"""Fresh-OOS back-test on inferred predictions (plan §10 Q4).

Consumes a fresh-prediction CSV produced by
``scripts.backtests.infer_fresh_predictions`` (the trained model scored on the
refreshed panel, for dates AFTER the cell's published test window) and runs the
same TopKDailyKellyLabelExit strategy + benchmarks as ``_001``.

Differences from the ``_001`` runner:
  * predictions come from the fresh CSV, not the cell's predictions/test.csv;
  * the calibrator is still fit on the cell's VAL split (leak-free);
  * the window is the fresh signal range; comparison_end is the data end, so
    positions opened too late to complete 50 BD are marked-to-market (open) at
    the end — the report states the resolved-vs-MTM split.

    uv run python -m scripts.backtests.run_fresh_oos \
        --cell <artifact_dir> --predictions <fresh.csv> --out <dir> --name <short>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtesting.backtest import Backtest
from backtesting.strategy import run_strategy
from scripts.backtests import benchmarks as bm
from scripts.backtests.calibration_step import fit_calibrator
from scripts.backtests.run_cell5_bayesian_kelly import (
    BREAKEVEN_P,
    FRACTIONAL_C,
    HORIZON,
    INITIAL_CASH,
    K,
    LOOKBACK,
    LOSS,
    NDX_TICKER,
    STOP_DD,
    TARGET_RETURN,
    WIN,
    _build_feeds,
    _equity_from_history,
    _gross_exposure_series,
    _load_closes,
    _predictions_dict,
)
from trading_strategies.sizing import DiscreteBoundedLossKelly
from trading_strategies.topk_daily_kelly_label_exit import (
    TopKDailyKellyLabelExit,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="cell5b_fresh_oos")
    ap.add_argument("--c", type=float, default=FRACTIONAL_C, help="fractional Kelly c")
    ap.add_argument("--selection-bound", default="mean", choices=["mean", "low"],
                    help="entry filter clears breakeven against p_mean or p_low")
    ap.add_argument("--horizon", type=int, default=None,
                    help="label-exit horizon in trading days; default reads the "
                         "cell's target.horizon_days (the _001 HORIZON constant is "
                         "only the fallback). +50%%/50d and +20%%/25d champions differ.")
    args = ap.parse_args()
    cell = Path(args.cell)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(exist_ok=True)

    # Per-cell horizon: champions differ (50d vs 25d), so use the cell's own
    # target.horizon_days rather than the hardcoded _001 constant (bug: champ20
    # was backtested at HORIZON=50). CLI --horizon overrides.
    import yaml
    _spec = yaml.safe_load((cell / "spec.yaml").read_text())
    horizon = args.horizon or int(_spec.get("target", {}).get("horizon_days", HORIZON))
    print(f"[cfg] horizon_days={horizon} (cell spec) "
          f"target_return={TARGET_RETURN} stop_dd={STOP_DD} K={K}")

    # 1. Calibrator fit on the cell's VAL (leak-free), transform fresh preds.
    print("[1] fitting calibrator on cell VAL ...")
    cal, col = fit_calibrator(cell)
    fresh = pd.read_csv(args.predictions, parse_dates=["date"])
    preds = _predictions_dict(fresh, cal)
    tickers = sorted(fresh.ticker.unique())
    fresh_start, fresh_end = fresh.date.min(), fresh.date.max()
    print(f"    fresh signals [{fresh_start.date()} .. {fresh_end.date()}] "
          f"{fresh.date.nunique()} dates, {len(tickers)} tickers")

    # 2. OHLCV through the data end (= comparison_end; late positions MTM).
    closes = _load_closes(tickers, fresh_start - pd.Timedelta(days=20),
                          fresh_end + pd.Timedelta(days=120))
    ndx = _load_closes([NDX_TICKER], fresh_start - pd.Timedelta(days=20),
                       fresh_end + pd.Timedelta(days=120))
    closes.update(ndx)
    ref = max((s.index for s in closes.values()), key=len)
    comparison_end = ref.max()
    n_resolvable = int((ref[(ref > fresh_start)] <= comparison_end).sum())
    # signals that can complete a full 50-BD horizon by the data end:
    last_full = ref[ref <= comparison_end]
    full_resolve_cutoff = last_full[-(horizon + 1)] if len(last_full) > horizon else fresh_start
    print(f"    comparison_end={comparison_end.date()} "
          f"(full {horizon}-BD resolution only for signals <= {full_resolve_cutoff.date()})")

    # 3. Feeds (roster only) + engine.
    roster = {t: s for t, s in closes.items() if t != NDX_TICKER}
    feeds = _build_feeds(roster, fresh_start - pd.Timedelta(days=20), comparison_end)
    bt = Backtest(feeds, lookback=LOOKBACK, initial_cash=INITIAL_CASH,
                  fill_mode="next_open", gap_policy="ffill_zero_volume")

    # 4. Run strategy.
    print("[4] running fresh-OOS strategy ...")
    print(f"    c={args.c} selection_bound={args.selection_bound}")
    strat = TopKDailyKellyLabelExit(
        predictions=preds, K=K, target_return=TARGET_RETURN, stop_drawdown=STOP_DD,
        horizon_days=horizon, sizer=DiscreteBoundedLossKelly(), sizer_payoffs=(WIN, LOSS),
        breakeven_p=BREAKEVEN_P, fractional_c=args.c, selection_bound=args.selection_bound,
    )
    history = run_strategy(bt, strat)
    eq = _equity_from_history(history)
    eq = eq[eq.index <= comparison_end]

    # 5. Benchmarks over [fresh_start, comparison_end].
    print("[5] benchmarks ...")
    timeline = ref[(ref >= fresh_start) & (ref <= comparison_end)]
    res = {}
    if NDX_TICKER in closes:
        e, m, _ = bm.buy_and_hold(closes[NDX_TICKER], fresh_start, comparison_end, INITIAL_CASH)
        res["ndx_bh"] = {"equity": e, "metrics": m}
    basket = {t: s for t, s in closes.items() if t != NDX_TICKER}
    e, m, _ = bm.equal_weight_basket(basket, fresh_start, comparison_end, INITIAL_CASH)
    res["ew_basket"] = {"equity": e, "metrics": m}
    e, m, _ = bm.event_driven_topk(
        preds, basket, K=K, target_return=TARGET_RETURN, stop_drawdown=STOP_DD,
        horizon_days=horizon, breakeven_p=BREAKEVEN_P, timeline=timeline,
        initial_cash=INITIAL_CASH)
    res["ew_topk_no_kelly"] = {"equity": e, "metrics": m}

    # 6. Metrics + persist.
    sm = bm.compute_metrics(eq)
    n_entry = sum(e.kind == "entry" for e in strat.events)
    n_exit = sum(e.kind == "exit" for e in strat.events)
    n_trim = sum(e.kind == "trim" for e in strat.events)
    sm["n_trades"] = n_entry + n_exit + n_trim
    ge = _gross_exposure_series(history)
    sm["gross_exposure_avg"] = float(ge.mean())
    # open (unresolved) positions at end
    n_open_end = len(strat._open)

    eq.to_csv(out / "equity_curve.csv", header=["equity"])
    pd.DataFrame([asdict(e) for e in strat.events]).to_csv(out / "picks.csv", index=False)
    ge.to_csv(out / "gross_exposure.csv", header=["gross_exposure"])

    from collections import Counter
    summary = {
        "config": {"fractional_c": args.c, "selection_bound": args.selection_bound,
                   "K": K, "target_return": TARGET_RETURN, "stop_drawdown": STOP_DD,
                   "horizon_days": horizon},
        "window": {"fresh_start": str(fresh_start.date()), "fresh_end": str(fresh_end.date()),
                   "comparison_end": str(comparison_end.date()),
                   "full_resolve_cutoff": str(full_resolve_cutoff.date())},
        "strategy": sm,
        "benchmarks": {k: v["metrics"] for k, v in res.items()},
        "turnover": {"entries": n_entry, "exits": n_exit, "trims": n_trim,
                     "open_at_end_unresolved": n_open_end},
        "exit_triggers": dict(Counter(e.trigger for e in strat.events if e.kind == "exit")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # figure
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eq.index, eq.values, label="Strategy (Bayesian+Kelly)", lw=2)
    for key, lab in [("ndx_bh", "NDX buy-hold"), ("ew_basket", "EW basket"),
                     ("ew_topk_no_kelly", "EW top-K (no Kelly)")]:
        e = res.get(key, {}).get("equity")
        if e is not None:
            ax.plot(e.index, e.values, label=lab, lw=1, alpha=0.8)
    ax.axhline(INITIAL_CASH, color="gray", ls="--", lw=0.7)
    ax.set_title(f"Fresh-OOS {args.name} vs benchmarks ($100K, gross)")
    ax.set_ylabel("equity ($)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figs" / "equity_overlay.png", dpi=130); plt.close(fig)

    # headline print
    print("\n=== FRESH-OOS HEADLINE ($100,000 start, gross) ===")
    print(f"window {fresh_start.date()} -> {comparison_end.date()} | "
          f"{n_entry} entries / {n_exit} exits / {n_trim} trims | {n_open_end} open(unresolved) at end")
    print(f"Strategy          end ${sm['end']:,.0f}  {sm['total_return']*100:+.1f}%  "
          f"maxDD {sm['max_dd']*100:.1f}%  avg_gross_exp {sm['gross_exposure_avg']:.2f}")
    for k, lab in [("ndx_bh", "NDX buy-hold     "), ("ew_basket", "EW basket        "),
                   ("ew_topk_no_kelly", "EW top-K no-Kelly")]:
        m = res.get(k, {}).get("metrics")
        if m:
            print(f"{lab}   end ${m['end']:,.0f}  {m['total_return']*100:+.1f}%  maxDD {m['max_dd']*100:.1f}%")
    print(f"\nexit triggers: {summary['exit_triggers']}")


if __name__ == "__main__":
    main()
