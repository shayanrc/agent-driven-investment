"""Cell-agnostic back-test on a gbdt cell's PUBLISHED test-window predictions.

Generalizes the `_001` runner across cells: reads the label geometry
(universe / threshold_pct / max_drawdown / horizon_days) from the cell's
spec.yaml, derives the Kelly payoffs + breakeven from it, picks the benchmark
index by universe, and back-tests the cell's `predictions/test.csv` on the
window it was scored on (comparison_end = test_end + horizon business days,
clipped to data availability — late positions marked-to-market).

Used for (a) re-running `_001` under the new headline sizing and (b) surveying
the best agent cells from the R-p@K registry. The calibrator is fit on each
cell's VAL split (leak-free). Selection bound + fractional Kelly are CLI knobs.

    uv run python -m scripts.backtests.run_backtest_cell \
        --cell <artifact_dir> --out <dir> --name <short> --c 0.25 --selection-bound mean
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from backtesting.backtest import Backtest
from backtesting.strategy import run_strategy
from scripts.backtests import benchmarks as bm
from scripts.backtests.calibration_step import fit_calibrator
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

K = 3
# Benchmark index by universe. ^RUI (Russell-1000) is not in the cache, so
# russell1000 is proxied by ^SPX (large-cap) — flagged in the summary.
INDEX_BY_UNIVERSE = {
    "nasdaq100": ("INDEX:^NDX", "NDX"),
    "sp500": ("INDEX:^SPX", "SPX"),
    "russell1000": ("INDEX:^SPX", "SPX (proxy: ^RUI uncached)"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--c", type=float, default=0.25)
    ap.add_argument("--selection-bound", default="mean", choices=["mean", "low"])
    ap.add_argument("--predictions", default=None,
                    help="default: <cell>/predictions/test.csv")
    args = ap.parse_args()
    cell = Path(args.cell)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); (out / "figs").mkdir(exist_ok=True)

    spec = yaml.safe_load((cell / "spec.yaml").read_text())
    tgt = spec["target"]
    universe = tgt["universe"]
    WIN = float(tgt["threshold_pct"]) / 100.0
    LOSS = float(tgt["max_drawdown"])
    HORIZON = int(tgt["horizon_days"])
    TARGET_RETURN, STOP_DD = WIN, LOSS
    BREAKEVEN_P = LOSS / (LOSS + WIN)  # b = WIN/LOSS → 1/(1+b)
    idx_ticker, idx_label = INDEX_BY_UNIVERSE[universe]
    print(f"[geometry] {universe} WIN={WIN} LOSS={LOSS} H={HORIZON} breakeven={BREAKEVEN_P:.3f} "
          f"index={idx_label} | c={args.c} sel={args.selection_bound}")

    # Calibrator on VAL; predictions = published test.csv.
    cal, _ = fit_calibrator(cell)
    pred_csv = Path(args.predictions) if args.predictions else cell / "predictions" / "test.csv"
    test = pd.read_csv(pred_csv, parse_dates=["date"])
    preds = _predictions_dict(test, cal)
    tickers = sorted(test.ticker.unique())
    t_start, t_end = test.date.min(), test.date.max()
    print(f"[preds] {pred_csv.name}: [{t_start.date()}..{t_end.date()}] "
          f"{test.date.nunique()} dates, {len(tickers)} tickers")

    # OHLCV: roster + index, wide enough to resolve the last signal's horizon.
    closes = _load_closes(tickers, t_start - pd.Timedelta(days=20),
                          t_end + pd.Timedelta(days=int(HORIZON * 2.2) + 30))
    idx = _load_closes([idx_ticker], t_start - pd.Timedelta(days=20),
                       t_end + pd.Timedelta(days=int(HORIZON * 2.2) + 30))
    closes.update(idx)
    ref = max((s.index for s in closes.values()), key=len)
    data_end = ref.max()
    # comparison_end = test_end + HORIZON business days, clipped to data end.
    after = ref[ref > t_end]
    comparison_end = after[HORIZON - 1] if len(after) >= HORIZON else data_end
    comparison_end = min(comparison_end, data_end)
    print(f"[window] data_end={data_end.date()} comparison_end={comparison_end.date()}")

    roster = {t: s for t, s in closes.items() if t != idx_ticker}
    feeds = _build_feeds(roster, t_start - pd.Timedelta(days=20), comparison_end)
    bt = Backtest(feeds, lookback=LOOKBACK, initial_cash=INITIAL_CASH,
                  fill_mode="next_open", gap_policy="ffill_zero_volume")

    strat = TopKDailyKellyLabelExit(
        predictions=preds, K=K, target_return=TARGET_RETURN, stop_drawdown=STOP_DD,
        horizon_days=HORIZON, sizer=DiscreteBoundedLossKelly(), sizer_payoffs=(WIN, LOSS),
        breakeven_p=BREAKEVEN_P, fractional_c=args.c, selection_bound=args.selection_bound,
    )
    history = run_strategy(bt, strat)
    eq = _equity_from_history(history); eq = eq[eq.index <= comparison_end]

    timeline = ref[(ref >= t_start) & (ref <= comparison_end)]
    res = {}
    if idx_ticker in closes:
        e, m, _ = bm.buy_and_hold(closes[idx_ticker], t_start, comparison_end, INITIAL_CASH)
        res["index_bh"] = {"equity": e, "metrics": m}
    basket = {t: s for t, s in closes.items() if t != idx_ticker}
    e, m, _ = bm.equal_weight_basket(basket, t_start, comparison_end, INITIAL_CASH)
    res["ew_basket"] = {"equity": e, "metrics": m}
    e, m, _ = bm.event_driven_topk(preds, basket, K=K, target_return=TARGET_RETURN,
                                   stop_drawdown=STOP_DD, horizon_days=HORIZON,
                                   breakeven_p=BREAKEVEN_P, timeline=timeline,
                                   initial_cash=INITIAL_CASH)
    res["ew_topk_no_kelly"] = {"equity": e, "metrics": m}

    sm = bm.compute_metrics(eq)
    n_entry = sum(e.kind == "entry" for e in strat.events)
    n_exit = sum(e.kind == "exit" for e in strat.events)
    n_trim = sum(e.kind == "trim" for e in strat.events)
    sm["n_trades"] = n_entry + n_exit + n_trim
    ge = _gross_exposure_series(history); sm["gross_exposure_avg"] = float(ge.mean())
    n_tickers_entered = len({e.ticker for e in strat.events if e.kind == "entry"})

    eq.to_csv(out / "equity_curve.csv", header=["equity"])
    pd.DataFrame([asdict(e) for e in strat.events]).to_csv(out / "picks.csv", index=False)
    summary = {
        "cell": cell.name,
        "config": {"fractional_c": args.c, "selection_bound": args.selection_bound, "K": K},
        "geometry": {"universe": universe, "win": WIN, "loss": LOSS, "horizon": HORIZON,
                     "breakeven_p": BREAKEVEN_P, "index": idx_label},
        "window": {"test_start": str(t_start.date()), "test_end": str(t_end.date()),
                   "comparison_end": str(comparison_end.date()), "data_end": str(data_end.date())},
        "strategy": sm,
        "benchmarks": {k: v["metrics"] for k, v in res.items()},
        "turnover": {"entries": n_entry, "exits": n_exit, "trims": n_trim,
                     "unique_tickers": n_tickers_entered,
                     "open_at_end": len(strat._open)},
        "exit_triggers": dict(Counter(e.trigger for e in strat.events if e.kind == "exit")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eq.index, eq.values, label=f"Strategy c={args.c}", lw=2)
    for key, lab in [("index_bh", f"{idx_label} buy-hold"), ("ew_basket", "EW basket"),
                     ("ew_topk_no_kelly", "EW top-K (no Kelly)")]:
        e = res.get(key, {}).get("equity")
        if e is not None:
            ax.plot(e.index, e.values, label=lab, lw=1, alpha=0.8)
    ax.axhline(INITIAL_CASH, color="gray", ls="--", lw=0.7)
    ax.set_title(f"{args.name} ({universe}, c={args.c}) vs benchmarks ($100K, gross)")
    ax.set_ylabel("equity ($)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figs" / "equity_overlay.png", dpi=130); plt.close(fig)

    print(f"=== {args.name} ($100K, gross) ===")
    print(f"Strategy   {sm['total_return']*100:+.1f}%  DD {sm['max_dd']*100:.1f}%  "
          f"exp {sm['gross_exposure_avg']:.2f}  {n_entry}ent/{n_tickers_entered}tk  triggers={summary['exit_triggers']}")
    for k, lab in [("index_bh", idx_label), ("ew_basket", "EWbasket"), ("ew_topk_no_kelly", "EWtopK")]:
        m = res.get(k, {}).get("metrics")
        if m:
            print(f"  {lab:10s} {m['total_return']*100:+.1f}%  DD {m['max_dd']*100:.1f}%")


if __name__ == "__main__":
    main()
