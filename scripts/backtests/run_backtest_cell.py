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
from scripts.backtests.plot_actions import plot_actions
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
    # NSE universes: indices use the NIFTY: prefix (not INDEX:), routed to the
    # nse_equities table by gbdt.data._cache_read. NIFTY:500 is the Nifty 500
    # total-market index — the natural benchmark for the nifty500 universe.
    "nifty500": ("NIFTY:500", "NIFTY500"),
    "nifty50": ("NIFTY:50", "NIFTY50"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--c", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=3,
                    help="top-K positions entered per day (default 3 = champion). In "
                         "equal sizing the per-name slice is c·gross_cap/K.")
    ap.add_argument("--selection-bound", default="mean", choices=["mean", "low"])
    ap.add_argument("--selection-mode", default="breakeven", choices=["breakeven", "rank"],
                    help="rank = top-K by p-rank, no absolute breakeven gate (V1.2)")
    ap.add_argument("--sizing-mode", default="kelly",
                    choices=["kelly", "equal", "rank_kelly", "inverse_vol", "prob_weight"])
    ap.add_argument("--prob-weight-alpha", type=float, default=1.0,
                    help="prob_weight sharpness: weight ∝ p**alpha (α=1 raw p; α>1 "
                         "concentrates on the highest-p picks) (_014)")
    ap.add_argument("--vol-window", type=int, default=20,
                    help="trailing trading-day window for realized-vol (sizing-mode "
                         "inverse_vol = risk parity: slice ∝ 1/vol). Default 20.")
    ap.add_argument("--rank-kelly-p", type=float, default=None,
                    help="eval hit-rate for rank_kelly sizing (default: eval R-p@K from the cell)")
    ap.add_argument("--rank-by", default="calibrated", choices=["calibrated", "raw"],
                    help="entry-ranking key: 'calibrated' (default = p_mean, the recalibrated "
                         "p) or 'raw' (the model's p_raw — finer resolution, recovers the "
                         "within-plateau ordering the quantized calibrated p loses; see _020). "
                         "Sizing + breakeven gate stay on the calibrated p either way.")
    ap.add_argument("--min-entry-p", type=float, default=0.0,
                    help="entry-p threshold: only candidates whose selection-bound p (p_mean, "
                         "or p_low with --selection-bound low) is >= this may be entered. "
                         "0.0 = off (champion). NOTE per _021 the calibrated p is quantized "
                         "into wide isotonic plateaus, so this acts as a DAY-LEVEL filter — "
                         "days whose top plateau sits below the threshold trade nothing.")
    ap.add_argument("--min-entry-p-raw", type=float, default=0.0,
                    help="entry threshold on the model's RAW p_raw (pre-calibration, the "
                         "0.04–0.39 scale) — the substantive variant since calibrated p is "
                         "plateaued. Pair with --rank-by raw. 0.0 = off.")
    ap.add_argument("--predictions", default=None,
                    help="default: <cell>/predictions/test.csv")
    args = ap.parse_args()
    K = args.k  # per-day top-K (default 3 = champion); shadows the module default
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

    # rank_kelly sizing needs a per-pick win probability; estimate it leak-free
    # as the cell's EVAL R-precision@K (fraction of top-K picks that are
    # label-positives), NOT the test window.
    rank_kelly_p = args.rank_kelly_p
    if args.sizing_mode == "rank_kelly" and rank_kelly_p is None:
        ev = pd.read_csv(cell / "predictions" / "eval.csv", parse_dates=["date"])
        hits = []
        for _, g in ev.groupby("date"):
            R = int(g["y_true"].sum())
            if R == 0:
                continue
            topk = g.nlargest(K, "p_calibrated")
            hits.append(topk["y_true"].sum() / min(K, R))
        rank_kelly_p = float(pd.Series(hits).mean())
        print(f"[rank_kelly] eval R-p@{K} = {rank_kelly_p:.3f} (used as per-pick win prob)")

    pred_csv = Path(args.predictions) if args.predictions else cell / "predictions" / "test.csv"
    test = pd.read_csv(pred_csv, parse_dates=["date"])
    # Raw-p entry threshold (_026): gate on the model's finest-resolution p_raw
    # (the 0.04–0.39 scale), filtered BEFORE calibration — the substantive variant,
    # since the calibrated p is plateaued (_021). Pair with --rank-by raw to also
    # select the highest-raw survivors. Filters the prediction rows directly.
    if args.min_entry_p_raw > 0.0:
        if "p_raw" not in test.columns:
            raise ValueError("--min-entry-p-raw needs a 'p_raw' column in the predictions CSV")
        b = len(test)
        test = test[test["p_raw"] >= args.min_entry_p_raw].copy()
        print(f"[min-entry-p-raw] >= {args.min_entry_p_raw}: kept {len(test)}/{b} rows, "
              f"{test.date.nunique()} days with >=1 tradable candidate")
    preds = _predictions_dict(test, cal)
    # Entry-p threshold (_026): gate candidates on the same bound the strategy selects on
    # (p_mean default; p_low if selection_bound="low"). Empty days are kept as empty lists
    # (no new entries) so the engine timeline is unchanged.
    n_cand_days = sum(bool(v) for v in preds.values())
    if args.min_entry_p > 0.0:
        bound_idx = 2 if args.selection_bound == "low" else 1  # tuple = (ticker, mean, low, high)
        before = sum(len(v) for v in preds.values())
        preds = {d: [t for t in lst if t[bound_idx] >= args.min_entry_p]
                 for d, lst in preds.items()}
        after = sum(len(v) for v in preds.values())
        n_cand_days = sum(bool(v) for v in preds.values())
        print(f"[min-entry-p] >= {args.min_entry_p}: kept {after}/{before} candidate rows, "
              f"{n_cand_days}/{len(preds)} days with >=1 tradable candidate")
    # rank_by="raw": rank the entry top-K on the model's finest-resolution raw
    # score instead of the quantized calibrated p (which ties wide plateaus and
    # degenerates to the alphabetical tie-break). Sizing/breakeven stay calibrated.
    rank_scores = None
    if args.rank_by == "raw":
        if "p_raw" not in test.columns:
            raise ValueError("--rank-by raw needs a 'p_raw' column in the predictions CSV")
        rank_scores = {
            pd.Timestamp(d): dict(zip(sub.ticker, sub.p_raw))
            for d, sub in test.groupby("date")
        }
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

    # inverse_vol sizing: trailing realized vol per (signal-date, ticker), causal
    # (returns through the signal day, matching the signal-day-close anchor). A
    # wider pre-buffer than the OHLCV load so early signal dates have a full window.
    vol_scores = None
    if args.sizing_mode == "inverse_vol":
        w = args.vol_window
        vhist = _load_closes(tickers, t_start - pd.Timedelta(days=max(120, w * 3)),
                             comparison_end)
        vol_by_tk = {tk: s.sort_index().pct_change().rolling(w).std()
                     for tk, s in vhist.items()}
        vol_scores = {}
        for d in preds:
            dd = pd.Timestamp(d)
            day = {}
            for tk, vs in vol_by_tk.items():
                if dd in vs.index:
                    v = vs.loc[dd]
                    if pd.notna(v) and v > 0:
                        day[tk] = float(v)
            vol_scores[dd] = day
        print(f"[vol] inverse_vol: {w}d realized vol, {len(vol_by_tk)} tickers")

    strat = TopKDailyKellyLabelExit(
        predictions=preds, K=K, target_return=TARGET_RETURN, stop_drawdown=STOP_DD,
        horizon_days=HORIZON, sizer=DiscreteBoundedLossKelly(), sizer_payoffs=(WIN, LOSS),
        breakeven_p=BREAKEVEN_P, fractional_c=args.c, selection_bound=args.selection_bound,
        selection_mode=args.selection_mode, sizing_mode=args.sizing_mode,
        rank_kelly_p=rank_kelly_p, rank_scores=rank_scores, vol_scores=vol_scores,
        prob_weight_alpha=args.prob_weight_alpha,
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
        "config": {"fractional_c": args.c, "selection_bound": args.selection_bound, "K": K,
                   "selection_mode": args.selection_mode, "sizing_mode": args.sizing_mode,
                   "min_entry_p": args.min_entry_p, "min_entry_p_raw": args.min_entry_p_raw,
                   "n_candidate_days": n_cand_days,
                   "rank_kelly_p": rank_kelly_p, "rank_by": args.rank_by,
                   "vol_window": args.vol_window if args.sizing_mode == "inverse_vol" else None,
                   "prob_weight_alpha": args.prob_weight_alpha if args.sizing_mode == "prob_weight" else None},
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

    # Action chart (strategy equity + index buy-hold + labeled buy/sell points).
    # Non-fatal: a plotting failure must not fail an otherwise-complete back-test.
    try:
        plot_actions(out)
    except Exception as e:  # noqa: BLE001
        print(f"[run_backtest_cell] WARN: action chart failed: {type(e).__name__}: {e}")

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
