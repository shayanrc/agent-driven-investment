"""_029: K-sweep (top-2/4/5 vs the champion's top-3) on the top-10-by-AUC cells; check
whether any (cell, K) breaches the total-return leaderboard (backtest_summary.csv).

K is a pure strategy knob applied AFTER prediction, so this needs NO inference / feature
build — it reuses each cell's committed ``predictions/test.csv``. Per cell we fit the VAL
calibrator + load prices ONCE, then sweep K∈{2,3,4,5} reusing them (champion config otherwise:
selection_mode=rank, sizing_mode=equal, c=1.0, selection_bound=mean). K=2→50%/name, 3→33%,
4→25%, 5→20% — a concentration sweep. Test-window backtest (comparison_end = test_end + horizon),
gross of costs, no regime gate.

    uv run python -m scripts.backtests.k_sweep_topauc
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

from backtesting.backtest import Backtest
from backtesting.strategy import run_strategy
from scripts.backtests import benchmarks as bm
from scripts.backtests.calibration_step import fit_calibrator
from scripts.backtests.run_cell5_bayesian_kelly import (
    INITIAL_CASH, LOOKBACK, _build_feeds, _equity_from_history, _load_closes, _predictions_dict)
from scripts.backtests.run_backtest_cell import INDEX_BY_UNIVERSE
from trading_strategies.sizing import DiscreteBoundedLossKelly
from trading_strategies.topk_daily_kelly_label_exit import TopKDailyKellyLabelExit

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results/gbdt/experiments"
KS = [2, 3, 4, 5]
N_TOP = 10


def main() -> None:
    reg = pd.read_csv(ROOT / "results/gbdt/data/r_precision_at_k.csv").sort_values("AUC", ascending=False)
    top = reg.head(N_TOP)["experiment"].tolist()
    board = pd.read_csv(ROOT / "results/backtests/data/backtest_summary.csv")
    cut = float(board.sort_values("total_return_strategy", ascending=False).iloc[9]["total_return_strategy"])
    print(f"board #10 cutoff (total_return_strategy) = {cut:+.4f}\nTop-{N_TOP}-by-AUC: {top}\n", flush=True)

    rows = []
    for cid in top:
        cell = EXP / cid
        auc = round(float(reg[reg.experiment == cid].AUC.iloc[0]), 3)
        base = round(float(reg[reg.experiment == cid].base_rate.iloc[0]), 4)
        if not all((cell / p).exists() for p in ["spec.yaml", "predictions/test.csv", "predictions/val.csv"]):
            print(f"SKIP {cid} (missing artifacts)", flush=True)
            rows.append({"cell": cid, "AUC": auc, "base_rate": base, "note": "missing artifacts"})
            continue
        try:
            spec = yaml.safe_load((cell / "spec.yaml").read_text())["target"]
            WIN = float(spec["threshold_pct"]) / 100; LOSS = float(spec["max_drawdown"]); H = int(spec["horizon_days"])
            BE = LOSS / (LOSS + WIN); idx_t, _ = INDEX_BY_UNIVERSE[spec["universe"]]
            cal, _ = fit_calibrator(cell)
            test = pd.read_csv(cell / "predictions/test.csv", parse_dates=["date"])
            preds = _predictions_dict(test, cal); tickers = sorted(test.ticker.unique())
            t0, t1 = test.date.min(), test.date.max()
            buf = pd.Timedelta(days=int(H * 2.2) + 30)
            closes = _load_closes(tickers, t0 - pd.Timedelta(days=20), t1 + buf)
            closes.update(_load_closes([idx_t], t0 - pd.Timedelta(days=20), t1 + buf))
            ref = max((s.index for s in closes.values()), key=len); data_end = ref.max()
            after = ref[ref > t1]; ce = min(after[H - 1] if len(after) >= H else data_end, data_end)
            roster = {t: s for t, s in closes.items() if t != idx_t}
        except Exception as e:  # noqa: BLE001 — surface + keep going
            print(f"SKIP {cid} (setup: {type(e).__name__}: {e})", flush=True)
            rows.append({"cell": cid, "AUC": auc, "base_rate": base, "note": f"setup fail: {e}"})
            continue
        for K in KS:
            feeds = _build_feeds(roster, t0 - pd.Timedelta(days=20), ce)
            bt = Backtest(feeds, lookback=LOOKBACK, initial_cash=INITIAL_CASH,
                          fill_mode="next_open", gap_policy="ffill_zero_volume")
            strat = TopKDailyKellyLabelExit(predictions=preds, K=K, target_return=WIN, stop_drawdown=LOSS,
                                            horizon_days=H, sizer=DiscreteBoundedLossKelly(), sizer_payoffs=(WIN, LOSS),
                                            breakeven_p=BE, fractional_c=1.0, selection_bound="mean",
                                            selection_mode="rank", sizing_mode="equal")
            eq = _equity_from_history(run_strategy(bt, strat)); eq = eq[eq.index <= ce]
            m = bm.compute_metrics(eq)
            ret = eq.pct_change().dropna()
            sh = float(ret.mean() / ret.std() * math.sqrt(252)) if ret.std() > 0 else float("nan")
            ne = sum(e.kind == "entry" for e in strat.events); tot = float(m["total_return"])
            rows.append({"cell": cid, "AUC": auc, "base_rate": base, "K": K, "total": tot,
                         "maxdd": float(m["max_dd"]), "sharpe": round(sh, 2), "entries": ne,
                         "alloc_pct": round(100 / K, 1), "breach_top10": tot > cut})
            print(f"{cid:48s} K={K} {tot*100:+7.1f}%  DD {m['max_dd']*100:5.1f}%  {ne}ent  breach={tot>cut}", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "results/backtests/data/_029_k_sweep_topauc.csv"
    df.to_csv(out, index=False)
    nb = int(df.get("breach_top10", pd.Series(dtype=bool)).sum())
    print(f"\nwrote {out} | cutoff={cut:+.4f} | breaches={nb}", flush=True)


if __name__ == "__main__":
    main()
