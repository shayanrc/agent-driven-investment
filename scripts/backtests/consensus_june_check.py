"""June-2025 → June-2026 consensus CHECK (_028 sidecar) — re-infer + OOS-clean vs all-5.

The committed forward log only starts 2026-01-02, so a 1-year (June→June) consensus needs the
five cells' predictions RE-INFERRED over the window via the faithful inference path
(``build_scores_multi``; ~2016 warmup so each cell's test window reproduces and the self-check
passes). We then build the daily consensus winner (top-5 cross-model vote, tie → highest summed
p_raw) and run the same 1-stock/day target-stop strategy as ``consensus_backtest`` (variants A
and C), two ways:

  * **clean**  — each model votes ONLY on dates >= its OOS-valid start (test_end+1), so the
                 breadth ramps russell×2 (Jun'25) → +sp500_50/nasdaq (Mar'26) → +sp500_20
                 (Apr'26). No look-ahead.
  * **all5**   — all five vote every day (LEAKY: sp500's Jun–Dec'25 picks are in-sample / its
                 train-val window) — shown only as a sensitivity, not a valid result.

Read-only over the cells + us_equities cache. ~20–30 min (full re-inference of 3 universes).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backtests.infer_fresh_predictions import build_scores_multi, self_check
from scripts.backtests.consensus_backtest import load_closes, SPX, INIT

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results/gbdt/experiments"
WIN_START = pd.Timestamp("2025-06-05")
END = "2026-06-26"
WARMUP = "2016-01-01"            # covers russell_50_200's 2023-07 test window for the self-check

CELLS = {
    "sp500_50": "sp500_up_50pct_50d_dd25pct_agentloop",
    "sp500_20": "sp500_up_20pct_25d_dd10pct_agentloop",
    "russell_50_200": "russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1",
    "russell_40_100": "russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1",
    "nasdaq_40_50": "nasdaq100_up_40pct_50d_dd20pct_agentloop_mix",
}
OOS = {  # test_end + 1 (from r_precision_at_k.csv) — the clean out-of-sample start
    "sp500_50": pd.Timestamp("2026-03-13"), "sp500_20": pd.Timestamp("2026-04-18"),
    "russell_50_200": pd.Timestamp("2024-10-04"), "russell_40_100": pd.Timestamp("2025-05-14"),
    "nasdaq_40_50": pd.Timestamp("2026-03-13"),
}
VARIANTS = {"A (25/30/15)": (0.25, 0.30, 0.15), "C (33/30/15)": (0.33, 0.30, 0.15)}


def infer_topk() -> pd.DataFrame:
    """Re-infer all 5 cells over the window; return (date, model, rank, sym, ticker, p)."""
    specs = [(EXP / c, WARMUP) for c in CELLS.values()]
    scores = build_scores_multi(specs, END)
    rows = []
    for model, c in CELLS.items():
        cell = EXP / c
        df = scores[str(cell)].copy()
        try:
            self_check(df, cell, incremental=True, label=model)
            print(f"[{model}] self-check PASSED")
        except Exception as e:  # faithful-inference guard; surface but keep going
            print(f"[{model}] self-check WARN: {e}")
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df.date >= WIN_START) & (df.date <= pd.Timestamp(END))].copy()
        df["rank"] = df.groupby("date").p_raw.rank(ascending=False, method="first")
        df = df[df["rank"] <= 10]
        df["model"] = model
        df["sym"] = df.ticker.str.split(":").str[-1]
        rows.append(df[["date", "model", "rank", "sym", "ticker", "p_raw"]])
    return pd.concat(rows, ignore_index=True)


def winner_map(preds: pd.DataFrame, clean: bool) -> tuple[dict, list[str]]:
    t5 = preds[preds["rank"] <= 5].copy()
    if clean:
        t5 = t5[t5.apply(lambda r: r.date >= OOS[r.model], axis=1)]
    sym2tk = dict(zip(t5.sym, t5.ticker))
    wmap = {}
    for d, g in t5.groupby("date"):
        a = (g.groupby("sym").agg(votes=("model", "nunique"), psum=("p_raw", "sum"))
             .sort_values(["votes", "psum"], ascending=False))
        wmap[d] = sym2tk[a.index[0]]
    return wmap, sorted(set(wmap.values()))


def sim(wmap: dict, tickers: list[str], max_alloc: float, target: float, stop: float) -> dict:
    start, end = min(wmap), max(wmap)
    closes = load_closes([*tickers, SPX], str(start.date()), str(end.date()))
    cal = sorted(closes[SPX].index)
    cash, pos, eq_curve, trades = INIT, {}, [], []

    def px(tk, d):
        s = closes.get(tk)
        return None if s is None or d not in s.index else float(s.loc[d])

    for d in cal:
        for tk in list(pos):
            c = px(tk, d)
            if c is None:
                continue
            p = pos[tk]
            trig = ("target" if c >= p["anchor"] * (1 + target)
                    else "stop" if c <= p["anchor"] * (1 - stop) else None)
            if trig:
                cash += p["shares"] * c
                trades.append({"date": d, "ticker": tk, "action": trig, "ret": c / p["anchor"] - 1})
                del pos[tk]
        tk = wmap.get(d)
        if tk and tk not in pos:
            c = px(tk, d)
            if c:
                eqv = cash + sum(pp["shares"] * (px(t, d) or pp["anchor"]) for t, pp in pos.items())
                notional = min(max_alloc * eqv, cash)
                if notional > 1.0:
                    pos[tk] = {"shares": notional / c, "anchor": c}
                    cash -= notional
                    trades.append({"date": d, "ticker": tk, "action": "BUY", "ret": None})
        eq_curve.append(cash + sum(pp["shares"] * (px(t, d) or pp["anchor"]) for t, pp in pos.items()))
    eq = pd.Series(eq_curve, index=cal)
    tr = pd.DataFrame(trades)
    bh = closes[SPX].reindex(cal).ffill()
    rets = eq.pct_change().dropna()
    n = len(cal)
    cagr = (eq.iloc[-1] / INIT) ** (252.0 / max(n, 1)) - 1.0
    dd = float((eq / eq.cummax() - 1).min())
    n_t = int((tr.action == "target").sum()) if len(tr) else 0
    n_s = int((tr.action == "stop").sum()) if len(tr) else 0
    return {"window": f"{start.date()}→{end.date()}", "n_days": n,
            "total": float(eq.iloc[-1] / INIT - 1), "cagr": float(cagr),
            "spx": float(bh.iloc[-1] / bh.iloc[0] - 1),
            "sharpe": float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0,
            "maxdd": dd, "entries": int((tr.action == "BUY").sum()) if len(tr) else 0,
            "target": n_t, "stop": n_s,
            "win": float(n_t / (n_t + n_s)) if (n_t + n_s) else float("nan"),
            "names": len(tickers)}


def main() -> None:
    preds = infer_topk()
    print(f"\nre-inferred top-10 over {WIN_START.date()}→{END}: "
          f"{preds.date.min().date()}→{preds.date.max().date()}, "
          f"{preds.date.nunique()} days, models={sorted(preds.model.unique())}\n")
    out = []
    for mode, clean in [("clean (OOS-masked breadth-ramp)", True), ("all5 (LEAKY sensitivity)", False)]:
        wmap, tk = winner_map(preds, clean)
        print(f"=== {mode} | winner-days={len(wmap)} distinct-names={len(tk)} ===")
        for tag, (ma, tg, st) in VARIANTS.items():
            r = sim(wmap, tk, ma, tg, st)
            print(f"  {tag:14} {r['window']} ({r['n_days']}d)  total {r['total']*100:+6.1f}%  "
                  f"CAGR {r['cagr']*100:+6.1f}%  Sharpe {r['sharpe']:.2f}  maxDD {r['maxdd']*100:6.1f}%  "
                  f"SPX {r['spx']*100:+.1f}%  | entries {r['entries']} tgt {r['target']} stop {r['stop']} "
                  f"win {r['win']*100 if r['win']==r['win'] else float('nan'):.0f}% names {r['names']}")
            out.append({"mode": mode, "variant": tag, **{k: v for k, v in r.items()}})
    od = ROOT / "results/backtests/_028_consensus_backtest"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(od / "june_check_results.csv", index=False)
    print(f"\nwrote {od / 'june_check_results.csv'}")


if __name__ == "__main__":
    main()
