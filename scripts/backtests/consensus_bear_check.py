"""2022-bear OOS consensus test (_028 sidecar) — genuinely out-of-sample bear regime.

The only bear regime outside ALL models' training data is **2022**, reachable only via the two
**bear2022 retrains** (`sp500_up_50pct_50d_dd25pct_bear2022`, `sp500_up_20pct_25d_dd10pct_bear2022`)
whose train/val/eval segments end before 2021-12-21 — so the 2022 bear (their test window) is true
out-of-sample. This is therefore necessarily a **2-model, sp500-only** consensus (the russell/nasdaq
candidates have no pre-2022 retrain), i.e. a downside STRESS TEST of the strategy, not the full
5-way cross-universe vote.

Reads each bear cell's `predictions/test.csv` directly (no re-inference). Consensus winner = the
most-voted top-5 stock across the 2 models (tie → highest summed p_raw); the strategy + sizing are
identical to `consensus_backtest` (1 stock/day, max_alloc cap, +target/-stop close-based exits,
hold-to-barrier). Benchmark = ^SPX buy-hold over the same bear window. $100k, gross.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.backtests.consensus_backtest import load_closes, SPX, INIT

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "results/gbdt/experiments"
BEAR = {"sp500_50_bear": "sp500_up_50pct_50d_dd25pct_bear2022",
        "sp500_20_bear": "sp500_up_20pct_25d_dd10pct_bear2022"}
VARIANTS = {"A (25/30/15)": (0.25, 0.30, 0.15), "B (33/40/20)": (0.33, 0.40, 0.20),
            "C (33/30/15)": (0.33, 0.30, 0.15), "D (25/40/20)": (0.25, 0.40, 0.20)}


def load_preds() -> pd.DataFrame:
    frames = []
    for m, c in BEAR.items():
        df = pd.read_csv(EXP / c / "predictions" / "test.csv", usecols=["date", "ticker", "p_raw"])
        df["date"] = pd.to_datetime(df.date)
        df["model"] = m
        df["sym"] = df.ticker.str.split(":").str[-1]
        df["rank"] = df.groupby("date").p_raw.rank(ascending=False, method="first")
        frames.append(df[df["rank"] <= 10])
    p = pd.concat(frames, ignore_index=True)
    n = p.groupby("date").model.nunique()
    both = set(n[n == 2].index)  # overlap window where both bear models have predictions
    return p[p.date.isin(both)].copy()


def winner_map(preds: pd.DataFrame, selector: str) -> tuple[dict, list[str]]:
    if selector == "consensus":
        t5 = preds[preds["rank"] <= 5]
        sym2tk = dict(zip(t5.sym, t5.ticker))
        wmap = {}
        for d, g in t5.groupby("date"):
            a = (g.groupby("sym").agg(votes=("model", "nunique"), psum=("p_raw", "sum"))
                 .sort_values(["votes", "psum"], ascending=False))
            wmap[d] = sym2tk[a.index[0]]
    else:
        g = preds[(preds.model == selector) & (preds["rank"] == 1)]
        wmap = {r.date: r.ticker for r in g.itertuples()}
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
    dd = float((eq / eq.cummax() - 1).min())
    n_t = int((tr.action == "target").sum()) if len(tr) else 0
    n_s = int((tr.action == "stop").sum()) if len(tr) else 0
    return {"window": f"{start.date()}→{end.date()}", "n_days": n,
            "total": float(eq.iloc[-1] / INIT - 1),
            "spx": float(bh.iloc[-1] / bh.iloc[0] - 1),
            "sharpe": float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0,
            "maxdd": dd, "entries": int((tr.action == "BUY").sum()) if len(tr) else 0,
            "target": n_t, "stop": n_s,
            "win": float(n_t / (n_t + n_s)) if (n_t + n_s) else float("nan"),
            "names": len(tickers), "_eq": eq}


def main() -> None:
    preds = load_preds()
    print(f"2022-bear OOS window (both bear models present): {preds.date.min().date()}→"
          f"{preds.date.max().date()}, {preds.date.nunique()} trading days\n")
    out, eqs = [], {}
    for sel in ["consensus", "sp500_50_bear", "sp500_20_bear"]:
        wmap, tk = winner_map(preds, sel)
        print(f"=== selector: {sel} (distinct names {len(tk)}) ===")
        for tag, (ma, tg, st) in VARIANTS.items():
            r = sim(wmap, tk, ma, tg, st)
            print(f"  {tag:14} {r['window']} ({r['n_days']}d)  total {r['total']*100:+6.1f}%  "
                  f"SPX {r['spx']*100:+.1f}%  maxDD {r['maxdd']*100:6.1f}%  Sharpe {r['sharpe']:5.2f}  | "
                  f"entries {r['entries']} tgt {r['target']} stop {r['stop']} "
                  f"win {r['win']*100 if r['win']==r['win'] else float('nan'):.0f}%")
            out.append({"selector": sel, "variant": tag,
                        **{k: v for k, v in r.items() if not k.startswith("_")}})
            if sel == "consensus":
                eqs[tag] = r["_eq"]
    od = ROOT / "results/backtests/_028_consensus_backtest"
    od.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(od / "bear_results.csv", index=False)
    # equity figure: consensus A & C vs SPX over the 2022 bear
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cal = list(eqs["A (25/30/15)"].index)
        spx = load_closes([SPX], str(cal[0].date()), str(cal[-1].date()))[SPX].reindex(cal).ffill()
        spx = spx / spx.iloc[0] * INIT
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(eqs["A (25/30/15)"].index, eqs["A (25/30/15)"].values, lw=2, color="#1f77b4",
                label=f"A 25/+30/-15 → {(eqs['A (25/30/15)'].iloc[-1]/INIT-1)*100:+.1f}%")
        ax.plot(eqs["C (33/30/15)"].index, eqs["C (33/30/15)"].values, lw=2, color="#2ca02c",
                label=f"C 33/+30/-15 → {(eqs['C (33/30/15)'].iloc[-1]/INIT-1)*100:+.1f}%")
        ax.plot(spx.index, spx.values, lw=1.5, color="gray", ls="--",
                label=f"SPX → {(spx.iloc[-1]/INIT-1)*100:+.1f}%")
        ax.axhline(INIT, color="black", lw=0.6)
        ax.set_title(f"_028 consensus — 2022 BEAR OOS test (genuinely out-of-sample)\n"
                     f"2-model sp500 bear2022 retrains · {cal[0].date()}→{cal[-1].date()} "
                     f"({len(cal)}d) · $100K · gross · benchmark ^SPX", fontsize=10, fontweight="bold")
        ax.set_ylabel("equity ($)"); ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(od / "figs" / "_028_consensus_bear2022.png", dpi=140)
        print(f"\nwrote bear_results.csv + figs/_028_consensus_bear2022.png")
    except Exception as e:
        print(f"\nwrote bear_results.csv (figure skipped: {e})")


if __name__ == "__main__":
    main()
