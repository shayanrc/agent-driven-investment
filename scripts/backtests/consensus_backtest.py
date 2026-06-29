"""Cross-model CONSENSUS backtest (_028) — one stock/day, top-5 vote, target/stop exits.

A backend-agnostic consensus over the `_019` forward log's five tracked models. Each trading
day (where all five models have logged) we pool each model's **top-5** picks and elect the
**most-voted** stock; ties break on the **highest summed p_calibrated** across the models that
voted for it. The strategy buys **one** stock per day — that day's winner — if it isn't already
held, sizing each new position at ``max_alloc`` of current equity (capped by available cash, so a
single name never exceeds ``max_alloc`` at entry). Positions are held until the **close** first
reaches ``+target`` (take-profit) or ``-stop`` (stop-loss) — path-honest, CLOSE-based, no time
horizon; anything still open at the data end is marked to market. Entry anchor = signal-day close.

The same trading rule can be driven by a single model's daily **rank-1** pick instead of the
consensus vote (``--selector <model>``) — the controlled comparison that isolates the value of
the vote vs. just following one constituent model.

Read-only over the committed forward log + the `us_equities` close cache. Benchmark = ^SPX
buy-hold over the same window. Initial cash $100k, gross of costs.

Usage:
    uv run python -m scripts.backtests.consensus_backtest --max-alloc 0.25 --target 0.30 --stop 0.15
    uv run python -m scripts.backtests.consensus_backtest --max-alloc 0.25 --target 0.30 --stop 0.15 --selector sp500_50
    uv run python -m scripts.backtests.consensus_backtest --compare       # consensus vs all 5 models, both variants
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results/backtests/data/forward_predictions_log.csv"
DB = ROOT / "data/processed.db"
SPX = "INDEX:^SPX"
INIT = 100_000.0
MODELS = ["sp500_50", "sp500_20", "russell_50_200", "russell_40_100", "nasdaq_40_50"]
VARIANTS = [(0.25, 0.30, 0.15), (0.33, 0.40, 0.20)]


def _window(min_models: int):
    """Return (log_df, sorted trading days with >= min_models present, sym->full-ticker map)."""
    df = pd.read_csv(LOG)
    df["sym"] = df.ticker.str.split(":").str[-1]
    nmod = df[df["rank"] <= 5].groupby("snapshot_date").model.nunique()
    days = sorted(pd.Timestamp(d) for d in nmod[nmod >= min_models].index)
    return df, days, dict(zip(df.sym, df.ticker))


def winner_map(selector: str, min_models: int = 5) -> tuple[dict, list[str]]:
    """{date -> full ticker} for the selector, over the >= min_models window.

    selector == 'consensus' -> daily most-voted top-5 stock (tie: highest summed p),
                               among whatever models are present that day.
    selector in MODELS       -> that model's daily rank-1 pick.
    """
    df, days, sym2tk = _window(min_models)
    dayset = set(days)
    if selector == "consensus":
        t5 = df[(df["rank"] <= 5) & df.snapshot_date.map(lambda d: pd.Timestamp(d) in dayset)]
        wmap = {}
        for d, g in t5.groupby("snapshot_date"):
            a = (g.groupby("sym").agg(votes=("model", "nunique"), psum=("p_calibrated", "sum"))
                 .sort_values(["votes", "psum"], ascending=False))
            wmap[pd.Timestamp(d)] = sym2tk[a.index[0]]
    else:
        g = df[(df.model == selector) & (df["rank"] == 1)
               & df.snapshot_date.map(lambda d: pd.Timestamp(d) in dayset)]
        wmap = {pd.Timestamp(r.snapshot_date): r.ticker for r in g.itertuples()}
    return wmap, sorted(set(wmap.values()))


def load_closes(tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    qs = ",".join("?" * len(tickers))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    px = pd.read_sql(
        f"SELECT date,ticker,close FROM us_equities_data WHERE ticker IN ({qs}) "
        f"AND date>=? AND date<(date(?, '+1 day'))",
        con, params=[*tickers, start, end])
    con.close()
    px["date"] = pd.to_datetime(px.date).dt.normalize()
    return {t: g.set_index("date").close.sort_index() for t, g in px.groupby("ticker")}


def run(selector: str, max_alloc: float, target: float, stop: float, min_models: int = 5) -> dict:
    wmap, tickers = winner_map(selector, min_models)
    start, end = min(wmap), max(wmap)
    closes = load_closes([*tickers, SPX], str(start.date()), str(end.date()))
    cal = sorted(closes[SPX].index)
    cash, pos, eq_curve, trades = INIT, {}, [], []

    def px(tk, d):
        s = closes.get(tk)
        return None if s is None or d not in s.index else float(s.loc[d])

    for d in cal:
        for tk in list(pos):                          # exits at today's close (first-touch)
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
        tk = wmap.get(d)                              # buy the day's winner (1/day, if not held)
        if tk and tk not in pos:
            c = px(tk, d)
            if c:
                equity = cash + sum(pp["shares"] * (px(t, d) or pp["anchor"]) for t, pp in pos.items())
                notional = min(max_alloc * equity, cash)
                if notional > 1.0:
                    pos[tk] = {"shares": notional / c, "anchor": c, "entry": d}
                    cash -= notional
                    trades.append({"date": d, "ticker": tk, "action": "BUY", "ret": None})
        eq_curve.append(cash + sum(pp["shares"] * (px(t, d) or pp["anchor"]) for t, pp in pos.items()))
    eq = pd.Series(eq_curve, index=cal)
    tr = pd.DataFrame(trades)
    bh = closes[SPX].reindex(cal).ffill()
    rets = eq.pct_change().dropna()
    n = len(cal)
    cagr = (eq.iloc[-1] / INIT) ** (252.0 / max(n, 1)) - 1.0
    sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    n_t = int((tr.action == "target").sum()) if len(tr) else 0
    n_s = int((tr.action == "stop").sum()) if len(tr) else 0
    return {
        "selector": selector, "max_alloc": max_alloc, "target": target, "stop": stop,
        "min_models": min_models, "window": f"{start.date()}→{end.date()}", "n_days": n,
        "total_return": float(eq.iloc[-1] / INIT - 1), "cagr": float(cagr),
        "spx_return": float(bh.iloc[-1] / bh.iloc[0] - 1),
        "spx_cagr": float((bh.iloc[-1] / bh.iloc[0]) ** (252.0 / max(n, 1)) - 1.0),
        "sharpe": sharpe, "max_dd": dd, "calmar": float(cagr / abs(dd)) if dd < 0 else float("nan"),
        "n_entries": int((tr.action == "BUY").sum()) if len(tr) else 0,
        "n_target": n_t, "n_stop": n_s,
        "win_rate": float(n_t / (n_t + n_s)) if (n_t + n_s) else float("nan"),
        "n_names": len(tickers), "open_at_end": len(pos), "_eq": eq, "_tr": tr,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector", default="consensus")
    ap.add_argument("--max-alloc", type=float, default=0.25)
    ap.add_argument("--target", type=float, default=0.30)
    ap.add_argument("--stop", type=float, default=0.15)
    ap.add_argument("--compare", action="store_true",
                    help="run consensus vs each constituent model (rank-1), both variants")
    args = ap.parse_args()
    if args.compare:
        for ma, tg, st in VARIANTS:
            print(f"\n=== variant: max_alloc={ma:.0%} target={tg:.0%} stop={st:.0%} "
                  f"(same rule; selection varies) ===")
            print(f"  {'selector':16}{'total':>8}{'excess':>8}{'maxDD':>8}{'entries':>8}"
                  f"{'tgt':>5}{'stop':>5}{'names':>6}")
            for sel in ["consensus", *MODELS]:
                r = run(sel, ma, tg, st)
                print(f"  {sel:16}{r['total_return']*100:>7.1f}%{r['excess'] if False else (r['total_return']-r['spx_return'])*100:>7.1f}%"
                      f"{r['max_dd']*100:>7.1f}%{r['n_entries']:>8}{r['n_target']:>5}{r['n_stop']:>5}{r['n_names']:>6}")
            print(f"  (SPX buy-hold over window: {run('consensus',ma,tg,st)['spx_return']*100:+.1f}%)")
        return
    r = run(args.selector, args.max_alloc, args.target, args.stop)
    print(f"{r['selector']}: total {r['total_return']*100:+.1f}%  SPX {r['spx_return']*100:+.1f}%  "
          f"maxDD {r['max_dd']*100:.1f}%  entries={r['n_entries']} tgt={r['n_target']} stop={r['n_stop']} "
          f"names={r['n_names']} open={r['open_at_end']}")


if __name__ == "__main__":
    main()
