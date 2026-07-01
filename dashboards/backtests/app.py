"""Streamlit dashboard for the ``/daily-predictions`` forward log (the ``_019`` node).

Run with::

    streamlit run dashboards/backtests/app.py

A **read-only viewer** over the committed ``results/backtests/data/forward_predictions_log.csv``
— the durable forward-OOS signal record. Two tabs:

  * **Predictions** — one day's read: per-model big-card panels (ticker colour-coded by lift,
    company name, target/stop levels) and the cross-model **consensus** panel (most-voted,
    ≥3/5 majority), with collapsible detail tables. The SMA200 regime gate lives in the sidebar.
  * **Backtests** — replay the logged picks on the price cache, independent of the backtest
    engine: pick a strategy (consensus or a model) + start date → equity vs index buy-hold with
    buy/sell trade markers, summary stats, and a trades table.

No inference — it only reads the log (refreshed by the ``/daily-predictions`` cadence). All the
usual caveats apply: raw ``p`` (calibrated probability, not certainty), bull-only edge (``_028``),
small effective-N, a forward test — not investment advice.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results/backtests/data/forward_predictions_log.csv"
DB = ROOT / "data/processed.db"  # us_equities cache (read-only) — prices + the us_equities_names table

# Display order (deployed champions first, then comparison candidates) + descriptive labels.
MODEL_ORDER = ["sp500_50", "sp500_20", "russell_50_200", "russell_40_100", "nasdaq_40_50"]
MODEL_LABEL = {
    "sp500_50": "sp500 +50% / 50d", "sp500_20": "sp500 +20% / 25d",
    "russell_50_200": "russell1000 +50% / 200d", "russell_40_100": "russell1000 +40% / 100d",
    "nasdaq_40_50": "nasdaq100 +40% / 50d",
}
MAJORITY = 3  # ≥3 of 5 models = panel majority (≥50%)
INIT_CASH = 100_000.0
_INDEX_TK = {"nasdaq_40_50": "INDEX:^NDX"}  # buy-hold benchmark index per strategy (default ^SPX)


@st.cache_data(show_spinner=False)
def _load(mtime: float) -> pd.DataFrame:  # mtime arg busts the cache when the log changes
    d = pd.read_csv(LOG)
    d["date"] = pd.to_datetime(d["snapshot_date"]).dt.date
    d["sym"] = d["ticker"].str.split(":").str[-1]
    return d


def load_log() -> pd.DataFrame:
    return _load(LOG.stat().st_mtime)


@st.cache_data(show_spinner=False)
def _names(_mtime: float) -> dict:
    """ticker → company name from the us_equities_names DB table (empty until
    fetch_ticker_names.py has populated it); read-only, cache-only, no network."""
    if not DB.exists():
        return {}
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT ticker, name FROM us_equities_names").fetchall()
    except sqlite3.OperationalError:  # table not created yet
        return {}
    finally:
        con.close()
    return {t: n for t, n in rows}


def consensus(day: pd.DataFrame, pool_k: int) -> pd.DataFrame:
    """Cross-model vote over each model's top-``pool_k``: most-voted wins, tie → highest Σp."""
    t = day[day["rank"] <= pool_k]
    if t.empty:
        return pd.DataFrame()
    g = (t.groupby("sym")
         .agg(models=("model", "nunique"), psum=("p_calibrated", "sum"),
              voters=("model", lambda s: ", ".join(sorted(s.unique()))))
         .reset_index()
         .sort_values(["models", "psum"], ascending=False)
         .reset_index(drop=True))
    return g


_DD_RE = re.compile(r"dd(\d+)pct")


def _gain_dd(r) -> tuple[float, float | None]:
    """(+gain fraction, −max-drawdown fraction) — gain from the log's threshold_pct,
    max drawdown parsed from the cell's ``ddNpct`` token."""
    m = _DD_RE.search(str(r.cell))
    return float(r.threshold_pct) / 100.0, (int(m.group(1)) / 100.0 if m else None)


def _target_desc(r) -> str:
    """What the probability predicts, e.g. '20% gain in 25d, max drawdown 10%'."""
    g, dd = _gain_dd(r)
    return f"{int(g * 100)}% gain in {int(r.horizon_days)}d, max drawdown {int(dd * 100) if dd is not None else '?'}%"


@st.cache_data(show_spinner=False)
def _closes_on(date_str: str, tickers: tuple[str, ...], _mtime: float) -> dict:
    """Close per ticker on ``date_str`` (half-open day interval, per the cache's time-suffixed
    ``date``), read-only from the us_equities cache. Cache-only — never hits the network."""
    if not tickers or not DB.exists():
        return {}
    nxt = str((pd.Timestamp(date_str) + pd.Timedelta(days=1)).date())
    ph = ",".join("?" * len(tickers))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        rows = con.execute(
            f"SELECT ticker, close FROM us_equities_data "
            f"WHERE ticker IN ({ph}) AND date >= ? AND date < ?",
            (*tickers, date_str, nxt)).fetchall()
    finally:
        con.close()
    return {t: c for t, c in rows}


def _picks(day: pd.DataFrame, models: list[str], k: int, closes: dict) -> pd.DataFrame:
    rows = []
    for m in models:
        for _, r in day[(day.model == m) & (day["rank"] <= k)].sort_values("rank").iterrows():
            g, dd = _gain_dd(r)
            entry = closes.get(r.ticker)
            rows.append({
                "live": "✓" if bool(r.deployed) else "",
                "Ticker": r.sym, "Model / Rank": f"{m} / {int(r['rank'])}",
                "probability": round(float(r.p_calibrated), 3), "predicting": _target_desc(r),
                "target": round(entry * (1 + g), 2) if entry else None,
                "stoploss": round(entry * (1 - dd), 2) if (entry and dd is not None) else None,
            })
    return pd.DataFrame(rows)


def _lift_color(lift: float) -> str:
    """Heat colour by lift (p ÷ base rate) — 🟢 ≥4× / 🟠 2–4× / ⚪ <2×. Skill-comparable across
    models, unlike absolute p (which base-rate differences make incomparable)."""
    return "#16a34a" if lift >= 4 else "#e08e0b" if lift >= 2 else "#6b7280"


def _card_html(sym: str, name: str, rank: int, p: float, lift: float, levels: str) -> str:
    """A big ticker card: symbol large + colour-coded by lift; company name, rank/prob, levels below."""
    nm = (name[:22] + "…") if len(name) > 23 else name
    return (f'<div style="text-align:center;padding:6px 0">'
            f'<div style="font-size:34px;font-weight:800;color:{_lift_color(lift)};line-height:1.1">{sym}</div>'
            f'<div style="font-size:11px;color:#9aa0a6;line-height:1.15;min-height:15px">{nm}</div>'
            f'<div style="font-size:13px;color:#9aa0a6;margin-top:2px">#{rank} · {p:.1%}</div>'
            f'<div style="font-size:14px;margin-top:3px">{levels}</div></div>')


def _render_panels(day: pd.DataFrame, models: list[str], k: int, closes: dict, names: dict) -> None:
    """One bordered panel per model: header (label + deployed/candidate + target event) and a row
    of big cards — the ticker large + colour-coded by lift, with target / stoploss levels below."""
    for m in models:
        g = day[(day.model == m) & (day["rank"] <= k)].sort_values("rank")
        if g.empty:
            continue
        r0 = g.iloc[0]
        with st.container(border=True):
            tag = "✓ deployed (live)" if bool(r0.deployed) else "candidate"
            st.markdown(f"##### {MODEL_LABEL.get(m, m)}  ·  {tag}")
            st.caption(f"predicting **{_target_desc(r0)}**")
            for col, (_, r) in zip(st.columns(len(g)), g.iterrows()):
                gg, dd = _gain_dd(r)
                e = closes.get(r.ticker)
                t = f"{e * (1 + gg):.2f}" if e else "—"
                s = f"{e * (1 - dd):.2f}" if (e and dd is not None) else "—"
                levels = (f'<span style="color:#16a34a">target</span> {t}'
                          f'&nbsp;&nbsp;&nbsp;<span style="color:#dc2626">stop</span> {s}')
                lift = float(r.p_calibrated) / float(r.base_rate) if r.base_rate else float("nan")
                col.markdown(_card_html(r.sym, names.get(r.ticker, ""), int(r["rank"]),
                                        float(r.p_calibrated), lift, levels), unsafe_allow_html=True)


def _votes_color(n: int) -> str:
    """Colour a consensus card by breadth of agreement — 🟢 majority / 🟠 2 / ⚪ 1 vote."""
    return "#16a34a" if n >= MAJORITY else "#e08e0b" if n >= 2 else "#6b7280"


def _render_consensus_panel(c: pd.DataFrame, names: dict) -> None:
    """Bordered panel of big cards for the consensus winner(s) — the names clearing the
    ≥MAJORITY/5 panel majority (or the single top plurality name if none do)."""
    sym_name = {t.split(":")[-1]: n for t, n in names.items()}
    maj = c[c.models >= MAJORITY]
    winners = maj if not maj.empty else c.head(1)
    with st.container(border=True):
        head = "🗳️ Consensus winner" + ("s" if len(winners) > 1 else "")
        note = "" if not maj.empty else f" · plurality (no ≥{MAJORITY}/5 majority)"
        st.markdown(f"##### {head}{note}")
        for col, (_, r) in zip(st.columns(max(len(winners), 1)), winners.iterrows()):
            nm = sym_name.get(r.sym, "")
            nm = (nm[:22] + "…") if len(nm) > 23 else nm
            col.markdown(
                f'<div style="text-align:center;padding:6px 0">'
                f'<div style="font-size:34px;font-weight:800;color:{_votes_color(int(r.models))};'
                f'line-height:1.1">{r.sym}</div>'
                f'<div style="font-size:11px;color:#9aa0a6;line-height:1.15;min-height:15px">{nm}</div>'
                f'<div style="font-size:13px;color:#9aa0a6;margin-top:2px">{int(r.models)}/5 votes · Σp {r.psum:.2f}</div>'
                f'<div style="font-size:12px;color:#9aa0a6;margin-top:3px">{r.voters}</div></div>',
                unsafe_allow_html=True)


def _render_regime(day: pd.DataFrame) -> None:
    """Per-universe SMA200 regime gate — stacked metrics for the sidebar."""
    st.markdown("**regime gate**")
    gates = (day.groupby("gate_index")
             .agg(on=("regime_on", "first"), close=("gate_close", "first"), sma=("gate_sma200", "first")))
    for idx, g in gates.iterrows():
        pct = (g.close / g.sma - 1) if g.sma else float("nan")
        st.metric(idx.split(":")[-1].lstrip("^"), "🟢 ON" if g.on else "🔴 OFF",
                  f"{pct:+.1%} vs SMA200", delta_color="normal" if g.on else "inverse")
    st.caption("ON ⇒ deploy · OFF ⇒ cash (`_016`–`_018`)")


def render_snapshot(df: pd.DataFrame, date, k: int, pool_k: int) -> None:
    day = df[df.date == date]
    st.subheader(f"📅 {date}")

    # Per-pick close on the snapshot date (cache-only) → target / stoploss levels.
    shown = tuple(sorted(day[day["rank"] <= k].ticker.unique()))
    closes = _closes_on(str(date), shown, DB.stat().st_mtime if DB.exists() else 0.0)
    models = [m for m in MODEL_ORDER if m in set(day.model)]  # deployed champions first, then candidates
    names = _names(DB.stat().st_mtime if DB.exists() else 0.0)

    st.markdown("### Predictions by model")
    st.caption("card colour = lift (p ÷ base rate) — 🟢 ≥4× · 🟠 2–4× · ⚪ <2×  "
               "(skill-comparable across models; the ×-figure under each ticker is the lift)")
    _render_panels(day, models, k, closes, names)

    with st.expander("📋 All picks — table", expanded=False):
        st.dataframe(_picks(day, models, k, closes), hide_index=True, width="stretch")
        st.caption("✓ = deployed (live) · target = close × (1 + gain%) · stoploss = close × (1 − max-DD%)")

    st.markdown("### 🗳️ Cross-model consensus")
    c = consensus(day, pool_k)
    if c.empty:
        st.info("no picks for this day")
        return
    _render_consensus_panel(c, names)
    disp = c.copy()
    disp["majority"] = disp.models.map(lambda n: "✓" if n >= MAJORITY else "")
    disp = disp.rename(columns={"sym": "Stock", "models": "# models", "psum": "Σp", "voters": "voting models"})
    disp["Σp"] = disp["Σp"].round(3)
    with st.expander("🗳️ vote detail — table", expanded=False):
        st.dataframe(disp[["Stock", "# models", "Σp", "majority", "voting models"]],
                     hide_index=True, width="stretch")
    st.caption(f"pool = each model's top-{pool_k}; tie → highest Σp; ✓ = ≥{MAJORITY}/5 panel majority. "
               "Bull-only amplifier (`_028`) — 1 stock/day, not promoted.")


def _bt_winners(df: pd.DataFrame, selector: str, start, regime_only: bool) -> dict:
    """{Timestamp → full ticker} the strategy buys: 'consensus' (most-voted top-5, tie → Σp) or a
    model's rank-1. Restricted to snapshot dates ≥ start; regime-ON days only if regime_only."""
    d = df[df.date >= start]
    if regime_only:
        d = d[d.regime_on]
    if selector == "consensus":
        wmap = {}
        for day, g in d[d["rank"] <= 5].groupby("date"):
            a = (g.groupby("ticker").agg(votes=("model", "nunique"), psum=("p_calibrated", "sum"))
                 .sort_values(["votes", "psum"], ascending=False))
            wmap[pd.Timestamp(day)] = a.index[0]
        return wmap
    g = d[(d.model == selector) & (d["rank"] == 1)]
    return {pd.Timestamp(r.date): r.ticker for r in g.itertuples()}


@st.cache_data(show_spinner=False)
def _bt_prices(tickers: tuple, start_iso: str, end_iso: str, _mtime: float) -> dict:
    """{ticker → DataFrame[open, close]} from us_equities_data — read-only, cache-only, normalized."""
    if not tickers or not DB.exists():
        return {}
    qs = ",".join("?" * len(tickers))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        px = pd.read_sql(
            f"SELECT date, ticker, open, close FROM us_equities_data WHERE ticker IN ({qs}) "
            "AND date >= ? AND date < (date(?, '+1 day'))", con, params=[*tickers, start_iso, end_iso])
    finally:
        con.close()
    px["date"] = pd.to_datetime(px.date).dt.normalize()
    return {t: g.set_index("date")[["open", "close"]].sort_index() for t, g in px.groupby("ticker")}


def _simulate(winners: dict, prices: dict, index_tk: str, target: float, stop: float,
              max_alloc: float, horizon: int, start_ts):
    """Replay the picks — signals are EOD, so an EOD signal on day t is filled at the NEXT trading
    day's OPEN (no same-day look-ahead); positions exit at the first of +target / −stop / horizon
    on the CLOSE (first-touch). 1 stock/day, equal max-alloc sizing, $100k init, gross of costs.
    The window (and the benchmark's normalization anchor) starts at ``start_ts`` — the same for every
    strategy so the index buy-hold is identical across SPX strategies; the strategy sits in cash until
    its first signal fires. Returns (equity, trades, benchmark) or None if prices are missing."""
    if not winners or index_tk not in prices:
        return None
    cal = [d for d in prices[index_tk].index if d >= start_ts]
    cash, pos, eq, trades, pending = INIT_CASH, {}, [], [], None

    def px(tk, d, col):
        s = prices.get(tk)
        if s is None or d not in s.index:
            return None
        v = float(s.loc[d, col])
        return v if v == v else None  # NaN → None

    for d in cal:
        if pending and pending not in pos:                      # yesterday's EOD signal → fill at today's OPEN
            o = px(pending, d, "open")
            if o:
                equity = cash + sum(pp["shares"] * (px(t, d, "close") or pp["anchor"]) for t, pp in pos.items())
                notional = min(max_alloc * equity, cash)
                if notional > 1.0:
                    pos[pending] = {"shares": notional / o, "anchor": o, "entry": d}
                    cash -= notional
                    trades.append({"date": d, "ticker": pending, "action": "BUY", "price": o, "ret": None})
        pending = None
        for tk in list(pos):                                    # exits at today's CLOSE (first-touch)
            c = px(tk, d, "close")
            if c is None:
                continue
            p = pos[tk]
            trig = ("target" if c >= p["anchor"] * (1 + target)
                    else "stop" if c <= p["anchor"] * (1 - stop)
                    else "horizon" if horizon and (d - p["entry"]).days >= horizon else None)
            if trig:
                cash += p["shares"] * c
                trades.append({"date": d, "ticker": tk, "action": trig, "price": c,
                               "ret": c / p["anchor"] - 1})
                del pos[tk]
        if d in winners:                                        # today's EOD signal → fill next OPEN
            pending = winners[d]
        eq.append(cash + sum(pp["shares"] * (px(t, d, "close") or pp["anchor"]) for t, pp in pos.items()))
    equity = pd.Series(eq, index=cal)
    bench = prices[index_tk]["close"].reindex(cal).ffill()
    bench = bench / bench.iloc[0] * INIT_CASH
    return equity, pd.DataFrame(trades), bench


_TRIG = {"target": "t", "stop": "s", "horizon": "h"}  # exit-label suffix, matching plot_actions.py


def _plot_actions(equity, trades, bench, strat_label: str, bench_label: str):
    """The `plot_actions.py` `actions.png` style — strategy equity (dark blue) + index buy-hold
    (grey dashed) + init-cash line, every trade marked (▲ buy / ▼ sell) and labelled with its
    ticker (entries stack upward, exits stack downward; exit suffix ·t target / ·s stop / ·h horizon)."""
    tot = equity.iloc[-1] / INIT_CASH - 1
    dd = float((equity / equity.cummax() - 1).min())
    bmk = bench.iloc[-1] / bench.iloc[0] - 1
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(equity.index, equity.values, lw=2, color="#1f4e79",
            label=f"{strat_label}  {tot * 100:+.0f}%  (DD {dd * 100:.0f}%)")
    ax.plot(bench.index, bench.values, lw=1.3, color="#888", ls="--",
            label=f"{bench_label} buy-hold  {bmk * 100:+.0f}%")
    ax.axhline(INIT_CASH, color="gray", lw=0.6, ls=":")
    if len(trades):
        pk = trades.copy()
        eqd = equity.reindex(equity.index.union(pk["date"].unique())).ffill().bfill()
        pk["y"] = pk["date"].map(eqd)
        ent, ex = pk[pk.action == "BUY"], pk[pk.action != "BUY"]
        ax.scatter(ent["date"], ent["y"], marker="^", s=46, color="#1a9850", zorder=5, edgecolor="white", lw=0.5)
        ax.scatter(ex["date"], ex["y"], marker="v", s=46, color="#d73027", zorder=5, edgecolor="white", lw=0.5)
        fs = 6 if len(pk) > 40 else 7.5
        step = fs + 2.5
        for is_entry, d_, sgn, col in [(True, ent, 1, "#1a7a3a"), (False, ex, -1, "#b2182b")]:
            for _, grp in d_.groupby("date"):
                for i, (_, r) in enumerate(grp.iterrows()):
                    tk = str(r["ticker"]).split(":")[-1]
                    lab = tk if is_entry else f"{tk}·{_TRIG.get(r['action'], '')}"
                    ax.annotate(lab, (r["date"], r["y"]),
                                xytext=(0, sgn * (9 + i * step)), textcoords="offset points",
                                ha="center", va="bottom" if sgn > 0 else "top", fontsize=fs, color=col)
    ax.set_ylabel("portfolio value ($)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{strat_label}  ·  from {equity.index[0].date()}\n"
                 "▲ buy   ▼ sell  (·t target  ·s stop  ·h horizon)", fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def render_backtests(df: pd.DataFrame) -> None:
    dates = sorted(df.date.unique())
    st.markdown("### Backtest — returns of the predicted trades")
    st.caption("Independent of the backtest engine — replays the logged picks on the price cache: "
               "1 stock/day (consensus winner, or the model's rank-1); signals are EOD so each is "
               "**filled at the next trading day's open**, then exits at the first of +target / −stop / "
               "horizon (close-based); equal max-alloc sizing, gross of costs.")

    c1, c2, c3 = st.columns([2, 2, 1])
    strat = c1.selectbox("strategy", ["consensus", *MODEL_ORDER],
                         format_func=lambda s: "consensus (top-5 vote)" if s == "consensus"
                         else MODEL_LABEL.get(s, s))
    start = c2.date_input("start date", value=dates[0], min_value=dates[0], max_value=dates[-1])
    regime_only = c3.checkbox("regime-ON only", value=True)
    e1, e2, e3, e4 = st.columns(4)
    target = e1.slider("target %", 5, 100, 30) / 100
    stop = e2.slider("stop %", 5, 50, 15) / 100
    max_alloc = e3.slider("max alloc %", 5, 100, 25) / 100
    horizon = e4.slider("horizon (days, 0 = off)", 0, 250, 60)

    winners = _bt_winners(df, strat, start, regime_only)
    if not winners:
        st.info("no trades for this strategy / window.")
        return
    index_tk = _INDEX_TK.get(strat, "INDEX:^SPX")
    tickers = tuple(sorted(set(winners.values()) | {index_tk}))
    prices = _bt_prices(tickers, str(start), str(max(dates)), DB.stat().st_mtime if DB.exists() else 0.0)
    res = _simulate(winners, prices, index_tk, target, stop, max_alloc, horizon, pd.Timestamp(start))
    if res is None:
        st.warning("price data unavailable for this window (is the cache seeded?).")
        return
    equity, trades, bench = res

    tot = equity.iloc[-1] / INIT_CASH - 1
    bmk = bench.iloc[-1] / bench.iloc[0] - 1
    dd = float((equity / equity.cummax() - 1).min())
    closed = trades[trades.action != "BUY"] if len(trades) else trades
    wr = float((closed.ret > 0).mean()) if len(closed) else float("nan")
    idx_lbl = index_tk.split(":")[-1].lstrip("^")
    m = st.columns(5)
    m[0].metric("strategy return", f"{tot:+.1%}")
    m[1].metric(f"{idx_lbl} buy-hold", f"{bmk:+.1%}")
    m[2].metric("max drawdown", f"{dd:.1%}")
    m[3].metric("closed trades", f"{len(closed)}")
    m[4].metric("win rate", f"{wr:.0%}" if wr == wr else "—")

    st.pyplot(_plot_actions(equity, trades, bench,
                            "consensus" if strat == "consensus" else MODEL_LABEL.get(strat, strat), idx_lbl))

    with st.expander("📋 trades — table", expanded=False):
        if len(trades):
            tt = trades.copy()
            tt["date"] = pd.to_datetime(tt.date).dt.date
            tt["ticker"] = tt.ticker.str.split(":").str[-1]
            tt["price"] = tt.price.round(2)
            tt["ret"] = tt.ret.map(lambda x: f"{x:+.1%}" if pd.notna(x) else "")
            st.dataframe(tt[["date", "ticker", "action", "price", "ret"]], hide_index=True, width="stretch")
        else:
            st.write("no closed trades in this window")


def main() -> None:
    try:
        st.set_page_config(page_title="daily predictions", layout="wide")
    except Exception:  # already set by the global launcher — harmless
        pass
    st.title("📈 Daily forward predictions")
    st.caption("Read-only viewer over the committed forward log (`_019`) · refreshed by `/daily-predictions`")
    if not LOG.exists():
        st.error(f"forward log not found: `{LOG}` — run `/daily-predictions` first.")
        return
    df = load_log()
    dates = sorted(df.date.unique())

    with st.sidebar:
        st.header("controls")
        if st.button("🔄 reload log"):
            st.cache_data.clear()
            st.rerun()
        date = st.selectbox("snapshot date", options=dates[::-1], index=0)
        k = st.slider("top-K per model", 1, 10, 3)
        pool_k = st.slider("consensus pool (top-N/model)", 3, 10, 5)
        st.caption(f"{len(dates)} snapshots · {dates[0]} → {dates[-1]}")
        st.divider()
        _render_regime(df[df.date == date])
        st.divider()
        st.caption("**models**  \n" + "  \n".join(f"`{m}` — {MODEL_LABEL[m]}" for m in MODEL_ORDER))

    snap, hist = st.tabs(["📈 Predictions", "📊 Backtests"])
    with snap:
        render_snapshot(df, date, k, pool_k)
    with hist:
        render_backtests(df)


if __name__ == "__main__":
    main()
