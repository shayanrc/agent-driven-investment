"""Streamlit dashboard for the ``/daily-predictions`` forward log (the ``_019`` node).

Run with::

    streamlit run dashboards/backtests/app.py

A **read-only viewer** over the committed ``results/backtests/data/forward_predictions_log.csv``
— the durable forward-OOS signal record. Two tabs:

  * **Snapshot** — one day's read: SMA200 regime gate, deployed-champion picks, comparison
    candidates, and the cross-model **consensus** (pool the models' top-N, most-voted wins,
    tie → highest Σp; ≥3/5 = panel majority).
  * **History** — consensus-winner timeline + frequency, per-model rank-1 picks, most-picked
    names, and the regime-gate timeline.

No inference — it only reads the log (refreshed by the ``/daily-predictions`` cadence). All the
usual caveats apply: raw ``p`` (calibrated probability, not certainty), bull-only edge (``_028``),
small effective-N, a forward test — not investment advice.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

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


def render_history(df: pd.DataFrame, pool_k: int) -> None:
    dates = sorted(df.date.unique())
    n = st.slider("show last N snapshots", 5, len(dates), min(30, len(dates)))
    sel = dates[-n:]
    sub = df[df.date.isin(sel)]

    ch = []
    for d in sel:
        c = consensus(sub[sub.date == d], pool_k)
        if not c.empty:
            w = c.iloc[0]
            ch.append({"date": d, "winner": w.sym, "# votes": int(w.models), "Σp": round(w.psum, 3)})
    chdf = pd.DataFrame(ch)

    st.markdown("**🗳️ Consensus winner — history**")
    a, b = st.columns([2, 1])
    a.dataframe(chdf.iloc[::-1], hide_index=True, width="stretch", height=300)
    b.caption("winner frequency")
    b.bar_chart(chdf.winner.value_counts())

    st.markdown("**🏅 Rank-1 pick per model**")
    r1 = (sub[sub["rank"] == 1].pivot_table(index="date", columns="model", values="sym", aggfunc="first")
          .reindex(columns=[m for m in MODEL_ORDER if m in set(sub.model)]))
    st.dataframe(r1.iloc[::-1], width="stretch", height=300)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🔁 Most-picked names** (top-3 appearances, all models)")
        st.bar_chart(sub[sub["rank"] <= 3].sym.value_counts().head(15))
    with c2:
        st.markdown("**🚦 Regime gate (SMA200)**")
        reg = sub.groupby(["date", "gate_index"]).regime_on.first().unstack().astype(int)
        reg.columns = [c.split(":")[-1] for c in reg.columns]
        st.line_chart(reg)
        st.caption("1 = risk-ON (deploy) · 0 = risk-OFF (cash)")


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
        render_history(df, pool_k)


if __name__ == "__main__":
    main()
