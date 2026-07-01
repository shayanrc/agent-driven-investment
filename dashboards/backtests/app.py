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

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "results/backtests/data/forward_predictions_log.csv"

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


def _picks(day: pd.DataFrame, models: list[str], k: int) -> pd.DataFrame:
    rows = []
    for m in models:
        for _, r in day[(day.model == m) & (day["rank"] <= k)].sort_values("rank").iterrows():
            rows.append({"Ticker": r.sym, "Model / Rank": f"{m} / {int(r['rank'])}",
                         "probability": round(float(r.p_calibrated), 3)})
    return pd.DataFrame(rows)


def render_snapshot(df: pd.DataFrame, date, k: int, pool_k: int) -> None:
    day = df[df.date == date]
    st.subheader(f"📅 {date}")

    # Regime gate — one metric per universe index (SMA200 overlay).
    gates = (day.groupby("gate_index")
             .agg(on=("regime_on", "first"), close=("gate_close", "first"), sma=("gate_sma200", "first")))
    cols = st.columns(max(len(gates), 1))
    for col, (idx, g) in zip(cols, gates.iterrows()):
        pct = (g.close / g.sma - 1) if g.sma else float("nan")
        col.metric(f"{idx.split(':')[-1]} regime", "🟢 ON" if g.on else "🔴 OFF",
                   f"{pct:+.1%} vs SMA200", delta_color="normal" if g.on else "inverse")
    st.caption("regime ON ⇒ strategies deploy · OFF ⇒ hold cash (`_016`–`_018`)")

    deployed = [m for m in MODEL_ORDER if m in set(day[day.deployed].model)]
    cand = [m for m in MODEL_ORDER if m in set(day[~day.deployed].model)]

    left, right = st.columns(2)
    with left:
        st.markdown("**Deployed champions** — the live signal")
        st.dataframe(_picks(day, deployed, k), hide_index=True, width="stretch")
    with right:
        st.markdown("**Candidates** — tracked for comparison, not live")
        st.dataframe(_picks(day, cand, k), hide_index=True, width="stretch")

    st.markdown("### 🗳️ Cross-model consensus")
    c = consensus(day, pool_k)
    if c.empty:
        st.info("no picks for this day")
        return
    w = c.iloc[0]
    m1, m2 = st.columns([1, 3])
    m1.metric("Consensus winner", w.sym, f"{int(w.models)}/5 · Σp {w.psum:.3f}")
    disp = c.copy()
    disp["majority"] = disp.models.map(lambda n: "✓" if n >= MAJORITY else "")
    disp = disp.rename(columns={"sym": "Stock", "models": "# models", "psum": "Σp", "voters": "voting models"})
    disp["Σp"] = disp["Σp"].round(3)
    m2.dataframe(disp[["Stock", "# models", "Σp", "majority", "voting models"]],
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
        st.caption("**models**  \n" + "  \n".join(f"`{m}` — {MODEL_LABEL[m]}" for m in MODEL_ORDER))

    snap, hist = st.tabs(["📅 Snapshot", "📊 History"])
    with snap:
        render_snapshot(df, date, k, pool_k)
    with hist:
        render_history(df, pool_k)


if __name__ == "__main__":
    main()
