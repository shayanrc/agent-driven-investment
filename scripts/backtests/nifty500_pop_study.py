#!/usr/bin/env python
"""Post-pop event study on nifty500 — reproduces docs/backtests/_031.

Event = a stock whose intraday HIGH clears the previous close by >10%
(split-adjusted). We characterize what happens next: forward returns, path
excursions (max run-up / max drawdown), volume conditioning, temporal ordering
of the peak vs trough, and a battery of exit/entry rules (dip ladders,
stop-losses, take-profit targets) benchmarked against buy-and-hold.

All prices are split-adjusted (adj factor f = adj_close/close applied to
open/high/low). Returns are measured from a stated entry (event close, or the
next day's open). Max drawdown is running-peak-relative on the close path
(entry = initial peak). Max run-up is the peak favorable excursion vs entry.

Run:
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
      uv run python -m scripts.backtests.nifty500_pop_study
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import data_pipelines.domains.nse_equities  # noqa: F401 (registers domain)
from data_pipelines.domains.nse_equities.universe import load_universe

POP = 0.10           # intraday-high vs prev-close threshold
VOL_MULT = 5.0       # "large volume spike" = vol / trailing-20d median >= this
VOL_BASE = 20        # trailing window for the volume baseline
HORIZONS = [1, 5, 10, 20, 50, 200]


def load_panel(data_root: str = "data") -> pd.DataFrame:
    con = sqlite3.connect(f"{data_root}/processed.db")
    uni = [t for t in load_universe("nifty500") if t.startswith("NSE:")]
    q = ("SELECT ticker,date,open,high,low,close,adj_close,volume "
         "FROM nse_equities_data WHERE ticker IN ({})").format(",".join("?" * len(uni)))
    df = pd.read_sql_query(q, con, params=uni)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["open", "high", "low", "close", "adj_close"])
    df = df[(df["close"] > 0) & (df["adj_close"] > 0) & (df["open"] > 0)]
    return df.sort_values(["ticker", "date"])


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per >10%-pop event with per-horizon return/run-up/MDD.

    Returns from the EVENT CLOSE. Also carries next-day-open entry stats and the
    volume ratio, plus the day-index of the max-drawdown trough and max-run-up
    peak (for the temporal-ordering analysis).
    """
    recs = []
    for _, g in df.groupby("ticker", sort=False):
        c = g["close"].values; ac = g["adj_close"].values
        h = g["high"].values; lo = g["low"].values; o = g["open"].values
        v = g["volume"].values.astype(float)
        n = len(ac)
        if n < 3:
            continue
        f = ac / c
        ah, al, ao = h * f, lo * f, o * f
        prev = np.full(n, np.nan); prev[1:] = ac[:-1]
        pop = (ah - prev) / prev
        for i in np.where(pop > POP)[0]:
            if i < VOL_BASE + 1 or i + 1 >= n:
                continue
            base = np.nanmedian(v[i - VOL_BASE:i])
            vr = (v[i] / base) if (base and base > 0 and np.isfinite(v[i])) else np.nan
            maxN = min(HORIZONS[-1], n - 1 - i)
            close_path = ac[i:i + maxN + 1]
            rm = np.maximum.accumulate(close_path)
            dd = (close_path - rm) / rm
            ru = close_path / close_path[0] - 1.0
            rec = {"pop": pop[i], "vr": vr, "maxN": maxN,
                   "open_next": ao[i + 1], "R_close": ac[i]}
            for H in HORIZONS:
                if maxN >= H:
                    rec[f"ret{H}"] = close_path[H] / close_path[0] - 1.0
                    rec[f"ru{H}"] = ru[:H + 1].max()
                    rec[f"mdd{H}"] = dd[:H + 1].min()
                    # ordering within [1..H]
                    rec[f"tdd{H}"] = int(np.argmin(dd[1:H + 1])) + 1
                    rec[f"tru{H}"] = int(np.argmax(ru[1:H + 1])) + 1
            recs.append(rec)
    return pd.DataFrame(recs)


def pct(x):
    return f"{100 * x:+.1f}%"


def tbl_unconditional(E, vol_filter=None, label=""):
    F = E if vol_filter is None else E[E["vr"] > vol_filter]
    print(f"\n### Forward performance from event close {label} (N base {len(F):,})")
    print(f"{'H':>6} {'N':>6} {'mean':>8} {'median':>8} {'%pos':>6} "
          f"{'mean RU':>8} {'med RU':>8} {'mean MDD':>9} {'med MDD':>8}")
    for H in HORIZONS:
        s = F[f"ret{H}"].dropna(); u = F[f"ru{H}"].dropna(); m = F[f"mdd{H}"].dropna()
        if not len(s):
            continue
        print(f"{('+'+str(H)+'d'):>6} {len(s):>6} {pct(s.mean()):>8} {pct(s.median()):>8} "
              f"{100*(s>0).mean():>5.1f}% {pct(u.mean()):>8} {pct(u.median()):>8} "
              f"{pct(m.mean()):>9} {pct(m.median()):>8}")


def tbl_volume_prevalence(E):
    V = E.dropna(subset=["vr"])
    print(f"\n### Volume-spike prevalence (N {len(V):,}, median ratio {V['vr'].median():.2f}x)")
    for thr in (1.5, 2, 3, 5, 10):
        n = (V["vr"] >= thr).sum()
        print(f"  >= {thr:>4.1f}x   {n:>6}  ({100*n/len(V):>5.1f}%)")


def tbl_ordering(E, vol_filter=VOL_MULT):
    F = E[E["vr"] > vol_filter]
    print(f"\n### Max-DD vs max-RU ordering (>5x vol, N base {len(F):,})")
    print(f"{'H':>6} {'N':>6} {'DD before RU':>13} {'RU before DD':>13} {'med day DD':>11} {'med day RU':>11}")
    for H in [5, 20, 50, 200]:
        sub = F.dropna(subset=[f"tdd{H}", f"tru{H}"])
        if not len(sub):
            continue
        b = (sub[f"tdd{H}"] < sub[f"tru{H}"]).mean()
        a = (sub[f"tdd{H}"] > sub[f"tru{H}"]).mean()
        print(f"{('+'+str(H)+'d'):>6} {len(sub):>6} {100*b:>12.1f}% {100*a:>12.1f}% "
              f"{sub[f'tdd{H}'].median():>10.0f} {sub[f'tru{H}'].median():>10.0f}")


def strategy_battery(df):
    """Dip ladder, buy-and-hold (close & next-open), stops, and targets at 200d.

    Recomputes from OHLC because the ladders/stops need the full forward
    intraday low/high path, not just the checkpoint returns.
    """
    W = 200
    ev = []  # (R_close, open_next, highs[1..W], lows[1..W], closes[1..W], end_close)
    for _, g in df.groupby("ticker", sort=False):
        c = g["close"].values; ac = g["adj_close"].values
        h = g["high"].values; lo = g["low"].values; o = g["open"].values
        v = g["volume"].values.astype(float)
        n = len(ac)
        if n < 3:
            continue
        f = ac / c
        ah, al, ao = h * f, lo * f, o * f
        prev = np.full(n, np.nan); prev[1:] = ac[:-1]
        pop = (ah - prev) / prev
        for i in np.where(pop > POP)[0]:
            if i < VOL_BASE + 1 or i + W >= n:
                continue
            base = np.nanmedian(v[i - VOL_BASE:i])
            vr = (v[i] / base) if (base and base > 0 and np.isfinite(v[i])) else np.nan
            if not (vr and vr > VOL_MULT):
                continue
            ev.append((ac[i], ao[i + 1], ah[i + 1:i + W + 1], al[i + 1:i + W + 1],
                       ac[i + 1:i + W + 1], ac[i + W]))
    print(f"\n### Strategy battery — >5x vol, 200d window, {len(ev):,} events")

    def mdd_upto(P0, closes):
        path = np.concatenate(([P0], closes))
        rm = np.maximum.accumulate(path)
        return ((path - rm) / rm).min()

    # buy-and-hold, next-open entry
    bret = np.array([endc / P0 - 1 for _, P0, _, _, _, endc in ev])
    bmdd = np.array([mdd_upto(P0, C) for _, P0, _, _, C, _ in ev])
    print(f"\n[buy next-open, hold 200d]  mean {pct(bret.mean())}  median {pct(np.median(bret))} "
          f" %pos {100*(bret>0).mean():.1f}%  med MDD {pct(np.median(bmdd))}")

    print("\n[dip ladder — GTC limit below event close, hold to window end]")
    print(f"{'level':>6} {'fill%':>7} {'med fill day':>13} {'cond mean':>10} {'cond med':>9} {'med MDD':>9} {'uncond mean':>12}")
    for L in (5, 10, 15, 20, 25, 30):
        fills = []
        for R, _, H, Lo, C, _ in ev:
            limit = R * (1 - L / 100.0)
            hit = np.where(Lo <= limit)[0]
            if hit.size:
                k = hit[0]
                fills.append((C[-1] / limit - 1.0, k + 1, mdd_upto(limit, C[k:])))
        fr = len(fills) / len(ev)
        ret = np.array([x[0] for x in fills]); fd = np.array([x[1] for x in fills])
        mdd = np.array([x[2] for x in fills])
        print(f"{('-'+str(L)+'%'):>6} {100*fr:>6.1f}% {np.median(fd):>12.0f} {pct(ret.mean()):>10} "
              f"{pct(np.median(ret)):>9} {pct(np.median(mdd)):>9} {pct(fr*ret.mean()):>12}")

    def run_exit(mode, S, side):
        """side='stop' (hard/trail) or 'target'. Entry = next-open."""
        rets, mdds, stopped = [], [], []
        for _, P0, H, Lo, C, endc in ev:
            peak, ex = P0, None
            for k in range(W):
                if side == "target":
                    if H[k] >= P0 * (1 + S / 100.0):
                        ex = (P0 * (1 + S / 100.0), k); break
                else:
                    lvl = (peak if mode == "trail" else P0) * (1 - S / 100.0)
                    if Lo[k] <= lvl:
                        ex = (lvl, k); break
                    if mode == "trail" and H[k] > peak:
                        peak = H[k]
            if ex is None:
                rets.append(endc / P0 - 1); mdds.append(mdd_upto(P0, C)); stopped.append(0)
            else:
                kk = ex[1]
                rets.append(ex[0] / P0 - 1)
                mdds.append(mdd_upto(P0, C[:kk + 1]))
                stopped.append(1)
        return np.array(rets), np.array(mdds), np.array(stopped)

    for mode, lab in (("hard", "HARD stop (entry)"), ("trail", "TRAIL stop (peak)")):
        print(f"\n[{lab} — buy next-open]")
        print(f"{'level':>6} {'hit%':>6} {'mean':>8} {'median':>8} {'%pos':>6} {'med MDD':>8}")
        for S in (10, 15, 20, 25, 30, 40):
            r, m, st = run_exit(mode, S, "stop")
            print(f"{('-'+str(S)+'%'):>6} {100*st.mean():>5.1f}% {pct(r.mean()):>8} "
                  f"{pct(np.median(r)):>8} {100*(r>0).mean():>5.1f}% {pct(np.median(m)):>8}")

    print("\n[take-profit target — buy next-open]")
    print(f"{'target':>7} {'hit%':>6} {'mean':>8} {'median':>8} {'%pos':>6} {'med MDD':>8}")
    for T in (5, 10, 15, 20, 30, 50):
        r, m, st = run_exit(None, T, "target")
        print(f"{('+'+str(T)+'%'):>7} {100*st.mean():>5.1f}% {pct(r.mean()):>8} "
              f"{pct(np.median(r)):>8} {100*(r>0).mean():>5.1f}% {pct(np.median(m)):>8}")

    print("\n[next-day-only limit fill rate — reference = event close]")
    for L in (1, 2, 3, 5):
        hits = sum(1 for R, _, _, Lo, _, _ in ev if Lo[0] <= R * (1 - L / 100.0))
        print(f"  -{L}%  {100*hits/len(ev):.1f}%")


def main():
    df = load_panel()
    print(f"loaded {len(df):,} bars, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()}→{df['date'].max().date()}")
    E = build_events(df)
    print(f">10% intraday-pop events: {len(E):,}")
    tbl_unconditional(E, None, "(all events)")
    tbl_volume_prevalence(E)
    tbl_unconditional(E, VOL_MULT, "(>5x volume)")
    tbl_ordering(E)
    strategy_battery(df)
    out = Path("results/backtests/data/nifty500_pop_events.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    E.to_parquet(out)
    print(f"\nevent table -> {out}")


if __name__ == "__main__":
    main()
