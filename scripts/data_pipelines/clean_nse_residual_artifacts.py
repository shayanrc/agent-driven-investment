"""Surgical cleanup of the residual NSE price artifacts the yfinance re-seed did
not catch (V5, post-re-seed). Explicit verified fixes only — no heuristics.

- SPLITS: yfinance's corporate-actions feed missed these mid-cap splits, so the
  pre-split prices stayed on the old basis. Multiply pre-ex-date OHLC by the exact
  price step (close[ex]/close[ex-1]); volume by the inverse. Returns are invariant
  to the exact ratio (a constant multiplicative factor cancels), so this removes
  the discontinuity cleanly. Verified sustained integer-ratio steps.
- ZEROVOL: leading/stale `volume==0` placeholder bars at a wrong price that create
  a fake spike when real trading begins. Drop the vol==0 bars for these equities
  (vol==0 is never a real bar for a liquid nifty500 name — unlike NIFTY:* indices,
  which are left untouched).
- BADPRINT: isolated single-bar bad prints (INDIAMART Diwali-Muhurat session).
  Replace OHLC with the mean of the two neighbours.

NOT touched (documented residuals): demergers (ABFRL, VEDL, JSL — real value events,
not splits), CDSL (IPO listing-day pop, not a split), and genuine crashes/rallies
(YESBANK, IDEA/Vodafone AGR, bank moves).
"""
import sqlite3

import pandas as pd

from data_pipelines.cache import processed_db_path

TABLE = "nse_equities_data"
SPLITS = ["NSE:CGCL", "NSE:GPIL", "NSE:MOTILALOFS"]  # ex-date auto-detected (the >35% step)
ZEROVOL = ["NSE:WHIRLPOOL", "NSE:NESTLEIND", "NSE:ABBOTINDIA",
           "NSE:KIRLOSENG", "NSE:J&KBANK", "NSE:PATANJALI"]
BADPRINT = [("NSE:INDIAMART", "2019-10-27"), ("NSE:INDIAMART", "2020-11-14")]
OHLC = ["open", "high", "low", "close", "adj_close"]


def _load(con, tk):
    df = pd.read_sql(f"SELECT * FROM {TABLE} WHERE ticker = ? ORDER BY date", con,
                     params=(tk,))
    df["_d"] = pd.to_datetime(df["date"])
    return df


def _write(con, tk, df):
    df = df.drop(columns=["_d"])
    con.execute(f"DELETE FROM {TABLE} WHERE ticker = ?", (tk,))
    df.to_sql(TABLE, con, if_exists="append", index=False)


def main() -> int:
    con = sqlite3.connect(processed_db_path("data"))
    log = []

    for tk in SPLITS:
        df = _load(con, tk)
        mv = df["close"].pct_change()
        exi = mv.abs().idxmax()  # the split step (largest move — verified to be the split)
        ratio = df["close"].iloc[exi] / df["close"].iloc[exi - 1]
        mask = df.index < exi
        for c in OHLC:
            df.loc[mask, c] = df.loc[mask, c] * ratio
        df.loc[mask, "volume"] = (df.loc[mask, "volume"] / ratio).round().astype("int64")
        _write(con, tk, df)
        log.append(f"SPLIT {tk}: ex={str(df['_d'].iloc[exi])[:10]} ratio={ratio:.4f} "
                   f"({int(mask.sum())} pre-rows adjusted)")

    for tk in ZEROVOL:
        df = _load(con, tk)
        n0 = int((df["volume"] == 0).sum())
        df = df[df["volume"] != 0].reset_index(drop=True)
        _write(con, tk, df)
        log.append(f"ZEROVOL {tk}: dropped {n0} vol==0 bars")

    for tk, d in BADPRINT:
        df = _load(con, tk)
        i = df.index[df["_d"] == pd.Timestamp(d)]
        if len(i) == 0:
            log.append(f"BADPRINT {tk} {d}: date not found — skipped")
            continue
        i = i[0]
        if 0 < i < len(df) - 1:
            for c in OHLC + ["volume"]:
                df.loc[i, c] = (df[c].iloc[i - 1] + df[c].iloc[i + 1]) / 2
            df.loc[i, "volume"] = int(df.loc[i, "volume"])
            _write(con, tk, df)
            log.append(f"BADPRINT {tk} {d}: interpolated")

    con.commit()
    con.close()
    print("\n".join(log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
