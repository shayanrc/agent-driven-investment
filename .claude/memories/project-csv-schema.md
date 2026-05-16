---
name: project-csv-schema
description: CSV schema for analog_mc v1 — configurable column names; NASDAQ100.csv specifics
metadata:
  type: project
---

The data loader takes column names from config (`date_col`, `close_col`), not hardcoded. The CSV may have only date + close (no OHLCV).

**NASDAQ100.csv specifics** (FRED-style):
- `observation_date` (YYYY-MM-DD)
- `NASDAQ100` (close price; no Adj Close, no OHLC, no volume)
- ~10,532 rows from 1986-01-02 to 2026-05-15

**Why:** Different sources name their close column differently (yfinance uses `Close`/`Adj Close`, FRED uses the asset name itself). Hardcoding breaks the asset-agnostic design.

**How to apply:** In the config, expose `date_col: str` and `close_col: str` (default to the NASDAQ100.csv values for v1). The loader returns log returns and a Series indexed by date. See [[project-data-source]].
