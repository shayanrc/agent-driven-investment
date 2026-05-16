---
name: project-data-source
description: Data source decision for analog_mc v1 — single CSV file, loader module deferred
metadata:
  type: project
---

analog_mc v1 reads from a single local CSV at `data/NASDAQ100.csv`. No yfinance, no automated download, no multi-ticker dispatch.

**Why:** User wants to keep v1 minimal and add a proper data-loader module later when multi-ticker / multi-source support is needed. The plan's "open question 1" (data source) was answered with "local CSVs only, add loader module later." Keeps the v1 implementation small and reproducible.

**How to apply:** `src/analog_mc/data.py` should accept a path and column-name mapping from config, not try to be a multi-source loader. When the user asks to "add the data loader," that's a separate, larger module — don't pre-bake hooks for it. See [[project-csv-schema]] for the file format.
