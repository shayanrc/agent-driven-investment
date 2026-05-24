---
name: project-data-source
description: analog_mc v1 reads from a single local CSV. The general-purpose loader (data_pipelines) has since shipped as its own module, but analog_mc has not switched over.
metadata:
  type: project
---

analog_mc v1 reads from a single local CSV at `data/NASDAQ100.csv`. No yfinance, no automated download, no multi-ticker dispatch *inside analog_mc*.

**Why:** User wanted v1 minimal and reproducible. The plan's "open question 1" was answered with "local CSVs only, add loader module later." That deferred loader has since landed as the **data_pipelines** module (see `[[project-overview]]`) — `data_pipelines.fetch("NYSE:AAPL", start, end)` is now available — but analog_mc itself has not been wired to consume it. The CSV-first contract still holds for analog_mc v1.

**How to apply:**
- For changes inside `src/analog_mc/`: keep the CSV-first contract. `data.py` accepts a path + column-name mapping from config; don't pre-bake hooks for data_pipelines integration unless the user explicitly asks to wire it up.
- For changes outside analog_mc, or new modules that need historical price data: use `data_pipelines.fetch(...)` rather than ad-hoc CSVs or vendor scripts.
- "Wire analog_mc to data_pipelines" would be a real plan (`V<N>_PLAN.md` + branch), not a casual change — gate on the user explicitly asking for it.

See `[[project-csv-schema]]` (TBD) for the analog_mc CSV file format.
