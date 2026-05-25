---
name: fetch-data
description: Fetch a daily OHLCV time series for one identifier via data_pipelines (handles cache lookup, adapter chain dispatch, schema normalization). Optional — /forecast calls data_pipelines.fetch() internally, so /fetch-data is for when you want to inspect data on its own.
---

# /fetch-data

Single-identifier fetch over the project's `data_pipelines` module. Composes with the cache automatically (warm-cache hits are <1s; cold-cache cold fetches take seconds to minutes depending on the provider chain).

## When to use

- You want to inspect data for an identifier on its own (separate from forecasting).
- You want to warm the cache before a heavy `/tune-preset` or `/forecast` run on an unfamiliar identifier.
- You want to verify which provider chain served a given gap (the `gaps_filled` field tells you).

You DO NOT need to run `/fetch-data` before `/forecast` — `/forecast` calls `data_pipelines.fetch()` internally.

## Usage

```
uv run python -m scripts.data_pipelines.skill_runner fetch \
    --identifier <id> \
    --start <YYYY-MM-DD> --end <YYYY-MM-DD> \
    [--frequency daily] [--json]
```

### Identifier convention

Identifiers carry a domain prefix. See `docs/data_pipelines/goal.md` for the full registry. Common prefixes:

| Prefix | Domain | Example |
|---|---|---|
| `NASDAQ:` | us_equities | `NASDAQ:AAPL` |
| `NYSE:` | us_equities | `NYSE:GS` |
| `INDEX:` | us_equities | `INDEX:^NDX` |
| `NSE:` | nse_equities | `NSE:RELIANCE` |
| `BSE:` | nse_equities | `BSE:RELIANCE` |
| `NIFTY:` | nse_equities | `NIFTY:50`, `NIFTY:BANK` |

### Output

Default: a single human-readable line summarizing rows, range, cache hit/miss, and how many providers failed (with the cache DB path).

`--json`: a structured object including `gaps_filled` (which adapter served which gap), `providers_failed` (per-provider failure reasons), and `cache_path`.

## Examples

```
# Warm cache hit.
uv run python -m scripts.data_pipelines.skill_runner fetch \
    --identifier NASDAQ:AAPL --start 2020-01-01 --end 2024-12-31

# Cold fetch with JSON output (useful when composing with another agent).
uv run python -m scripts.data_pipelines.skill_runner fetch \
    --identifier NSE:RELIANCE --start 2020-01-01 --end 2024-12-31 --json
```

## Errors

| Exit code | Cause |
|---|---|
| 0 | success |
| 2 | `UnknownDomain` (unregistered prefix), `MissingAPIKey` (provider key not configured), `ProviderError` / `AllProvidersFailed` (every adapter in the domain's chain failed on this request) |

All error classes carry the identifier in the stderr message.

## Notes

- `frequency` is `daily` only in v1.
- The cache is at `data/processed.db` — a single SQLite file. Inspect it via `/data-health` rather than reading raw SQL.
