# data_pipelines

Generic time-series ingestion module. v1 ships two domains: `us_equities` (NYSE / NASDAQ / US indices) and `nse_equities` (NSE India / NIFTY indices).

For the **why**: [`goal.md`](goal.md).
For the **how** (architecture, stages, correctness constraints, per-domain implementation findings): [`V1_IMPLEMENTATION_PLAN.md`](V1_IMPLEMENTATION_PLAN.md).
For **adding a new domain**: [`adding_a_domain.md`](adding_a_domain.md).

## Quick start

```python
from data_pipelines import fetch
import data_pipelines.domains.us_equities   # register us_equities
import data_pipelines.domains.nse_equities  # register nse_equities

df = fetch("NYSE:AAPL", start="2020-01-01", end="2026-05-01")
df = fetch("NSE:RELIANCE", start="2020-01-01", end="2025-12-31")
df = fetch("NIFTY:50",    start="2023-01-01", end="2025-12-31")
```

CLI:

```bash
uv run python -m data_pipelines fetch NYSE:AAPL --start 2020-01-01 --end 2026-05-01
uv run python -m data_pipelines fetch NSE:RELIANCE --start 2020-01-01 --end 2025-12-31
uv run python -m data_pipelines seed --domain us_equities  --universe sp500   --start 2010-01-01 --end 2026-05-22
uv run python -m data_pipelines seed --domain nse_equities --universe nifty50 --start 2020-01-01 --end 2025-12-31
uv run python -m data_pipelines reprocess --domain us_equities --all
uv run python -m data_pipelines list-cached
uv run python -m data_pipelines list-domains
uv run python -m data_pipelines health --domain nse_equities
```

## Data layout

```
data/
├── raw/<provider>/<domain>/<exchange>/<ticker>/<UTC_ts>_<start>_<end>.<ext>
│       # immutable per-fetch audit trail (D8); one file per provider call
└── processed.db
        # SQLite (WAL); per-domain tables:
        #   <domain>_data — canonical rows, PK (ticker, <time_column>)
        #   <domain>_meta — per-ticker provenance JSON (sources[], range, schema_version)
```

Raw is the audit trail and reprocess source — `reprocess` re-derives the
`processed.db` rows for a ticker from `raw/` without any API call. The single
SQLite file replaces the pre-v1.5 per-ticker `daily.parquet` + `_meta.json`
layout: bulk seeds went from ~6 minutes of fsync churn to <30 seconds, and
cross-ticker queries are now SQL instead of a directory walk.

## Provider strategy (us_equities)

| Tier | Provider | Role | When |
|---|---|---|---|
| Seed | Stooq | bulk cold-start, deep backfill | cache empty OR gap > 90 trading days; requires `STOOQ_API_KEY` (free) |
| Update | Tiingo | routine incremental updates | default; requires `TIINGO_API_KEY` |
| Fallback | yfinance | last resort per ticker | only on Tiingo failure; no key |

`adj_close` is split-AND-dividend adjusted for Tiingo / yfinance and split-only
for Stooq; the quality flag is recorded in `_meta.json` (D4).

## Correctness constraints (D1–D8)

See [`V1_IMPLEMENTATION_PLAN.md §Critical correctness constraints`](V1_IMPLEMENTATION_PLAN.md#critical-correctness-constraints).

D1 schema invariance, D2 atomic writes, D3 deterministic reads, D4 adjustment
semantics, D5 typed failure surfaces, D6 API key safety, D7 date/timezone
discipline, D8 raw immutability + reprocess determinism.

## API keys & .env

Both keys are required for v1 — Stooq's CSV endpoint added a key gate in
2026-Q2 (discovered at v1 smoke test).

```bash
cp .env.example .env
$EDITOR .env   # fill in STOOQ_API_KEY and TIINGO_API_KEY
```

`.env` is loaded automatically when `import data_pipelines` runs (via
python-dotenv). Real shell / CI env vars take precedence over `.env`, so
production overrides always win. `.env` is gitignored — never commit it.

Free registration:
- Stooq: <https://stooq.com/q/d/?s=aapl.us&get_apikey> (captcha only)
- Tiingo: <https://api.tiingo.com/account/api/token>

Keys are read at fetch time only; never logged, never embedded in raw
filenames or `_meta.json` (D6).

## Build status

- v1: framework primitives + us_equities domain shipped.
- v1.5: SQLite processed cache (replaces per-ticker parquet/JSON), Russell 1000 seed.
- v1.7: nse_equities domain shipped (jugaad + nselib + yfinance; NIFTY 50
  universe + index; per-prefix chains; vendored jugaad-data subtree with
  cinfo-shape patch). Two framework lifts triggered by this domain are in:
  `chain_for_gap(identifier, ...)` per-prefix routing and dispatcher
  partial-fill continuation.
- Test suite: 327 data_pipelines tests passing. Online smoke tests gated
  on `PYTEST_ONLINE=1` (Stooq) and `TIINGO_API_KEY` (Tiingo) — skipped
  by default.
- Parking lot for follow-ups: [`V2_TBD.md`](V2_TBD.md).
