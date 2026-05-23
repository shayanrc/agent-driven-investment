# data_pipelines

Generic time-series ingestion module. v1 ships one domain (us_equities).

For the **why**: [`goal.md`](goal.md).
For the **how** (architecture, stages, correctness constraints): [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).
For **adding a new domain**: [`adding_a_domain.md`](adding_a_domain.md).

## Quick start

```python
from data_pipelines import fetch
import data_pipelines.domains.us_equities   # register the domain

df = fetch("NYSE:AAPL", start="2020-01-01", end="2026-05-01")
```

CLI:

```bash
uv run python -m data_pipelines fetch NYSE:AAPL --start 2020-01-01 --end 2026-05-01
uv run python -m data_pipelines seed --universe sp500 --start 2010-01-01 --end 2026-05-22
uv run python -m data_pipelines reprocess --all
uv run python -m data_pipelines list-cached
uv run python -m data_pipelines list-domains
uv run python -m data_pipelines health --domain us_equities
```

## Data layout

```
data/
├── raw/<provider>/<domain>/<exchange>/<ticker>/<UTC_ts>_<start>_<end>.<ext>
│       # immutable per-fetch audit trail
└── processed/<domain>/<exchange>/<ticker>/
    ├── daily.parquet
    └── _meta.json
```

Raw is the audit trail and reprocess source — `reprocess` re-derives `processed/`
from `raw/` without any API call.

## Provider strategy (us_equities)

| Tier | Provider | Role | When |
|---|---|---|---|
| Seed | Stooq | bulk cold-start, deep backfill | cache empty OR gap > 90 trading days; requires `STOOQ_API_KEY` (free) |
| Update | Tiingo | routine incremental updates | default; requires `TIINGO_API_KEY` |
| Fallback | yfinance | last resort per ticker | only on Tiingo failure; no key |

`adj_close` is split-AND-dividend adjusted for Tiingo / yfinance and split-only
for Stooq; the quality flag is recorded in `_meta.json` (D4).

## Correctness constraints (D1–D8)

See [`IMPLEMENTATION_PLAN.md §Critical correctness constraints`](IMPLEMENTATION_PLAN.md#critical-correctness-constraints).

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

- v1: framework primitives + us_equities domain complete (201 unit tests passing).
- Online smoke tests gated on `PYTEST_ONLINE=1` (Stooq) and `TIINGO_API_KEY`
  (Tiingo) — both skipped by default.
- Next: agent-tool wrapper around `fetch_with_meta`; second domain plug-in
  (FRED / NSE / commodities) once needed.
