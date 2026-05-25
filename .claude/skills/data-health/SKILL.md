---
name: data-health
description: Report cache coverage and freshness for the data_pipelines processed cache. Three call modes — project-wide, per-identifier, per-domain. Use before /forecast when in doubt about cache freshness.
---

# /data-health

Diagnostic report over the `data_pipelines` processed cache (`data/processed.db`). Three modes — pick the one matching your question.

## When to use

- Before `/forecast --identifier <id>` when you do not know whether the cache covers your requested range.
- Before `/tune-preset` to confirm the asset has enough history.
- For project-wide cache audit (no args).

## Usage

```
uv run python -m scripts.data_pipelines.skill_runner health \
    [--identifier <id>] [--domain <name>] [--data-root <path>] [--json]
```

### Modes

| Args | Mode | Output |
|---|---|---|
| (none) | Project-wide audit | Per-domain breakdown of identifier counts, total rows, oldest/newest `last_fetch_utc`. |
| `--identifier NASDAQ:AAPL` | Single identifier | Schema version, row count, range covered, last fetch UTC, source list (provider + adjustment quality + covers). |
| `--domain us_equities` | Per-domain | Per-identifier table within that domain. |

### Options

| Flag | Purpose |
|---|---|
| `--data-root` | Override the default `data/` root (useful for tests with a temp DB). |
| `--json` | Emit a JSON object; same payload as the table but parseable. |

## Examples

```
# Project-wide audit.
uv run python -m scripts.data_pipelines.skill_runner health

# Single identifier (no cache → reports cached=false).
uv run python -m scripts.data_pipelines.skill_runner health --identifier NASDAQ:AAPL

# Domain audit.
uv run python -m scripts.data_pipelines.skill_runner health --domain nse_equities

# Machine-readable for composition.
uv run python -m scripts.data_pipelines.skill_runner health --identifier NASDAQ:AAPL --json
```

## Notes

- Cheap (<1s typical) — reads only the cache's meta tables; data tables are not scanned.
- An identifier that has never been fetched is reported as `cached=false`. Run `/fetch-data` to populate it.
- The `sources` list per identifier reflects every provider that contributed rows (provenance for the adjustment-quality merge per the `data_pipelines` schema docs).
