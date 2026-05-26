# Universe YAML — format spec

Canonical specification for the universe-constituent YAML files shipped under
`configs/data_pipelines/domains/<domain>/universe_*.yaml`, and for the
matching `universes::<name>` registry block that the `gbdt` module reads from
`configs/gbdt/default.yaml`.

The two pieces are described together because they reference each other and
the lint test (`tests/data_pipelines/test_universe_yaml_lint.py`) walks both.

---

## 1. Purpose

A *universe* is a named, dated, ordered set of tradable identifiers (plus the
benchmark index those identifiers belong to). It is the input contract to
downstream modules — `gbdt` panel construction, `analog_mc` cohort
selection, etc. — so its on-disk shape must be stable, lint-checkable, and
free of decorative fields.

The universe YAML answers two questions:

- **What is in this universe right now?** (current constituent list, as of a
  pinned `listed_at` date).
- **Which benchmark index does it map to?** (for index-relative feature
  families, fan-chart anchoring, etc.).

Point-in-time historical membership is **explicitly out of scope** for this
format (see `docs/data_pipelines/V2_TBD.md` open question 1). The format is a
"current-snapshot" contract.

---

## 2. File location & naming

```
configs/data_pipelines/domains/<domain>/universe_<name>.yaml
```

- `<domain>` is one of the registered domain directories — currently
  `nse_equities` or `us_equities`. One universe YAML lives in exactly one
  domain directory.
- `<name>` is snake_case and matches the top-level `universe:` field
  *exactly* (the lint test asserts this). Filename → universe name is the
  canonical mapping.

Examples:

```
configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml
configs/data_pipelines/domains/nse_equities/universe_nifty_midcap_150.yaml
configs/data_pipelines/domains/us_equities/universe_sp500.yaml
```

---

## 3. Universe YAML schema

Every universe YAML has exactly these four top-level keys. **No other
top-level keys are permitted** — the lint test rejects extras.

| Field        | Type           | Required | Description                                                                                                                                              |
| ------------ | -------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `universe`   | `str`          | yes      | The universe's canonical name in snake_case. **Must match the filename stem** after the `universe_` prefix is stripped (`universe_nifty50.yaml` → `nifty50`). |
| `listed_at`  | `date` (ISO)   | yes      | Pinned snapshot date for this constituent list. Bumped manually when the list is refreshed.                                                              |
| `indices`    | `list[str]`    | yes      | One or more benchmark index identifiers (in cache-prefixed form, e.g. `"NIFTY:50"`, `"INDEX:^SPX"`) that this universe is associated with. Non-empty.    |
| `tickers`    | `list[str]`    | yes      | The constituent list. **Fully prefixed** (`"NSE:RELIANCE"`, `"NASDAQ:AAPL"`, `"NYSE:JPM"`) — the framework does not synthesize prefixes from metadata. Non-empty. |

### Ticker-prefix conventions

Constituents always carry their domain prefix. The framework looks at the
prefix to route the read to the right cache table; it never strips or
appends a prefix based on metadata elsewhere. So the YAML is the single
source of truth for what identifier the cache will be queried with.

| Domain         | Constituent prefix(es) used | Example                     |
| -------------- | --------------------------- | --------------------------- |
| `nse_equities` | `NSE:`                      | `NSE:RELIANCE`              |
| `us_equities`  | `NASDAQ:`, `NYSE:`          | `NASDAQ:AAPL`, `NYSE:JPM`   |

US universes (sp500, russell1000) routinely span both NASDAQ and NYSE within
the same `tickers:` list; this is expected, not an error.

### Minimal example

```yaml
# NIFTY 50 — current constituents (as of 2026-05-24).
# Source: nselib.capital_market.nifty50_equity_list().

universe: nifty50
listed_at: 2026-05-24
indices:
  - "NIFTY:50"
tickers:
  - "NSE:RELIANCE"
  - "NSE:TCS"
  - "NSE:HDFCBANK"
  # ... 47 more
```

---

## 4. Derivation invariants (NSE family)

Several NSE universes are derived as set unions of finer-grained ones. When
refreshing constituents, the relationship must hold; the lint test does not
enforce this (cost: would require loading and unioning multi-thousand-ticker
lists on every test run) but the comment at the top of each derived YAML
documents the invariant so a refresh script can validate it.

| Derived universe          | Definition                                                  |
| ------------------------- | ----------------------------------------------------------- |
| `nifty100`                | `nifty50 ∪ nifty_next_50`                                   |
| `nifty500`                | `nifty100 ∪ nifty_midcap_150 ∪ nifty_smallcap_250`          |
| `nifty_midsmallcap_400`   | `nifty_midcap_150 ∪ nifty_smallcap_250`                     |
| `nifty_largemidcap_250`   | `nifty100 ∪ nifty_midcap_150`                               |
| `nifty_total_market`      | `nifty500 ∪ nifty_microcap_250`                             |

US universes (sp500, nasdaq100, russell1000) are independent constituent
lists; no derivation invariants apply.

---

## 5. gbdt registry contract

To make a universe usable by the `gbdt` runner, the universe YAML must be
*registered* under `configs/gbdt/default.yaml::universes::<name>`. The
registry entry is what the runner consults when a spec names a universe;
it does not auto-discover YAMLs from disk.

The registry block has exactly these three required keys per universe.
**No other keys are permitted** — the lint test rejects extras.

| Field                  | Type           | Required | Description                                                                                                                                              |
| ---------------------- | -------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source`               | `str`          | yes      | Repo-relative path to the constituent YAML (`configs/data_pipelines/domains/<domain>/universe_<name>.yaml`). The file must exist.                       |
| `index_ticker`         | `str`          | yes      | Benchmark identifier in the data_pipelines cache. Drives `F1`/`F5`/`F9`/`F9b` macro features. NSE uses the `NIFTY:` family (e.g. `"NIFTY:100"`); US uses the `INDEX:` family (e.g. `"INDEX:^SPX"`). |
| `annualization_factor` | `int`          | yes      | Trading days per year used for `√t` vol annualization. `250` for NSE, `252` for US.                                                                      |

### Example

```yaml
universes:
  nifty50:
    source: configs/data_pipelines/domains/nse_equities/universe_nifty50.yaml
    index_ticker: "NIFTY:50"
    annualization_factor: 250
  sp500:
    source: configs/data_pipelines/domains/us_equities/universe_sp500.yaml
    index_ticker: "INDEX:^SPX"
    annualization_factor: 252
```

### Hard-fail behavior

`gbdt.data.universe_metadata(name)` raises `KeyError` when the spec names a
universe that has no registry block — *no silent fallback to NIFTY-style
defaults*. This is the PR #100 fix: a missing block previously caused
`nifty100` to be benchmarked against `NIFTY:50` (Exp 2). The pre-flight
flow in `.claude/skills/gbdt-experiment/SKILL.md` § "Universe self-service"
is responsible for writing the block before the runner sees the spec.

---

## 6. Adding a new universe

1. **Write the constituent YAML** at `configs/data_pipelines/domains/<domain>/universe_<name>.yaml` following the schema in section 3. Use fully-prefixed tickers.
2. **Register the universe** by appending a `universes::<name>` block under `configs/gbdt/default.yaml::universes` with `source`, `index_ticker`, and `annualization_factor` (section 5).
3. **Run the lint test** — `uv run pytest tests/data_pipelines/test_universe_yaml_lint.py -v`. Fix any schema violations before continuing.
4. **Back-fill the cache** by running `data_pipelines.fetch()` per ticker (the `/gbdt-experiment` skill's pre-flight does this automatically; for ad-hoc use, `uv run python -m data_pipelines fetch <ticker> <start> <end>` per identifier).
5. **Reference the universe by name** in experiment specs (`target.universe: <name>`); the runner resolves the rest from the registry.
