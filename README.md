# agent-driven-investment

Investment research infrastructure driven by AI agents. Claude acts as the quantitative researcher — fetching data, running probabilistic forecasts, executing ML experiments, and interpreting results — through a composable suite of agent-callable tools. The human sets direction and reads the reports; the agent does the heavy lifting.

---

## Modules

The repo hosts five independently versioned modules under a shared layout:

| Module | What it does | Status |
|---|---|---|
| **analog_mc** | Probabilistic 60-day price-path forecasting via analog Monte Carlo | v2.4 canonical; v5 experiments completed |
| **data_pipelines** | Generic time-series ingestion: fetch, cache, normalize across providers | v1 shipped; US equities + NSE equities domains |
| **forecasters** | Agent-callable forecasting surface with saved-model presets | v1 shipped; analog_mc wired as backend #1 |
| **gbdt** | Categorical-outcome GBDT classifiers for event-probability forecasting | v1 shipped; CatBoost + agent-driven FS/HP loop |
| **backtesting** | Multi-asset backtesting engine with structural look-ahead-bias elimination | v1 shipped |

Each module has a goal doc at `docs/<module>/goal.md` that defines success criteria and unacceptable trade-offs.

---

## The agent surface

Seven Claude Code skills compose into an end-to-end research workflow:

```
/data-health           Check cache coverage and freshness
/fetch-data            Fetch daily OHLCV for any identifier via data_pipelines
/list-presets          Discover available forecasting presets
/forecast              Run a preset against a time series (single-origin fan chart)
/tune-preset           Fit a new preset via walk-forward grid search (hours of compute)
/gbdt-experiment       Run a GBDT experiment end-to-end from a YAML spec
/arxiv-search          Search arXiv for related papers
```

The typical agent workflow:

1. `/data-health` to check whether cached data covers the target asset and range.
2. `/fetch-data` to backfill gaps (or skip — `/forecast` calls `data_pipelines.fetch()` internally).
3. `/list-presets` to find a tuned preset, or `/tune-preset` to create one.
4. `/forecast` to produce a calibrated probabilistic forecast.
5. `/gbdt-experiment` to run event-probability experiments on specific (universe, direction, threshold, horizon) cells.

---

## What each module does

### analog_mc

Produces calibrated Monte Carlo distributions of forward price paths. Given a daily price series and a forecast origin date:

1. Computes causal multi-horizon z-scores (20/50/200-day trailing windows) and EWMA volatility.
2. Finds historical analogs via weighted Euclidean distance on the z-score vectors.
3. Converts distances to probabilities via n_eff-parameterized softmax (temperature solved so effective sample size = n_eff).
4. Samples forward 10-day blocks of realized returns from the chosen analogs.
5. Vol-scales each block to the current regime, removing analog drift against a shared baseline.
6. Maintains running EWMA volatility across blocks for path-coherent scaling.

Walk-forward validation searches (weights, n_eff) per fold on a held-out block. Diagnostic-gated upgrades (PIT histograms, reliability diagrams, ACF comparison, fat-tail anchor panel) decide whether complexity additions ship.

Six critical correctness constraints (C1-C6) are non-negotiable: causal rolling features, n_eff parameterization, per-analog vol scaling, running EWMA, forward-only sampling, and walk-forward boundary discipline.

**Key results:** v2.1 trailing-momentum drift eliminated PIT slope and reduced high-vol CRPS by 5.2%. v2.2 conditional block sampling implemented but deferred — the ACF gap is structural, not a seam artifact. v4 experiments (local-linear, corrwindow, joint) improved fat-tail anchors but regressed elsewhere; none promoted.

### data_pipelines

Generic time-series ingestion framework. A single `fetch()` call handles provider selection, caching, fallbacks, and schema normalization:

```python
from data_pipelines import fetch
df = fetch("NYSE:AAPL", start="2010-01-01", end="2026-05-15")
df = fetch("NIFTY:RELIANCE", start="2010-01-01", end="2026-05-15")
```

**Shipped domains:**
- **US equities** — NYSE + NASDAQ, S&P 500 / NASDAQ 100 / Russell 1000 universes. Adapter chain: Stooq (bulk seed) → Tiingo (incremental) → yfinance (fallback).
- **NSE equities** — Indian markets, NIFTY 50 through NIFTY Total Market universes (30+ pre-registered). Adapter chain: jugaad-data → nselib → yfinance.

Two-layer cache: immutable raw downloads at `data/raw/<provider>/` + canonical-schema SQLite at `data/processed.db`. Repeated calls hit the cache; gaps are detected and backfilled automatically.

### forecasters

Backend-agnostic forecasting surface. A preset is a saved-model artifact (not a config blob) carrying the backend name, tuned hyperparameters, fit metadata, and validation metrics:

```bash
/forecast --preset v24-default --identifier NASDAQ100 \
          --start 2010-01-01 --end 2026-05-01 \
          --origin 2024-01-01 --horizon 60
```

The caller never imports a backend directly or learns its internal parameters. Cross-asset drift between the preset's training data and the forecast target is surfaced as an explicit warning in the result.

### gbdt

Categorical-outcome classifiers that predict P(price-move event within horizon). Each experiment is one (universe, direction, threshold, horizon) tuple:

```yaml
target:
  universe: nifty50
  direction: up
  threshold_pct: 10
  horizon_days: 20
```

The agent is the ML iteration loop: it reads a per-iteration diagnostic bundle (train-vs-val gap, feature importance, calibration curve), decides which features to prune and how to adjust hyperparameters, and writes a human-readable report with its reasoning. Calibration is the headline metric — a well-calibrated model with modest AUC is a v1 success; high AUC without calibration is a failure.

279-column feature pool across 16 families, including cross-sectional rank/z-score features. Walk-forward validation with conditional isotonic calibration gated by Spiegelhalter Z-test.

### backtesting

Multi-asset backtesting engine with a two-phase execution lifecycle that makes look-ahead bias structurally impossible. The engine executes orders and reports fills; strategy logic and performance evaluation are the caller's responsibility.

Six correctness constraints (B1-B6): look-ahead elimination, causal fill, portfolio consistency, deterministic replay, and lot-size integrity.

---

## Repo layout

Every module-specific file is namespaced under `<top>/<module>/`:

```
src/<module>/                  # package code
docs/<module>/                 # design docs, experiment reports
tests/<module>/                # tests
configs/<module>/              # YAML configs
scripts/<module>/              # orchestration and investigation scripts
runs/<module>/<timestamp>/     # per-run artifacts (gitignored)
results/<module>/data/         # aggregated experiment JSONs (checked in)
dashboards/<module>/           # Streamlit views
dashboards/app.py              # thin global launcher (auto-discovers modules)
.claude/skills/<name>/         # agent skill definitions
.claude/memories/              # project-shared context
data/                          # cache (raw + processed, gitignored)
```

Top-level directories are reserved for cross-module concerns.

---

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) for environment management (Python >= 3.12).

```bash
# install all dependencies (analog_mc, data_pipelines, forecasters, gbdt, backtesting — all editable)
uv sync

# run the full test suite (866 tests)
uv run pytest

# run tests for a single module
uv run pytest tests/analog_mc/
uv run pytest tests/data_pipelines/
uv run pytest tests/gbdt/

# run an analog_mc walk-forward on the default NASDAQ100 config
uv run python -m analog_mc walk-forward --config configs/analog_mc/default.yaml

# run a gbdt experiment from a YAML spec
uv run python -m gbdt.experiment configs/gbdt/experiments/nifty50_up_10pct_20d_pilot.yaml

# fetch data via the data_pipelines CLI
uv run python -m data_pipelines fetch NYSE:AAPL --start 2010-01-01

# launch the Streamlit dashboard
uv run streamlit run dashboards/app.py
```

---

## Configuration

Each module is config-driven via YAML files in `configs/<module>/`:

- **analog_mc** — forecast horizon, block length, z-score windows, n_eff candidates, vol-clip bounds, drift mode. See `configs/analog_mc/default.yaml`.
- **data_pipelines** — per-domain universe definitions with ticker lists and adapter chains. See `configs/data_pipelines/domains/`.
- **forecasters** — preset artifacts with backend, hyperparameters, fit metadata, validation metrics. See `configs/forecasters/presets/`.
- **gbdt** — per-experiment YAML specs defining (universe, direction, threshold, horizon) + defaults. See `configs/gbdt/experiments/`.
- **backtesting** — strategy configuration with initial capital, slippage, commission parameters. See `configs/backtesting/examples/`.

---

## Data

- **analog_mc** reads from a local CSV (default `data/NASDAQ100.csv`, FRED-style columns). The loader accepts configurable column names — drop in any CSV with a date and close column.
- **data_pipelines** manages a two-layer cache: raw provider downloads at `data/raw/` and a canonical SQLite store at `data/processed.db`. The `fetch()` function handles everything — gap detection, provider fallback, schema normalization.
- **gbdt** reads pooled multi-stock panels via `data_pipelines.fetch()` from the SQLite cache.
- **forecasters** composes with `data_pipelines.fetch()` automatically when given an `--identifier`.

`data/` is gitignored except for `.gitkeep` markers.

---

## Design principles

- **Diagnostic-gated complexity.** Upgrades are gated on specific diagnostic findings, not on ideas that sound promising. Each candidate addition must (1) fix a diagnostic trigger, (2) pass the full calibration bundle, and (3) not regress existing quality metrics.
- **Causality as a non-negotiable.** Every feature at index t uses only data with index <= t. Zero look-ahead, enforced by tests on synthetic data — the single most important correctness property across all modules.
- **Agent-first API design.** All public functions prefer clean signatures, JSON-serializable outputs, and explicit error types — shapes that wrap cleanly as agent tool calls.
- **Per-module goal docs.** Before editing any module, read its `docs/<module>/goal.md`. It defines what the module is optimizing for and which trade-offs are unacceptable.
- **Plans live on branches.** Each new plan doc gets its own branch; plan + implementation + reports land together as a PR.
- **Determinism.** Same inputs + same seed = bit-identical outputs. No silent reordering, no nondeterministic dispatch.

---

## Project stats

- **~15,300 lines** of source code across 5 modules
- **866 tests** covering correctness constraints, schema invariants, look-ahead detection, and integration
- **82 commits** of iterative, diagnostic-driven development
- **30+ pre-registered universes** across US and Indian equity markets
- **100+ experiment configs** spanning multiple universes, thresholds, and horizons

---

## License

Not yet specified.
