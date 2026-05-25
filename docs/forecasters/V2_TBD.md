# forecasters — v1.1 / v2 parking lot

Items noticed during the v1 build that are out of scope for the v1 PR but
warrant a follow-up.

## Backend-adapter cleanups (analog_mc)

- **`tune()` produces a preset with `ticker: NASDAQ100` / `data_path: data/NASDAQ100.csv` baked in** because `_build_config()` returns a full `Config` dataclass whose defaults are NASDAQ100-shaped. These fields are unused by `forecast()` (which uses the input DataFrame), but they look wrong in a preset for a different asset. Fix: filter Config defaults from the baked preset to only those fields that genuinely matter at inference time (zscore_horizons, ewma_halflife, vol_clip_*, drift_mode, momentum_lookback, momentum_shrinkage, conditional_block_sampling, matcher_distance, corrwindow_length, vol_model, local_linear_correction, plus the baked weights + n_eff). Anything else is search-time only and shouldn't pollute the preset.
- **`_resolve_data_columns` should warn when a preset's baked date_col/close_col is ignored** in favor of the canonical/FRED fallback. The current behavior silently shifts (the probe order is caller-override → canonical `date`/`adj_close` → FRED `observation_date`/`NASDAQ100` → preset hint, so a preset baked with `date_col: observation_date / close_col: NASDAQ100` applied to a `data_pipelines` canonical DataFrame "just works" via the canonical branch, and the preset hint becomes dead at runtime). A `warnings.append("preset's baked date_col/close_col not present in input DataFrame; used <fallback> instead.")` when the fallback wins over a present preset hint would make it diagnostic. Worth landing as the next pre-v1.1 hygiene commit.

## NSE data layer

- **NIFTY 500 yfinance fallback is noisy.** Every backfill of an internal NSE holiday gap (Holi, Diwali, ad-hoc closures) results in a yfinance "possibly delisted" log line because nselib's `index_data` requires `from_date != to_date` and yfinance is asked for a single-day range that has no data. Cosmetic, not functional, but floods the orchestrator's log. Two options: (a) suppress the yfinance fallback for single-day gaps; (b) treat single-day all-empty gaps as `EmptyPayload` upstream of yfinance so it never gets asked. Belongs in `data_pipelines/domains/nse_equities/dispatch` style logic, not forecasters.
- **NIFTY 500 cache only reaches 2020** because the existing `data/processed.db` was seeded with that range. The v1 acceptance demo uses 2020-2026; the goal.md text suggests "the deepest history the adapter chain reaches" (~2005 per the pre-flight). A follow-up should extend the canonical cache back to 2005-2010 to unlock a more demanding tune (longer fit window → more folds → tighter validation).

## Framework

- **Forecast cache invalidation on data drift.** *Soundness item (not just hygiene).* Today the cache key includes `(preset, identifier, start, end, origin, horizon, seed)` but NOT a hash of the actual fetched data. If the underlying `data_pipelines` cache changes (e.g., a backfill adds a previously missing day for the same `(identifier, start, end)`), a stale forecast cache entry is served. Probability is low in practice since the cache key plus the start/end range usually pins the data, but the failure mode is silent and incorrect, not just suboptimal — promote to a tracked v1.1 bug rather than a "nice to have." Fix: add a `data_hash`-of-input column in the cache key (hash the fetched-and-sliced DataFrame's `(date, adj_close)` tuples).
- **`/list-presets` could surface `data_hash` mismatch potential** — e.g., a `compatible_with_current_data` column showing whether each preset's `fitted_on.data_hash` matches what `data_pipelines` would currently return for the same identifier+range. Useful when many presets exist.

## Tests

- **Live acceptance demo is wrapped behind `--phase` flags** but not exposed in CI. The fixture-based smoke (`test_acceptance_demo.py`) covers the orchestrator's verify/report glue but not the live tune. That's the right CI scope, but documenting the manual pre-merge invocation in `goal.md` would make the contract clearer.

## Acceptance demo

- **`_compute_realized` in `scripts/forecasters/run_acceptance_demo.py` is canonical-schema-only.** Documented in the helper's docstring (it reaches for `df["date"]` / `df["adj_close"]` directly, fine for any `data_pipelines`-sourced identifier). A future demo targeting a FRED-style or CSV-style identifier would need either the backend adapter's `_resolve_data_columns`-style probe or an explicit upstream column rename. Promote when a second-asset acceptance demo lands.

## When to promote to a v1.1 plan

Most of these are individually small. A coherent slice — "v1.1: clean-up + extend NIFTY 500 history" — would be promotable when (a) the noisy yfinance backfill becomes a real operational problem, or (b) someone tunes a second backend (ARIMA, GARCH) and the preset-bake hygiene matters more.
