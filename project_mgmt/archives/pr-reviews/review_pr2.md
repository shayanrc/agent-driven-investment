# PR #2 (v1-skills → main) — Review

## Verdict
COMMENT — substantively in good shape; plan-match is solid, anti-goals respected, acceptance demo reproduces, 646/5 test counts match. A handful of low-severity hygiene items (dead imports, one preset-bake hygiene already in V2_TBD) and one medium-severity gap worth landing before/with merge (drift-warning observability + the `_resolve_data_columns` silent-shift).

## Severity summary
- Blocking: 0
- High: 0
- Medium: 2
- Low: 5

## Plan-match check

**Stage 1 (preset schema + loader)** — `src/forecasters/{errors,presets}.py` and `configs/forecasters/presets/v24-default.yaml` match the spec. `validate_preset` enforces every required top-level key, the `name`/filename-stem agreement, `schema_version == 1`, the `sha256:` prefix on `data_hash`, and UTC-only `fitted_at`. `__source_path__` / `__content_hash__` injection is documented and stripped on re-serialization. Done.

**Stage 2 (analog_mc backend adapter)** — `src/analog_mc/forecaster.py` exposes `forecast` and `tune`, plus the new `close_series_from_dataframe` in `src/analog_mc/data.py` (additive, CSV-first path preserved). Both functions populate `warnings` for irregular conditions (one-fold search fallback, horizon/block_length mismatch, padded horizon dates, baked-weights absence). Wire-format-conformant per the contract tests.

**Stage 3 (dispatcher + result validation + data composition)** — `src/forecasters/{data,dispatch}.py`. Dispatch is a 1-entry dict-of-lambdas (anti-goal compliant). Result validation runs BEFORE drift warning injection (so backend can't overwrite drift warnings — comment-documented). `prepare_data` handles both `--identifier` (lazy import of `data_pipelines` domains) and `--data-path` (CSV/parquet with date-slice). Done.

**Stage 4 (forecast-result cache)** — `src/forecasters/cache.py`. Atomic temp+rename, key includes preset name + content hash + identifier/data_path + range + origin + horizon + seed (none-vs-zero distinguished). Tests cover hit/miss, preset-edit invalidation, idempotent concurrent writes. Done.

**Stage 5 (CLI entry point)** — `scripts/forecasters/run.py` ships all four subcommands (`forecast`, `tune`, `fetch`, `list-presets`). `--no-cache` correctly writes to tempdir; `--output-dir` writes to explicit dir; typed errors are caught at `main()` and emitted as `ClassName: msg` to stderr with rc=2. Done.

**Stage 6 (forecaster SKILL.md files)** — Five SKILL.md files match the spec (3 forecasters-owned + 2 data_pipelines-owned). `/tune-preset` bakes in the loop-pattern launch-prompt template + reference to `feedback-experiment-agent-loop.md`. `/forecast` SKILL.md explicitly calls out no `--backend`, no backend-internal flags. Done.

**Stage 7 (integration tests + README)** — `test_forecast_e2e.py` and `test_cross_asset_drift.py` cover the e2e and drift paths; `docs/forecasters/README.md` reads cleanly. Done.

**Stage 8 (data_pipelines skill runner)** — `scripts/data_pipelines/skill_runner.py` ships both subcommands with full identifier/domain/no-args modes and JSON option. `tests/data_pipelines/test_skill_runner.py` covers happy path, unknown-domain, and cached-vs-not-cached. Done.

**Stage 9 (acceptance demo)** — `scripts/forecasters/run_acceptance_demo.py` is a thin 4-phase orchestrator (fetch/tune/forecast/verify); `docs/forecasters/_acceptance_demo.md` is the auto-generated PASS report with headline numbers (CRPS 0.04251 vs RW 0.04370, 90-band 68.33%, 0 drift warnings). `tests/forecasters/test_acceptance_demo.py` smoke-tests the verify+report glue against a fixture. Stage 9's "smallest possible adapter fix" is the NIFTY 500 slug additions in `src/data_pipelines/domains/nse_equities/{config,registry}.py` — six new entries, scoped and minimal. Done.

## Anti-goal compliance

| Anti-goal | Status |
|---|---|
| 1. No `Forecaster` ABC / `ForecasterRegistry` | PASS — `_BACKENDS` is a 1-entry dict-of-lambdas in `dispatch.py:32` |
| 2. No typed `ForecastResult` / `ForecastInput` | PASS — validation is runtime assertions in `_validate_result_contract` |
| 3. `/forecast` does NOT take `--backend` | PASS — `run.py:273-287` shows no `--backend` arg on the forecast subparser |
| 4. No walk-forward exposure on `/forecast` | PASS — only `--origin`/`--horizon` |
| 5. No backend-internal flags (`--n-eff`, `--corr-window`) | PASS — only `--config-overrides path.yaml` |
| 6. Drift never silently absorbed | PASS — `_detect_drift` is injected after backend returns; backend cannot suppress |
| 7. Canonical presets read-only at runtime | PASS — `/tune-preset` writes only to `results/forecasters/presets/` (or `--output-root`) |
| 8. No AI attribution | PASS — `git log main..HEAD` and grep across SKILL.md / docs / src clean |

## C1-C6 compliance

The backend adapter (`src/analog_mc/forecaster.py`) is a thin wrapper — it does not implement any modeling logic, it delegates to existing analog_mc primitives:
- **C1 (causal features):** delegated to `analog_mc.features.compute_features`, unchanged.
- **C5 (strictly forward sampling):** the adapter passes `candidate_idx = np.arange(0, origin_idx)` to `_simulate_forecast`, which goes through `eligible_candidates` (`simulate.py:154`: `forward_ok = candidate_idx + config.block_length < origin_idx`). No re-shuffling.
- **C6 (walk-forward boundary discipline):** the adapter's `_get_weights_and_n_eff` one-fold fallback constructs `train_idx = arange(0, train_end)` and `val_idx = arange(train_end, origin_idx)` — strict left-to-right separation with `train_end = origin_idx - val_size`. No leakage. `tune()` delegates to the existing `run_walk_forward` for multi-fold behavior.

No correctness invariants violated by the adapter.

## Acceptance demo verification

- `_acceptance_demo.md` reports CRPS 0.04251 < RW 0.04370, 90-band 68.33%, 0 drift warnings, 1 total warning (the "padded horizon_dates" warning for a 0-future-date origin) — matches PR description headline numbers exactly.
- `results/forecasters/presets/nifty500-v1.yaml` is a valid v1 preset (backend `analog_mc`, schema_version 1, sha256 data_hash, n_observations 1524, ISO UTC `fitted_at`, baked `weights: [1.0, 0.0, 0.0]` + `n_eff: 80.0`).
- The orchestrator's five `assertions` (`scripts/forecasters/run_acceptance_demo.py:330-336`) line up with goal.md's acceptance criteria: preset_validates, forecast_warnings_empty, coverage_90_in_range, crps_finite, crps_beats_baseline.
- Five documented plan deviations (NIFTY 500 slug addition, `_resolve_data_columns` refactor, fitted_on-range slicing for hash match, 2020 cap on history, Config defaults leaking into baked preset) all present in the diff and either justified inline (1-3) or parked in `V2_TBD.md` (4-5).

## Test adequacy

- **Full suite:** `uv run pytest -q` reports `646 passed, 5 skipped in 271.44s` — matches PR claim exactly.
- `test_forecast_e2e.py` runs the full dispatcher path with slim overrides (100 paths) on NASDAQ100 CSV, checks contract shape (`paths.shape == (100, 60)`), absence of drift warning, finite positive CRPS in `(0, 1)`, and determinism (`r1 == r2` for same seed). It does NOT check "median path within ±X% of V5.A.2 baseline at every step" or "CRPS within ±Y of baseline" as the V1_PLAN Stage 7 description suggests; the slim 100-path config explains the looser bracket but the deviation from plan is not called out in V2_TBD.
- `test_cross_asset_drift.py` does verify the drift warning fires with quantified data_hash mismatch — it asserts both the preset's `fitted_hash` AND the current `data_hash` substrings appear in the warning, then confirms the forecast still produces a valid `(100, 60)` paths array.
- `test_dispatch.py` covers `ResultContractError` for missing top-level keys, paths-not-ndarray, paths.shape mismatch, horizon_dates length mismatch, summary percentile length mismatch, metadata missing keys, and warnings-None. Also covers `UnknownBackendError` and `dispatch_tune` rejecting wrong-backend presets. Strong coverage.

## Findings (detailed)

### [Medium] `_resolve_data_columns` silently uses fallback columns instead of preset hints
**File:** `src/analog_mc/forecaster.py:242-271`
**Observation:** When a preset baked column hints (e.g., `date_col: observation_date`, `close_col: NASDAQ100` — as `nifty500-v1.yaml` currently does), and the input DataFrame happens to also carry the canonical `date`/`adj_close` pair, the preset's hint is silently ignored. The probe order (caller override → canonical → FRED → preset) means a NASDAQ100-baked preset applied to a `data_pipelines` DataFrame just works, but a user inspecting the preset has no way to know their `date_col` field is dead at runtime. V2_TBD.md already calls this out ("`_resolve_data_columns` should warn when a preset's baked date_col/close_col is ignored") but it's not parked behind a release gate.
**Suggested action:** Either land the 3-line warning now (when fallback wins over a present preset hint, `warnings.append("preset's baked date_col/close_col not present in input DataFrame; used <fallback> instead.")`), or move the item from "Backend-adapter cleanups" to a Day-1-after-merge issue with a tracked link.

### [Medium] Cache key does not include a hash of the actual fetched data
**File:** `src/forecasters/cache.py:35-70`
**Observation:** Cache key parts = `(preset_name, preset_content_hash, identifier, data_path, start, end, origin, horizon, seed)`. If the `data_pipelines` cache silently backfills a previously missing day for an identifier (e.g., a yfinance gap-fill arrives between calls), a stale forecast is served from the forecasts cache. V2_TBD.md captures this. The probability is low in practice (identifier+range usually pins the data), but a `(identifier, start, end)` change is what `data_pipelines` users today reach for, so the trust failure mode is plausible. Worth tracking as a v1.1 bug rather than a "nice to have."
**Suggested action:** Promote to a tracked issue with severity, OR add a one-line note in V2_TBD explicitly marking this as a soundness-not-just-hygiene item.

### [Low] Dead `replace` import in `analog_mc/forecaster.py`
**File:** `src/analog_mc/forecaster.py:40`
**Observation:** `from dataclasses import fields, replace` — `replace` is never referenced (AST-confirmed). Only `fields` is used.
**Suggested action:** Drop `replace` from the import.

### [Low] Dead `shutil` and `crps_sample` imports in `run_acceptance_demo.py`
**File:** `scripts/forecasters/run_acceptance_demo.py:35,54`
**Observation:** `import shutil` (line 35) has no references in the file. `from analog_mc.scoring import crps_per_step, crps_sample` (line 54) imports `crps_sample` but only `crps_per_step` is used.
**Suggested action:** Drop both unused names.

### [Low] `nifty500-v1.yaml` carries NASDAQ100-shaped Config defaults
**File:** `results/forecasters/presets/nifty500-v1.yaml:5-8`
**Observation:** `ticker: NASDAQ100`, `data_path: data/NASDAQ100.csv`, `date_col: observation_date`, `close_col: NASDAQ100` are baked into a NIFTY 500 preset because `_build_config` constructs a full `Config` whose defaults are NASDAQ100-shaped and `tune()` rounds-trips `config.to_dict()` into `hyperparameters`. The fields are unused at inference time (forecast() takes a DataFrame, not a CSV path), but they look wrong in a preset for an Indian-equities asset and will confuse anyone reading the file. Already parked in V2_TBD.md.
**Suggested action:** As parked. The filter could be a 5-line list of "inference-time-relevant fields" in `tune()`; consider landing alongside the next preset bake.

### [Low] `_compute_realized` in acceptance demo hardcodes `date` / `adj_close` columns
**File:** `scripts/forecasters/run_acceptance_demo.py:262-280`
**Observation:** Unlike `_resolve_data_columns` in the backend adapter, the verify-phase helper reaches for `df["date"]` and `df["adj_close"]` directly. Fine for NSE-via-data_pipelines (always canonical schema), but couples the demo orchestrator to that contract — if a future identifier returns FRED-style columns the demo will fail with a less helpful `KeyError` than the adapter would.
**Suggested action:** Optional — use the same `_resolve_data_columns`-style probe, or document that the demo is canonical-schema-only.

### [Low] `test_skill_runner.test_health_no_args_json` does not verify `oldest_last_fetch_utc`
**File:** `tests/data_pipelines/test_skill_runner.py:100-106`
**Observation:** The seeded DB has exactly one row with `last_fetch_utc = "2026-05-24T12:00:00Z"`. The test asserts `total_identifiers` and `total_rows` but skips the `oldest_/newest_last_fetch_utc` keys. Easy addition that would catch a regression where the `min/max` is computed off the wrong field. Minor.
**Suggested action:** Add `assert payload["oldest_last_fetch_utc"] == payload["newest_last_fetch_utc"] == "2026-05-24T12:00:00Z"`.

## Recommendation

Land as-is or with the 4 low-severity hygiene items (`replace`/`shutil`/`crps_sample` dead imports + the preset-bake filter) addressed in a tiny pre-merge commit; the two medium items are V2_TBD.md as is reasonable but worth promoting to tracked issues rather than only living in the parking lot. The plan-match, anti-goal compliance, C1-C6 preservation, acceptance demo, and test counts all check out — this is mergeable work.
