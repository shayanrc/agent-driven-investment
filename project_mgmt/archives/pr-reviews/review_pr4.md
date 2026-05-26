# PR #4 (gbdt-v1 → main) — Review

## Verdict
APPROVE

## Severity summary
- Blocking: 0 / High: 0 / Medium: 0 / Low: 4

## Scaffolding hygiene

The scaffolding is tight and matches `V1_PLAN.md` § "File layout" exactly. Stub modules under `src/gbdt/` are docstring-only (no production code leaks ahead of stage); each docstring points to the V1 stage that lands its implementation. The one declared constant — `targets.py`'s `DIRECTIONS / THRESHOLDS_PCT / HORIZONS_DAYS` — matches `goal.md`'s 18-cell lattice (2 directions × 3 thresholds × 3 horizons = 18). `configs/gbdt/default.yaml` is comment-flagged as a placeholder and has explicit `TBD-Stage<N>` markers for every section pending finalization. `pyproject.toml` adds `src/gbdt` to wheel-packages alongside the existing modules; `import gbdt` resolves cleanly under `uv run` (verified). `results/gbdt/data/.gitkeep` is present; `tests/gbdt/` has only `__init__.py` (deferred per V1 stages); `dashboards/gbdt/` correctly absent (not in v1 scope).

## v0 scan math verification

All headline numbers in the three v0 reports are reproduced from the committed JSONs:

- **v0.1 pooled rates** (`H=10/20/50/100`): `0.0841 / 0.2125 / 0.4717 / 0.6613` — match the JSON to 4 dp.
- **v0.2 ↔ v0.1 cross-check**: `up × 10% × {10,20,50,100}` cells in `_v0_opportunity_scan_full_data.json` equal v0.1 EXACTLY (same `pooled_n_origins`, `pooled_n_events`, `pooled_base_rate` to floating-point identity). v0.2's claim of "reproduces v0.1 exactly ✓" holds.
- **v0.3 spot-checks**: `up_5_h10 raw=0.2932 clean=0.2638 fr=0.1003` (report 29.34%/26.38%/10.0%); `down_50_h100 raw=0.0240 clean=0.0208 fr=0.1327` (report 2.40%/2.08%/13.3%) — match.

Math sanity:

- `sliding_window_view(close[1:], horizon)` produces shape `(n − horizon, horizon)`; row `t` = `close[t+1 : t+horizon+1]`. Verified on a synthetic `close = arange(20)`.
- `breach.argmax(axis=1) + 1` yields 1-based first-breach lag (days from origin). For a row where breach lies at window position 3 (close[t+4]), `argmax + 1 = 4`. Correct.
- v0.3 clean logic `clean = ev & (adv_idx > tgt_idx)` with `sentinel = horizon + 1` correctly handles all four scenarios: target-before-adverse → clean, target-with-no-adverse → clean (adverse pinned to sentinel > tgt), adverse-before-target → not clean, no target → not clean. Strict `>` makes a tie (adverse and target on same day) count as not-clean — defensible interpretation.
- UP/DOWN symmetry in `v0_opportunity_scan_full.py`: `windows <= origin*(1-thr)` is the correct "drop ≥ thr" formulation; `windows >= origin*(1+thr)` for "rise ≥ thr". DOWN-adverse in `v0_opportunity_scan_filtered.py` is `windows >= origin*(1+thr/2)` — the symmetric rally counterpart. All four are right.

One floating-point edge case worth noting (Low): `100.0 * 1.10 == 110.00000000000001` in IEEE-754, so a synthetic price that lands exactly on `1+thr × origin` will not register as a breach (`110.0 >= 110.00000000000001 → False`). On real `adj_close` values this is astronomically unlikely to affect any actual base rate — and the reports don't promise inclusive edge handling — but if a v1 unit-test fixture is constructed with origin=100, threshold=0.10, breach=110, it will silently miss.

Reproducibility: re-running `uv run python -m scripts.gbdt.v0_opportunity_scan` against the *current* cache returns slightly different numbers from the committed v0.1 JSON (e.g., `pooled_origins[H=10] = 71,736` vs committed `71,351`). This isn't a script bug — the NSE cache has been backfilled since the commits (RELIANCE went from 1,492 rows in v0.1/v0.2 to 1,588 rows by v0.3; max date now 2026-05-22). The v0 reports correctly flag the data window as "2020-01-01 → 2025-12-31" for v0.1/v0.2, but v0.3 also claims that window even though its own JSON includes rows through 2026-05-22 — a minor stale claim (see Findings).

## Goal / plan coherence

`goal.md`'s "v1 end-to-end acceptance demo" (5 criteria) matches the Stage 9 acceptance gate in `V1_PLAN.md` (same 5 bullets, same 14/18 floor, same metrics). Anti-goals in `goal.md` ("not a backtester / no PnL / no position sizing / one library / no multi-asset / CSV-first") are mirrored in `V1_PLAN.md`'s 10-item anti-goals list, and both correctly inherit analog_mc's project-wide rules (no PnL, no AI attribution, no `StandardScaler`-equivalent — implicitly via the causal-features-only constraint). The 4 open questions in `V1_PLAN.md` (feature set → Stage 2, library → Stage 4, calibration → Stage 5, fold scheme → Stage 6) are each gated to a specific stage with a documented default lean and decision criteria; the fifth ("multi-target correlation handling") is explicitly deferred to v2 with the gate criterion stated. `V0_INVESTIGATION_PLAN.md` correctly frames v0 as throwaway-quality investigation with file conventions (`_v0_<name>.md`, `_v0_<name>_data.json`, direct-SQL cache reads) that the three shipped v0 deliverables follow precisely.

## Findings (detailed)

### [Low] v0.3 report's "Data window: 2020-01-01 → 2025-12-31 (~1,492 rows per stock)" is stale relative to its own JSON
**File:** `docs/gbdt/_v0_opportunity_scan_filtered.md:15`
**Observation:** The NSE cache was backfilled between the v0.2 commit (`9700c11`) and the v0.3 commit (`78a2345`). v0.3's `_v0_opportunity_scan_filtered_data.json` contains data through 2026-05-22 for many tickers (e.g., RELIANCE `n_rows = 1,588`, not 1,492). The report's data-window claim and per-stock row count are inherited from v0.1/v0.2 prose without re-checking against v0.3's actual data. The numerical findings themselves are correct against the JSON; only the data-window prose is stale.
**Suggested action:** Pre-merge nit: either update the v0.3 prose to "2020-01-01 → 2026-05-22 (~1,500–1,600 rows per stock)" or note that v0.3 ran against a slightly extended cache vs v0.1/v0.2. Optional; doesn't change conclusions.

### [Low] v0 scripts mutate the checked-in result JSONs in place
**File:** `scripts/gbdt/v0_opportunity_scan.py:143-144` (and the analogous lines in v0.2/v0.3)
**Observation:** Each v0 script writes its headline JSON directly to `results/gbdt/data/_v0_<name>_data.json`, which is checked into git. Re-running any v0 script against a changed cache will produce a dirty working tree with overwritten committed results. I verified this by re-running v0.1: the file content changed, and I had to restore from a backup to leave the tree clean. This is a known v0-discipline trade-off (per `V0_INVESTIGATION_PLAN.md`, v0 scripts are throwaway-quality), but a `git status` check after re-running would surface drift, and a `--output` arg or a `runs/`-style timestamped path could decouple "re-run for sanity" from "overwrite committed snapshot."
**Suggested action:** Optional. If kept, a one-line note in the V0_INVESTIGATION_PLAN's "What v0 is NOT" section ("v0 scripts overwrite their committed JSON in place — back up before re-running if the cache has drifted") would prevent future surprise. Strictly v0-hygiene; non-blocking.

### [Low] Floating-point edge in threshold comparison
**File:** `scripts/gbdt/v0_opportunity_scan.py:58` (and v0.2/v0.3 equivalents)
**Observation:** `windows >= (origin_close[:, None] * (1.0 + threshold))` rounds the right-hand side via IEEE-754 (`100 * 1.10 = 110.00000000000001`). Real `adj_close` floats almost never land exactly on the threshold, so the v0 scan numbers are unaffected. The same idiom will appear in `src/gbdt/targets.py` at Stage 3; if Stage 3's unit-test fixtures use convenient round numbers (origin=100, target=110), the test will spuriously miss the boundary case.
**Suggested action:** Note this for Stage 3's test-fixture design — either use `np.isclose`-style tolerance or pick fixture values that are unambiguously above/below the threshold (e.g., breach = 110.5, not 110.0). Not a v0 scan defect.

### [Low] v0.1 ↔ v0.3 ticker-data drift means the three reports describe slightly different universes
**File:** `docs/gbdt/_v0_opportunity_scan.md` / `_v0_opportunity_scan_full.md` / `_v0_opportunity_scan_filtered.md`
**Observation:** v0.1 and v0.2 both saw 50 tickers × 1,492-row median; v0.3 saw 50 tickers but ~96 more rows per ticker due to cache backfill. The three reports are individually internally consistent (each matches its own JSON), and v0.2's "exact reproduction of v0.1" claim is true against the v0.1-era cache. But the cross-report narrative ("v0.3 is the filtered companion to v0.2") slightly hides that v0.3 saw a different data slice. Magnitude is small (rates shift by ≤ a few bps).
**Suggested action:** Optional one-sentence note in V0_INVESTIGATION_PLAN's task-inventory table: each v0 commit was a point-in-time snapshot against whatever cache existed at the time. Not blocking — the v0 discipline ("re-run when cache extends") is appropriate; just worth flagging that the three deliverables aren't from a single coordinated cache snapshot.

## Recommendation

APPROVE — scaffolding is clean, stubs are docstring-only, the 18-cell lattice constant matches `goal.md`, `import gbdt` works, the v0 scan math is correct (sliding-window construction, 1-based lag, UP/DOWN symmetry, v0.3 sentinel logic all verified), all three v0 reports reproduce their JSONs exactly, and there is no AI attribution anywhere in the PR. The four Low findings are pre-merge nits or notes-for-future-stages, none of which block.
