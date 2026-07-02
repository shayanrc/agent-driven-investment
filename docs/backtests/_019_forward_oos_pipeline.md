# _019: the forward-OOS pipeline — a daily, idempotent prediction cadence

## TL;DR

`_015`–`_018` left exactly one genuinely-open item that isn't a new modelling
question: **forward OOS** — re-score the validated sp500 champions as the cache ages
and keep a dated record. This memo turns that from a manual chore into an **automated
daily cadence**: refresh US data → incrementally re-score both sp500 cells → evaluate
the regime gate → append the top picks to a committed, append-only **forward log**.

Three pieces, each verified:

1. **Incremental inference (`infer_fresh_predictions --since`)** — builds features from a
   trailing ~7-year slice instead of the full 1990→ panel. **Proven bit-identical to the
   full build** (max|Δp_raw| = **0.000e+00** on the current cache), ~3× faster (~6 min/cell
   vs ~18). Safe because every feature is a bounded-lookback rolling stat (≤200d window;
   F16 streaks reset when z re-enters the band), the slice is kept ≥ the 1600-row
   eligibility floor (so the cross-sectional ticker set matches the full build), and the
   slice still covers the model's test window — so the existing **<1e-4 self-check still
   guards faithfulness on every run** (an insufficient warmup or changed ticker set would
   make the test reproduction diverge and abort).
2. **The pipeline (`scripts/backtests/daily_forward_predictions.py`)** — idempotent,
   self-gating, **backfilling**. Disk pre-flight → seed sp500 → cheap stock-bellwether
   pre-gate (skip inference if the *stock* panel hasn't advanced — the index EOD leads the
   constituent bars intraday, so gating on ^SPX would falsely trigger) → incremental infer
   → regime gate → append top-10/day/model → optional local commit.
3. **The interface** — a `/daily-predictions` **skill** (your manual/agent trigger; gives
   the regime read + picks interactively) and a **systemd *user* timer** with
   `Persistent=true` for the not-always-on machine (fires the missed run on next
   wake/login; the backfill makes a multi-day gap correct).

## What gets committed (and what doesn't)

| Artifact | Location | Tracked? | Why |
|---|---|---|---|
| **Forward log** (top-10/day/model + regime state) | `results/backtests/data/forward_predictions_log.csv` | **yes** (force-added) | small, append-only, the durable record to score against realized outcomes |
| Full per-(date,ticker) CSVs | `results/backtests/_019_fwd_oos/*_fresh.csv` | no (gitignored) | large, **exactly** regenerable via the same `--since` command |

Log schema (**v2, unified — 16 cols**): `snapshot_date, cell, model, universe,
threshold_pct, horizon_days, rank, ticker, p_calibrated, base_rate, gate_index,
gate_close, gate_sma200, regime_on, deployed, logged_at`. The gate columns are
**universe-aware** (`gate_index` = `^SPX` for sp500/russell1000, `^NDX` for
nasdaq100; renamed from the sp500-only `spx_close`/`spx_sma200`). `cell` is the full
gbdt experiment name (join key → `backtest_summary.csv` / `r_precision_at_k.csv`).
`deployed`=True marks the two champions wired into `/daily-predictions`; `deployed`=False
marks higher-ranked registry cells tracked **for comparison only** (mirrors the registry
`daily_preds` flag). `p_calibrated` is the model's **native** probability (the inference
path returns `p_raw`; isotonic recalibration is monotonic, so top-K ranks are unaffected).
Migrated in place by `scripts/backtests/migrate_forward_log_v2.py` (value-preserving for
the champion rows).

## Initial log (backfilled from each cell's OOS start → latest settled day)

- **sp500_50** (+50%/50d): from **2026-03-13** (65 days); **sp500_20** (+20%/25d): from
  **2026-04-20** (40 days) — each model's genuine OOS begins after its own `test_end`.
- Rows: **1050** (top-10 per model per trading day), latest settled snapshot **2026-06-15**.
- **Regime captured per day** — the log already records a **risk-OFF stretch** for sp500_50
  in spring 2026 (SPX briefly below its 200d SMA), the kind of day the gate (`_017`/`_018`)
  says to sit in cash; the 2026-06-15 snapshot is risk-ON.
- Latest equal-weight top-3 (the champion's positions, 2026-06-15) — **WDC tops both**:
  sp500_50 = **WDC 0.278, GLW 0.225, INTC 0.188**; sp500_20 = **WDC 0.398, CIEN 0.367,
  COHR 0.354** (a tight semis / optical / storage cluster).

## Candidate cells tracked for comparison (v2, added 2026-06-26)

Beyond the two deployed champions, the cadence now also scores the **three registry cells
that outrank them** on back-test total return (`backtest_summary.csv`), tracked
**`deployed=False`** (NOT promoted — a champion swap is a separate decision):

| model | cell | universe | gate | target |
|---|---|---|---|---|
| `russell_50_200` | `russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1` | russell1000 | ^SPX (proxy) | +50% / 200d |
| `nasdaq_40_50` | `nasdaq100_up_40pct_50d_dd20pct_agentloop_mix` | nasdaq100 | ^NDX | +40% / 50d |
| `russell_40_100` | `russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1` | russell1000 | ^SPX (proxy) | +40% / 100d |

- **Backfilled from 2026-04-01** (the `backfill_from` floor) → 54 settled days each through
  2026-06-23 (540 rows/cell), matching the champions' visible window for a clean side-by-side.
  April is **genuinely OOS** for all three — their gbdt `test_end`s are 2024-10 / 2026-03 /
  2025-05, all before April (the registry's `oos_*` columns are the *back-test* window, NOT
  the model's test boundary — don't confuse them).
- **Extended to 2026-01-01 (2026-06-27)** — the `backfill_from` floor moved April → January.
  January is genuine OOS for the two **russell** cells (test_ends 2024-10 / 2025-05), which now
  span **2026-01-01→06-25**. `nasdaq_40_50` (test_end **2026-03-12**) has January *inside* its
  test window, so its floor clamps to its **2026-03-13** OOS start (the runner caps
  `since = max(test_end, backfill_from − 1d)`, never logging in-sample dates). The two sp500
  champions likewise stay at their test_ends (2026-03-13 / 04-18) — January is in-sample for them.
- **Leaderboard caveat**: "outrank the champions" is partly a window artifact — those
  back-test returns span 1–3 years vs the champions' ~6 months. This forward log is the
  apples-to-apples record that will settle it as OOS history accrues.
- **Warmup fix**: faithfully reproducing an *old* test window needs the warmup anchored to
  the **earliest test_start per universe** (not to `since`) — a too-shallow trailing slice
  shifts long-lookback / cross-sectional features and `self_check` aborts (caught + fixed
  during this backfill). russell1000 now warms from ~2016-03 — a heavier daily build, shared
  across both russell cells.
- **2026-06-26 refresh** — seeded all three universes → 06-26 and extended the log to the
  latest settled day (**2026-06-25**; 06-26 is an in-progress partial, excluded by the #182
  guard). The first candidate backfill had run cache-only while russell1000 (~50%) / nasdaq100
  were stale at 05-22, so the candidates were re-seeded (back to 05-20) and re-scored fresh.
  Two candidate-only coverage quirks surfaced here — the candidates dropped **2026-05-26/27/28**
  entirely and scored only ~470 of ~890 eligible tickers on **06-01→06-08** — and were then
  **root-caused and fixed** (below).
- **`_align_panel` forward-row fix (2026-06-26)** — the coverage quirks above were a bug in the
  inference path's training-panel alignment, NOT missing data. `_align_panel` drops historical
  provider gap-fill rows the model never trained on, using `snap` = the matched cached
  feature-matrix's **max date** as the cutoff. But `_training_panel_index` had matched a
  **forward-extending regen** (a russell1000 matrix ending **2026-06-18**, built while ~half the
  universe was stale at 05-22), so `snap` sat in the OOS period and `_align_panel` pruned every
  freshly-seeded forward row `≤ 06-18` not in that stale row-set — collapsing 05-26→06-18 to
  ~471/890 (and 05-26/27/28 to 0, since the regen skipped them). Fix: **cap the cutoff at the
  cell's `test_end`** (`snap = min(snap, test_end)`) so alignment only prunes historical rows and
  **never drops genuine OOS rows**. After the fix the candidates score the full universe across
  the whole forward window with no date gaps (890 on every day incl. 05-26). The deployed sp500
  champions were unaffected (sp500 was never stale, so its regen carried full coverage) and their
  self-check still passes — the cap can't change the `≤ test_end` reproduction window.

## Verification

- **Incremental == full**: rebuilt sp500_50 both ways on the identical current cache →
  max|Δp_raw| = 0.000e+00 across 31,590 rows (a transient 6e-3 vs the older `_019` CSVs was
  pure data staleness — that CSV predated a provider revision of the 06-15 bar).
- **Idempotency**: a second immediate run logs nothing (`[done] no new snapshots — no-op`).
- **Self-gating**: a stock-bellwether (AAPL/MSFT/JPM) pre-gate skips the ~6 min/cell
  inference when the stock panel hasn't advanced past the last log.
- **Settling discipline**: only complete trading days enter the log; today's in-progress bar
  is never logged. NOTE (2026-06-17): the original "0 rows for today" was *partly* a cache-read
  off-by-one (a string compare on the `…00:00:00` date column dropped the `end` day), since
  fixed in **#182** — which also adds an explicit partial-day guard so the cadence logs the
  latest *complete* day and skips only today's in-progress bar. See the **Update** below.
- **Self-check**: every incremental run reproduces `test.csv` to <1e-4 (faithful) or aborts.

## How to run

```bash
# manual / interactive (gives the regime read + picks):
uv run python -m scripts.backtests.daily_forward_predictions [--deployed-only] [--commit]
# or the skill:  /daily-predictions
# unattended (not-always-on): TWO systemd user timers — a DAILY --deployed-only run
#   (deployed sp500 champions only, ~14 min) + a WEEKLY full 5-cell run (candidate
#   refresh). See scripts/backtests/systemd/README.md
```

**Cadence tiering (V1.6 seed track, 2026-07-02):** the full 5-cell run is ~30–60 min — ~half of it the sequential ~1,006-ticker russell1000 seed + its two ~10-min builds, all for the non-deployed comparison candidates. `--deployed-only` runs just the two deployed sp500 champions (one sp500 build, sp500-only seed) → **~14 min daily**; a weekly systemd unit runs the full cadence to refresh the candidates (backfilled). Independently, the V1.6 feature-build refactor (F16-meta 93→1 pass + native `groupby.rolling`/`shift`, both bit-identical) cut the sp500 build 279 s → 209 s. See `docs/gbdt/V1.6_incremental_feature_cache_plan.md`.

**View the log** — `streamlit run dashboards/backtests/app.py`: a read-only dashboard over the forward log. **Predictions** tab = one big-card panel per model (ticker large, colour-coded by **lift** = p ÷ base rate, with the company name + `target`/`stop` price levels beneath), then a cross-model **consensus** panel (names clearing the ≥3/5 majority), with the full all-picks and vote-detail tables collapsible below; the SMA200 **regime gate** lives in the **sidebar** (alongside the date / top-K / consensus-pool controls). **Backtests** tab = an **independent return simulation** of the logged picks (no backtest engine): a strategy selector (consensus vote, or a model's rank-1) + a start-date picker, with target / stop / max-alloc / horizon controls and a regime-ON toggle — each EOD signal is filled at the **next trading day's open** (no look-ahead), then exits at the first of +target / −stop / horizon (close-based, equal max-alloc sizing, gross of costs), rendered `plot_actions`-style (strategy equity vs the universe-index buy-hold with ticker-labelled buy/sell markers) with summary stats + a trades table; the benchmark is anchored to the chosen start date so it's identical across SPX strategies. No inference — reads the committed log; target/stoploss prices come from the `us_equities_data` cache and company names from the `us_equities_names` table (below), both cache-only. (Tabs were renamed from Snapshot / History.)

**Company names** — a dedicated **additive** `us_equities_names(ticker, name, updated_utc)` table in `processed.db` (the us_equities cache itself stores no names — its meta table is fetch bookkeeping only). `scripts/backtests/fetch_ticker_names.py` populates/refreshes it via yfinance for the log's distinct tickers: incremental (only-missing by default; `--refresh-all`, `--stale-days N`), bounded retries, and **never overwrites a good name** on a failed fetch. Additive ⇒ `data_pipelines`' seed/upsert never touches it (survives re-seeds); the dashboard reads it read-only and degrades to bare tickers if it's empty (e.g. a fresh checkout before the refresh has run).

## Caveats / scope

- **~6 min/cell** even for a 1-day increment (the slice rebuild is fixed-cost). The further
  win is the **parallel/tail-only feature build** (gbdt `GBDTPERF`, `V1.4_TBD` row 3) — the
  incremental slice already cut ~18→6 min; parallelizing the per-ticker build would cut it
  again. Acceptable for a daily cadence as-is.
- Tracks only the **two validated sp500 cells**; adding universes/cells = edit `CELLS` in
  the runner. (r1k stays on the ^SPX proxy — ^RUI remains uncached.)
- The bellwether pre-gate now gates on `date < today` (**#182**), so it skips inference when
  there is no new *complete* day — the earlier intraday over-trigger (build, then emit 0 rows
  for today's partial bar) is resolved. Per-cell incremental cost is now **~2 min** (V1.5
  vectorization, **#180**), down from ~6.
- The log is the **signal record**, not a backtest — joining it against realized
  forward outcomes (to score live hit-rates / rolling excess) is the natural follow-up once
  enough days accrue.
- This is forward *evidence accrual*, not a new edge: the strategy, sizing, and regime gate
  are settled (`_005`–`_018`). The log grows the honest effective-N over wall-clock time.

## Update (2026-06-17) — off-by-one fix, V1.5 speedup, first complete-day entry

- **Cache end-date off-by-one fixed (#182).** Cached dates carry a `…00:00:00` time
  component, so `_cache_read`'s `date <= end` *string*-dropped the `end` day — every caller
  (this cadence included) silently lost the most recent day. Fixed with a half-open interval
  `[start_day, end_day + 1)`; regression test added (`tests/gbdt/test_data_loader.py`). A
  **partial-day guard** (only log days `< today`, in both the pre-gate and the append filter)
  now makes the partial-bar exclusion explicit rather than an accident of the bug.
- **Feature build vectorized 3.84× (V1.5, #180).** The incremental re-score is now ~2 min/cell
  (was ~6), bit-identical — so the "~6 min/cell" figures above are superseded.
- **First real complete-day entry logged (#183):** 2026-06-16 (regime risk-ON; top-3
  WDC/GLW/COHR + SMCI/CIEN/COHR), log → 1070 rows. Verified end-to-end after the fix:
  inference emits the latest complete day, the guard skips today's in-progress bar.

## Reproducibility

- Branch `backtests-v23-daily-forward-pipeline`. New: `daily_forward_predictions.py`,
  `infer_fresh_predictions.py` `--since` incremental mode, `.claude/skills/daily-predictions/`,
  `scripts/backtests/systemd/`. Forward log under `results/backtests/data/`; full CSVs
  gitignored. Graph + `docs/gbdt/V1.4_TBD.md` (GBDTPERF) updated.

## Predicted-probability / early-calibration readout (2026-06-27)

After the January backfill the log holds enough OOS history to ask the first real question of
the predictions themselves: **do the predicted probabilities separate realized winners from
losers?** This is an *interim shape* read, NOT a settled calibration verdict — the long
horizons are mostly unresolved and the window is a strong bull. Script (regenerable as horizons
mature): `scripts/backtests/analyze_forward_prob.py`. Figures: `results/backtests/_019_fwd_oos/figs/_019_prob_*.png`.

**Method.** Replicate the trained triple-barrier label exactly (`gbdt.targets.build_target`,
path-honesty / CLOSE-based: target = first close ≥ `+threshold%`; stop = close ≤ `−max_drawdown%`;
first-touch wins) and bucket every logged pick by realized outcome: **TARGET** / **STOP** (locked
barriers), **ENDED_POS/NEG** (full horizon elapsed, no barrier), **CURRENT_POS/NEG** (horizon not
yet elapsed — open, marked to last close). `max_drawdown` parsed from the cell name; the cache
`close` is the same series the label sees, so the classification is faithful by construction.

**Findings.**
1. **The log persists top-10/day** (the "top 3" is only the daily on-screen readout); this analysis uses all ten.
2. **Absolute `p` tracks the horizon, not skill** — `russell_50_200` (+50%/200d) sits ~0.63 vs
   `sp500_50` (+50%/50d) ~0.12 for the *same* +50% target. Compare elevation over base rate only:
   top-1 lift is sp500_50 ≈7.6×, russell_50_200 ≈5.1×, nasdaq ≈3.9×, russell_40_100 ≈3.8×, sp500_20 ≈3.2×.
3. **TGT-vs-STOP separation** (rank-AUC = P(a random target pick has higher `p` than a random
   stop pick); 0.5 = none, >0.5 = correct, <0.5 = inverted):

   | model | top-10 AUC | MW p | top-3 AUC | MW p |
   |---|--:|--:|--:|--:|
   | sp500_50 | **0.76** | <0.001 | 0.39 | 0.418 |
   | sp500_20 | 0.41 | 0.008 | 0.21 | <0.001 |
   | russell_50_200 | 0.36 | <0.001 | 0.40 | 0.022 |
   | russell_40_100 | 0.49 | 0.665 | 0.47 | 0.540 |
   | nasdaq_40_50 | 0.38 | <0.001 | 0.59 | 0.066 |

   On **top-10, `sp500_50` is the only clear separator** (AUC 0.76); the others are significantly
   **inverted** (sp500_20 / russell_50_200 / nasdaq, AUC 0.36–0.41) or **flat** (russell_40_100, n.s.).
4. **…but it is rank-driven and collapses at top-3.** `sp500_50`'s separation vanishes among its
   three highest-conviction picks (AUC 0.39, n.s., only **5** stop-outs). At the top-3 tier **no model**
   cleanly separates target from stop (nasdaq is weakly positive but not significant). The top-10
   edge is a rank-spread effect (lower-ranked = lower-`p` picks stop more), not a property of the sharpest picks.
5. **Volatility confound.** Where `p` co-moves with outcome it does so *inverted* — higher `p` →
   more stops — because `p` tracks momentum/volatility and high-vol names tag both barriers / overshoot
   the floor. `sp500_20` is monotonically inverted (TGT 0.171 < STOP 0.194 < CUR± ≈ 0.27).
6. **`russell_40_100` is non-discriminative** — `p` is compressed (~0.28, std 0.015 across the full
   top-10) and flat across all six buckets at every tier.

**Caveats (these bound the read hard).** CURRENT± (open, marked-to-last) dominate; `russell_50_200`'s
200d horizon has **zero** settled ENDED±. Settled barrier samples are tiny at top-3 (STOP=5 for sp500_50).
The strong-bull window inflates target-hits far above training base rates (sp500_50 realized 72% vs
base 2.6%), leaving `p` little room to discriminate; fast movers also resolve first (early-resolution
bias). True calibration (do `p≈0.15` picks hit ~15%?) is **not yet assessable** — re-run as horizons
mature across a mixed (non-bull) regime.

**Verdict.** In this immature, bull-dominated, mostly-still-open window the predicted probabilities
do **not** rank realized success at the conviction tier that matters (top-3); `sp500_50`'s top-10
separation is a rank-spread artifact that does not survive at top-3. **No promotion/demotion implication
yet** — this is a baseline to re-measure as the horizons settle.

Figures: [overview](../../results/backtests/_019_fwd_oos/figs/_019_prob_dist_overview.png) ·
top-10 [box](../../results/backtests/_019_fwd_oos/figs/_019_prob_outcome6_top10_box.png) /
[violin](../../results/backtests/_019_fwd_oos/figs/_019_prob_outcome6_top10_violin.png) ·
top-3 [box](../../results/backtests/_019_fwd_oos/figs/_019_prob_outcome6_top3_box.png) /
[violin](../../results/backtests/_019_fwd_oos/figs/_019_prob_outcome6_top3_violin.png).
Reproduce: `uv run python -m scripts.backtests.analyze_forward_prob --as-of 2026-06-27`.
