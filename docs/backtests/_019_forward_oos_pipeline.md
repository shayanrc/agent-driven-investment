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

Log schema: `snapshot_date, model, threshold_pct, horizon_days, rank, ticker,
p_calibrated, base_rate, regime_on, spx_close, spx_sma200, logged_at`.

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
uv run python -m scripts.backtests.daily_forward_predictions [--commit]
# or the skill:  /daily-predictions
# unattended (not-always-on): systemd user timer — see scripts/backtests/systemd/README.md
```

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
