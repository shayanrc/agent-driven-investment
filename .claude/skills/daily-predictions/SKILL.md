---
name: daily-predictions
description: Run the daily forward-prediction cadence — refresh US market data, incrementally re-score the tracked gbdt cells (the two deployed sp500 champions +50%/50d & +20%/25d, plus three higher-ranked comparison candidates across sp500/russell1000/nasdaq100) on the freshest panel, evaluate the per-universe SMA200 regime gate, and append the top picks to the committed forward-prediction log. Idempotent, self-gating, and backfilling: safe to run any trading day (or after a multi-day gap) on a machine that isn't always on. Runner: scripts/backtests/daily_forward_predictions.py.
---

# /daily-predictions

One verb: **produce and log today's forward predictions** for the tracked cells — the two deployed sp500 champions (`deployed=True`) plus three higher-ranked registry candidates tracked for comparison (`deployed=False`). Composes the data refresh + faithful incremental inference + regime gate + the append-only forward log into a single idempotent cadence (the `_019` forward-OOS node).

## What it does (in order)

1. **Disk pre-flight** — aborts if < 10 G free (FS-wedge guard, per `[[feedback-disk-wedge-pattern]]`).
2. **Seed** each tracked universe (sp500 / russell1000 / nasdaq100) to `--end` (idempotent warm-cache tail fetch; includes ^SPX / ^NDX). `--no-seed` skips this and scores cache-only (for backfills over already-cached history, or offline).
3. **Self-gate + backfill** — reads the last snapshot already in the forward log per model; if the cache hasn't advanced past it, exits as a clean no-op. If the machine was off for several days, it backfills *every* missing trading day, not just today.
4. **Incremental inference** (`infer_fresh_predictions --since`, ~5× cheaper than a full rebuild — a trailing ~7y slice that still covers the test window, so the faithfulness self-check still guards it) for each cell, one shared feature build per universe → fresh CSVs (gitignored scratch under `results/backtests/_019_fwd_oos/`).
5. **Regime gate** — SMA200 on the cell's universe index per new date (^SPX for sp500/russell1000, ^NDX for nasdaq100; the deployment-critical overlay: `_017`/`_018`).
6. **Append** the top-10 per (date, model) to `results/backtests/data/forward_predictions_log.csv` — the durable, checked-in record. Idempotent: a `(snapshot_date, model)` already present is never re-appended. **Partial-day guard (#182):** only *complete* trading days (strictly before today) are logged — today's in-progress intraday bar is excluded (in both the pre-gate and the append filter) and logs on the next run after it finalizes.
7. **Optional commit** (`--commit`) — `git add -f` + local commit of the log (no push).

## Usage

This is a **foreground run under ~30 minutes** (five cells, one shared incremental build per universe — sp500/nasdaq ~2–3 min each; the russell1000 build is heavier because its candidate test windows force a ~2016 warmup anchor — plus the seed), so run it in the **foreground with a `timeout`** — do NOT background it (per `.claude/memories/feedback-sub-agent-foreground.md`; the ≥2 h background+Monitor+ScheduleWakeup pattern does not apply here):

```bash
uv run python -m scripts.backtests.daily_forward_predictions [--commit] [--end YYYY-MM-DD]
# foreground with a guard timeout, e.g. 1800s
```

- `--commit` — commit the appended log locally (recommended for the unattended timer path; omit for an interactive run you want to review before committing).
- `--end` — as-of date (default: today). The pipeline uses data through the latest available trading day ≤ end.

## After an interactive run, report the read

When invoked in a session (not the silent timer), after the run finishes summarize for the user:
- the **regime gate** state (risk-ON ⇒ strategy deploys; risk-OFF ⇒ hold cash — this is the gating verdict, `_016`–`_018`);
- the **picks as a table** — columns **Ticker | Model/Rank | p**, **top 3 per model**; lead with the two **deployed** champions (6 rows — the gating read), then the three **candidate** cells (9 rows) in a clearly-separated block marked `deployed=False`. `Model/Rank` is one combined column, e.g. `sp500_50 / 1`; raw `p`, not lift. See `[[feedback-predictions-table-format]]`. Note cross-model overlap in prose.
- the honest caveats (modest absolute p / lift-not-certainty; bull-only edge; small effective-N; not investment advice — size as the forward-test it is).

## Notes

- **Unattended automation** (not-always-on machine): a **systemd *user* timer with `Persistent=true`** fires the missed run on next wake/login; the backfill logic makes that correct. Units: `scripts/backtests/systemd/`.
- The full per-(date,ticker) CSVs are **gitignored** (large, exactly regenerable via the same `--since` command); only the compact top-K **forward log** is checked in.
- Tracks **five cells**: the two deployed sp500 champions (`deployed=True`, `[[project-gbdt-tuning-playbook]]`) + three higher-ranked registry candidates (`deployed=False`) across sp500/russell1000/nasdaq100, backfilled from 2026-04-01 for comparison — NOT promoted (a champion swap is a separate decision; see `_019`). The `deployed` column distinguishes them in the log. Adding/removing cells is an edit to `CELLS` in the runner.
