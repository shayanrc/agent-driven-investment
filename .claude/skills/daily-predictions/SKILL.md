---
name: daily-predictions
description: Run the daily forward-prediction cadence for the validated sp500 champion models — refresh US market data, incrementally re-score both cells (sp500 +50%/50d and +20%/25d) on the freshest panel, evaluate the SMA200 regime gate, and append the top picks to the committed forward-prediction log. Idempotent, self-gating, and backfilling: safe to run any trading day (or after a multi-day gap) on a machine that isn't always on. Runner: scripts/backtests/daily_forward_predictions.py.
---

# /daily-predictions

One verb: **produce and log today's forward predictions** for the two validated sp500 champions. Composes the data refresh + faithful incremental inference + regime gate + the append-only forward log into a single idempotent cadence (the `_019` forward-OOS node).

## What it does (in order)

1. **Disk pre-flight** — aborts if < 10 G free (FS-wedge guard, per `[[feedback-disk-wedge-pattern]]`).
2. **Seed** the sp500 universe to `--end` (idempotent warm-cache tail fetch; includes ^SPX).
3. **Self-gate + backfill** — reads the last snapshot already in the forward log per model; if the cache hasn't advanced past it, exits as a clean no-op. If the machine was off for several days, it backfills *every* missing trading day, not just today.
4. **Incremental inference** (`infer_fresh_predictions --since`, ~5× cheaper than a full rebuild — a trailing ~7y slice that still covers the test window, so the faithfulness self-check still guards it) for both cells → fresh CSVs (gitignored scratch under `results/backtests/_019_fwd_oos/`).
5. **Regime gate** — SMA200 on the universe index per new date (the deployment-critical overlay: `_017`/`_018`).
6. **Append** the top-10 per (date, model) to `results/backtests/data/forward_predictions_log.csv` — the durable, checked-in record. Idempotent: a `(snapshot_date, model)` already present is never re-appended. **Partial-day guard (#182):** only *complete* trading days (strictly before today) are logged — today's in-progress intraday bar is excluded (in both the pre-gate and the append filter) and logs on the next run after it finalizes.
7. **Optional commit** (`--commit`) — `git add -f` + local commit of the log (no push).

## Usage

This is a **well-under-30-minute foreground run** (two cells, incremental **~2 min each** since the V1.5 feature-build vectorization (#180), + seed), so run it in the **foreground with a `timeout`** — do NOT background it (per `.claude/memories/feedback-sub-agent-foreground.md`; the ≥2 h background+Monitor+ScheduleWakeup pattern does not apply here):

```bash
uv run python -m scripts.backtests.daily_forward_predictions [--commit] [--end YYYY-MM-DD]
# foreground with a guard timeout, e.g. 1800s
```

- `--commit` — commit the appended log locally (recommended for the unattended timer path; omit for an interactive run you want to review before committing).
- `--end` — as-of date (default: today). The pipeline uses data through the latest available trading day ≤ end.

## After an interactive run, report the read

When invoked in a session (not the silent timer), after the run finishes summarize for the user:
- the **regime gate** state (risk-ON ⇒ strategy deploys; risk-OFF ⇒ hold cash — this is the gating verdict, `_016`–`_018`);
- the **equal-weight top-3** per model (the champion's actual positions) + cross-model overlap;
- the honest caveats (modest absolute p / lift-not-certainty; bull-only edge; small effective-N; not investment advice — size as the forward-test it is).

## Notes

- **Unattended automation** (not-always-on machine): a **systemd *user* timer with `Persistent=true`** fires the missed run on next wake/login; the backfill logic makes that correct. Units: `scripts/backtests/systemd/`.
- The full per-(date,ticker) CSVs are **gitignored** (large, exactly regenerable via the same `--since` command); only the compact top-K **forward log** is checked in.
- Tracks only the two validated sp500 cells (`[[project-gbdt-tuning-playbook]]`); adding universes/cells is an edit to `CELLS` in the runner.
