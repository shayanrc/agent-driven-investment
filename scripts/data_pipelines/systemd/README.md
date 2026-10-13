# Daily data update — systemd user timers

Keeps the `data_pipelines` cache fresh independently of the `/daily-predictions`
cadence. That separation matters: the cadence seeds the cache as part of its
*scoring* run and fails closed, so when a cell cannot reproduce its training book
the data refresh is blocked with it. These units refresh the data on their own
schedule.

Two timers:

| Timer | When | Runs | Covers |
|---|---|---|---|
| `data-update.timer` | `Mon..Fri 07:30` | `daily_update` | sp500 + russell1000 + nasdaq100 → last completed NYSE session |
| `data-update-valuation.timer` | `Sat 08:00` | `daily_update --with-valuation` | the above **+** `us_fundamentals` + the valuation-panel rebuild |

The weekly run also picks up **Friday's** US session, which a Mon–Fri 07:30 IST
schedule cannot see until Monday (Friday's close lands ~01:30 IST Saturday).

`data-update.timer` runs an hour before the predictions cadence so the cache is
already fresh when scoring starts.

## What the updater does

1. Disk pre-flight (≥ 10 GB) — the project's FS wedges near-full.
2. Compute the last **completed** session from the NYSE calendar in
   `America/New_York` terms. Deliberately ET-based: IST is a calendar day ahead
   of ET for the entire US session, so a local-date check can count a still-open
   session as finished.
3. Seed each universe from **the cache's own last session** up to `end`.
   Deliberately not a fixed trailing window: a fixed window can only ever refill
   itself, so an outage longer than it would leave a hole that never closes. Spans
   over 90 sessions are chunked, which also keeps them off the domain's slow
   Stooq-first "big gap" tier.
4. **Corporate-action seam guard.** Rows fetched at one time sit on the
   adjustment basis then in force; rows fetched later on the basis in force now.
   Any ticker with a split in between gets a discontinuity exactly at the seam —
   `close` continuous but `adj_close` stepping by the split factor (NASDAQ:CRWD
   4:1), or both series jumping (NYSE:DD 1-for-3 reverse split). A daily
   incremental refresh recreates this continuously, so the guard runs every time:
   it detects ratio steps and implausibly large one-day moves, then re-fetches the
   affected tickers' **full history from a single provider response** so each
   series has one basis. Recently-repaired tickers are skipped for the scan
   window so a genuine large move is not re-fetched daily.

   The repair is **crash-safe**. Each ticker is journalled to
   `data/.daily_update_pending.json` (written to disk) *before* its rows are
   touched, deleted on its own, then re-fetched. If the re-fetch fails, the
   series is rebuilt offline from the immutable raw store
   (`data_pipelines reprocess`); if that also fails the journal entry is left and
   healed at the start of the next run. This matters because the naive form —
   delete everything, then re-fetch in a loop — loses data permanently: a ticker
   with zero rows is invisible to the seam scan, the state file records only
   successes, and no scheduled seed fetches enough history to restore it.

5. `--with-valuation` additionally refreshes `us_fundamentals` and rebuilds
   `results/valuation/data/valuation_panel.parquet`. That leg requires the
   default `--data-root` (the panel builder reads `<repo-root>/data` and resolves
   fundamentals through the library default); a custom root fails loudly rather
   than silently rebuilding from the wrong cache.

Exit status is non-zero if any seed, repair or optional step failed.

## Install (one time)

1. Edit the `WorkingDirectory` + `ExecStart` paths in both `.service` files to
   your repo root.
2. Link the units and enable the timers:

   ```bash
   mkdir -p ~/.config/systemd/user
   for u in data-update.service data-update.timer \
            data-update-valuation.service data-update-valuation.timer; do
     ln -snf "$PWD/scripts/data_pipelines/systemd/$u" ~/.config/systemd/user/
   done
   systemctl --user daemon-reload
   systemctl --user enable --now data-update.timer data-update-valuation.timer
   ```

3. (Optional, to run while logged out:)
   `sudo loginctl enable-linger "$USER"`

## Operate

```bash
systemctl --user list-timers 'data-update*'      # next/last fire
systemctl --user start data-update.service       # run now
journalctl --user -u data-update.service -n 50   # last run's output
```

Manual run outside systemd:

```bash
uv run python -m scripts.data_pipelines.daily_update [--with-valuation]
```

## Notes

- The updater does **not** advance the forward log — that is
  `/daily-predictions` (see `scripts/backtests/systemd/`). Data and scoring are
  deliberately decoupled.
- The fundamentals leg's documented chain is macrotrends → SEC EDGAR → yfinance.
  macrotrends is HTTP-403 IP-blocked on some hosts; when that happens EDGAR
  serves every row and carries native `filed_date`, so the point-in-time lag is
  unaffected. The provider actually used is reported in the run log.
- Repair state lives at `data/.daily_update_repairs.json` (gitignored with the
  rest of `data/`). Deleting it just re-enables scanning for those tickers.
