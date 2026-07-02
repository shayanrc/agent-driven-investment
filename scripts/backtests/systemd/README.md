# Daily forward-prediction cadence — systemd user timer

Unattended automation for `scripts/backtests/daily_forward_predictions.py` on a
**not-always-on** machine. A **user** timer (no root) with `Persistent=true` runs
the most recent missed occurrence on the next boot/wake/login; the pipeline itself
self-gates (no-op on non-trading days) and **backfills** every missed trading day,
so a weekend / holiday / vacation gap is caught up correctly on the next run.

**Two tiers (V1.6 seed track).** The **daily** timer runs `--deployed-only`: it
seeds + scores only the two deployed sp500 champions (the signal record), skipping
the heavier russell1000/nasdaq100 comparison candidates, so the daily run stays fast
(one sp500 build; no ~1,006-ticker russell seed). A **weekly** timer
(`daily-predictions-weekly`) runs the FULL cadence (all five cells) to refresh the
candidates — the backfill logs every candidate day since the last full run. Both
write the same forward log.

On-demand alternative: just run `/daily-predictions` (or the runner directly) when
you sit down — the backfill makes a manual cadence equally correct. The timer is
only for "I don't want to remember to."

## Install (one time)

1. Edit the `/path/to/agent-driven-investment` placeholders in **both**
   `daily-predictions.service` and `daily-predictions-weekly.service` to your repo
   root (and confirm the venv path `.venv/bin/python` exists — `uv sync` creates it).
2. Link the units into the user systemd dir and enable the timer:

   ```bash
   mkdir -p ~/.config/systemd/user
   for u in daily-predictions.service daily-predictions.timer \
            daily-predictions-weekly.service daily-predictions-weekly.timer; do
     ln -snf "$PWD/scripts/backtests/systemd/$u" ~/.config/systemd/user/
   done
   systemctl --user daemon-reload
   systemctl --user enable --now daily-predictions.timer daily-predictions-weekly.timer
   ```

3. (Optional, so it runs while you're logged out too — otherwise it runs on login):
   ```bash
   sudo loginctl enable-linger "$USER"
   ```

## Operate

```bash
systemctl --user list-timers daily-predictions.timer   # next/last fire
systemctl --user start  daily-predictions.service       # run now (manual catch-up)
journalctl --user -u daily-predictions.service -n 50    # last run's output
systemctl --user disable --now daily-predictions.timer  # stop the cadence
```

## Notes

- The service runs with `--commit` (appends + commits the forward log locally; **no push**).
  Push / open a PR on your own cadence (e.g. weekly) to share the accrued log.
- `OnCalendar=Mon..Fri 08:30` targets the prior trading day's finalized close. Adjust
  the time/zone to taste; the pipeline tolerates any fire time (it uses whatever data
  is available ≤ today and self-gates if nothing is new).
- This is a deterministic pipeline — no Claude session is involved in the timer path.
  Run `/daily-predictions` interactively when you also want the regime read + picks.
