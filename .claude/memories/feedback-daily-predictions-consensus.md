---
name: feedback-daily-predictions-consensus
description: When reporting the daily predictions, ALWAYS also include the cross-model consensus (top-5 vote across the 5 tracked models; winner = most-voted).
metadata:
  type: feedback
---

Whenever presenting the `/daily-predictions` forward signal to the user, ALSO show the **cross-model consensus** for the same snapshot, alongside the per-model picks table ([[feedback-predictions-table-format]]).

**Why:** the user tracks the consensus (the `_028` cross-model vote) as a first-class part of the daily read, not a one-off ad-hoc request. Stated 2026-07-01 ("from now on, whenever you report the daily predictions include the consensus as well").

**How to apply:** for the snapshot being reported, pool all five models' **top-5** picks, tally votes per stock (tie → highest summed `p`), and render a **vote table** — columns **Stock | # models | Σp | voting models** — plus the **consensus winner** (the strategy's single daily pick). Flag which names clear the **≥50%-of-panel (≥3/5)** majority. Compute from the committed `results/backtests/data/forward_predictions_log.csv` (the logged top-K per model) — no re-inference needed. Keep the `_028` framing (bull-only momentum amplifier; 1 stock/day; not promoted; needs the SMA200 regime gate) + the usual caveats in prose. The `/daily-predictions` SKILL "report the read" step points here.
