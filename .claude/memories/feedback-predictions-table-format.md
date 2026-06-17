---
name: feedback-predictions-table-format
description: Always present model predictions as a table — columns Ticker | Model/Rank | p — top 3 per model. Raw p, not lift.
metadata:
  type: feedback
---

When showing model predictions (gbdt / forecasting / the `/daily-predictions` forward signal) to the user, ALWAYS render them as a markdown table with columns **Ticker | Model/Rank | p** (`p` = calibrated probability), containing the **top 3 for each model** — e.g. both sp500 champions (sp500_50 + sp500_20) → 6 rows. `Model/Rank` is one combined column, e.g. `sp500_50 / 1`.

**Why:** the user wants a consistent, compact, scannable format for the actionable picks; prose bullet lists / per-model top-10 dumps are harder to read at a glance. Stated 2026-06-17; promoted from a per-user note to this shared project memory so any session presents predictions the same way.

**How to apply:** default to this table whenever asked for predictions / "what to buy" / today's signal. Keep the **regime-gate state** and the honest caveats (lift-not-certainty, bull-only edge, not advice) as surrounding prose, but the picks go in the table. Top-3-per-model is the floor — extra rows (top-10) or columns (base rate, regime) are fine if the user asks, but always include at least Ticker / Model-Rank / p for the top 3 of both models. Show **raw `p`, not lift, in the table** (consistent with CLAUDE.md § Reporting conventions). The `/daily-predictions` SKILL's "report the read" step points here.
