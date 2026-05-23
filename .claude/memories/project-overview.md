---
name: project-overview
description: Repo hosts multiple forecasting modules; analog_mc is the first. Module-specific goal lives in docs/<module>/goal.md.
metadata:
  type: project
---

This repo will host multiple forecasting/analytics modules. **analog_mc** is the first and currently only one — analog Monte Carlo probabilistic price-path forecasting.

**For analog_mc specifics (what it optimizes for, what success looks like, eventual deployment shape):** read [`docs/analog_mc/goal.md`](../../docs/analog_mc/goal.md). That document is the authoritative source — don't duplicate its content here.

**For *how* analog_mc works:** `docs/analog_mc/IMPLEMENTATION_PLAN.md` (architecture spec, 6 correctness constraints, 11-stage build order) and `docs/analog_mc/ALGORITHM.MD` (step-by-step math). CLAUDE.md at the repo root has the short rules; the long-form why is in those plan docs.

**How to apply:** When a new module gets added, give it its own `docs/<module>/goal.md` following the same pattern. The CLAUDE.md rule "read `docs/<module>/goal.md` before editing files in that module" applies generically.
