---
name: feedback-implementation-discipline
description: Follow the IMPLEMENTATION_PLAN stage order; surface deviations rather than silently change
metadata:
  type: feedback
---

For analog_mc, follow the 11-stage build order in `docs/analog_mc/IMPLEMENTATION_PLAN.md` strictly. Do not skip ahead. If implementation reveals a problem with a documented decision, surface it explicitly and ask before deviating.

**Why:** The plan is the output of a long design conversation. Each decision was made for a specific reason (often a subtle correctness or diagnostic concern), and the plan's "What not to do" section flags the most likely silent-failure modes. The plan explicitly says: "Do not silently change architectural decisions." The diagnostic infrastructure (Stages 6, 9) is what makes the pipeline trustworthy — building out of order defeats that engineering.

**How to apply:**
- Implement stages in order: 1 (config/data/features) → 2 (folds) → 3 (distance/prob) → 4 (sampling) → 5 (forecast) → 6 (CRPS) → 7 (search) → 8 (walk-forward) → 9 (diagnostics) → 10 (aggregate) → 11 (dashboard).
- Within each stage, write unit tests *before* moving on. The causality test (C1) is the single highest-value test.
- If an architectural deviation seems necessary, raise it with the user (cite the plan section) and get explicit approval before changing. Examples already approved: per-module namespacing (`<top>/<module>/` instead of flat layout) and `data/NASDAQ100.csv`-only loader for v1.
- Don't pre-implement v2 features (trailing-momentum drift, conditional block sampling, tail inflation). They are gated on specific diagnostic findings.
