---
name: project-overview
description: High-level description of the agent-driven-investment repo and the analog_mc module
metadata:
  type: project
---

This repo will host multiple forecasting/analytics modules; `analog_mc` is the first.

**analog_mc** is a probabilistic forecasting pipeline. It generates Monte Carlo simulations of forward price paths by finding historical analogs (k-NN in multi-horizon z-score space), sampling forward 10-day blocks of realized returns from those analogs, and rescaling each block to match the current vol regime.

Asset-agnostic by design — horizons, z-score windows, and clip bounds are config-driven. v1 default config targets a quarterly forecast on a broad equity index (60-day forecast = 6 × 10-day blocks; z-score horizons (20, 50, 200)).

**Why:** The plan is the output of a long design conversation in `docs/analog_mc/IMPLEMENTATION_PLAN.md`. Architectural decisions there were each made for a reason and should not be silently changed — see [[feedback-implementation-discipline]].

**How to apply:** When implementing or modifying any pipeline component, treat the IMPLEMENTATION_PLAN.md as the source of truth. The 6 critical correctness constraints (C1–C6) and the 11-stage build order are non-negotiable. See [[project-layout]] for where module code lives.
