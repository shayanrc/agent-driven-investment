# gbdt F17 macro features + clean-feature-A/B HP matching

The **F17 macro feature family** (PRs #197–#200, branch `gbdt-macro-features` lineage)
augments the gbdt panel with daily macro-regime context. The companion finding —
**how to run a clean feature A/B** — is the reusable methodology and is the more
important half of this memory.

## F17 macro family

- **What:** daily FRED-style series broadcast to every `(date, ticker)` row via the
  existing `_broadcast_index_to_panel()` seam, **lagged 1 trading day** (C1 causal),
  with `<id>_level`, `<id>_chg_{20,60}`, `<id>_z_{60,120}` transforms (~45 cols for 9
  series). Code: `src/gbdt/features.py::fred_macro_features` + `MACRO_SERIES`.
- **Opt-in only:** enabled via the `features.candidates: all_macro` token
  (`_ALL_FAMILIES + ("F17",)`). F17 is **NOT** in `_ALL_FAMILIES`, so the default
  `"all"` token stays **byte-identical** — existing/champion models are untouched.
- **Cache key:** `compute_key` folds `gbdt.data.macro_panel_signature(...)` (a cheap
  `fred_macro_meta` lookup) so a different macro series-set invalidates correctly. A
  collision here (macroreal silently loading macroproxy's 4-series matrix) was the
  bug caught adversarially in `_259`; non-macro keys stay byte-identical.
- **Provenance caveat:** the `fred_macro` cache holds 9 series under FRED ids but with
  **NON-FRED provenance** — DGS10/DGS3MO/T10Y2Y/DFII10/T10YIE = Treasury, DFF = NY-Fed
  EFFR, VIXCLS/DTWEXBGS = Yahoo, BAMLH0A0HYM2 (HY-OAS) = `-log(HYG/IEF)` Yahoo proxy —
  because FRED egress was unreachable on this host (see `[[external-data-fetch-on-this-host]]`,
  per-user). Re-running an `all_macro` spec reads whatever is cached; the proxy-era
  results (`_258`) are not reproducible from the current 9-series cache.

## Clean feature A/B requires MATCHED HP (the load-bearing lesson)

Do **NOT** compare feature variants (base vs `+macro`) by running each arm under the
auto-loop (`callback_mode: default`). The loop tunes **each arm independently**, so
the per-arm HP divergence confounds the feature effect with the search's choices.

- `_260` (both arms default-auto) showed macro "sign-flipping" — e.g. sp500 +20% base
  loop → R-p@1 **0.347**, macro loop → **0.120**. That gap was purely *different HP*,
  not a macro property.
- `_262` PINNED identical HP for both arms (`backend.hp_starting: {min_child_weight: 10}`
  + `max_iterations: 1`, a single fit) so only `features.candidates` differed. The
  verdict **reversed**: macro beats the champion's exact config on **both** sp500 cells
  (+50% R-p@1 0.540 → 0.680, above the champion's all-time 0.640; +20% 0.280 → 0.373).
- The matched HP need not be each arm's optimum — the per-cell **delta** is what's
  measured, so a fixed neutral/champion HP held constant across arms is correct.
- Champion reproduction shortcut: both committed sp500 champions converged to a *single*
  config (`mcw=10`, all 279 feats, n_iter=1), so they reproduce via `hp_starting` +
  single fit — **no agent loop needed** to re-baseline them at a new snapshot.

This generalizes beyond macro to **any** gbdt feature comparison. Codified as a
CLAUDE.md "What not to do — gbdt" foot-gun. Related single-cell rules: `[[project-gbdt-tuning-playbook]]`.

## Tooling + gotchas

- `scripts/gbdt/gen_macro_sweep_specs.py [universe]` — emits `<cell>_swbase.yaml`
  (`all`) + `<cell>_swmacro.yaml` (`all_macro`) per canonical cell, both with the
  matched `mcw=10` single-fit config. `scripts/gbdt/run_macro_sweep.sh <uni> [snapshot]`
  — sequential, resumable (skip-if-test.csv) driver.
- **Entry-point gotcha:** pin the snapshot with `python -m gbdt experiment <spec>
  --snapshot-end <DATE>` (the CLI subcommand). `python -m gbdt.experiment <spec>` (the
  module form used by the older `run_<uni>_sweep.sh`) does **NOT** accept
  `--snapshot-end` and exits 2 with *"expected one positional spec path"*.

## Status (as of 2026-06-22)

Opt-in, merged, **promising but NOT deployed**. Matched re-baseline `_262` beats both
sp500 champions on one window; lattice sweep `_263` (sp500 17-cell base-vs-macro) +
multi-window validation are the deployment gate. Caveats: one window, HP-sensitive,
HY-OAS proxy. Do **not** wire macro into `/daily-predictions` until validation confirms.
Experiment trail: `docs/gbdt/_258`–`_263`.
