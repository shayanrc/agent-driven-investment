---
name: project-daily-predictions-fundamentals-inference
description: The /daily-predictions inference path builds technical-only features by default; F18/F19 fundamentals cells take a separate fundamentals-aware build branch (added _030). Don't assume a fund cell is deployable without it.
metadata:
  type: project
---

The `/daily-predictions` cadence scores through `infer_fresh_predictions._build_one`. By default it builds **technical-only** features via the on-disk incremental cache (`incremental_feature_cache.build_or_extend` → `build_feature_matrix(panel, index_df, annualization=…)`, i.e. `families="all"`, `fund_df=None`). The valuation panel is loaded **only in the training path** (`gbdt/experiment_runner.py`), never the inference path.

So an **F18/F19 fundamentals cell** (feature token `all_fundamentals` / `all_fundamentals2`) cannot be scored by the default path — the `fund_*` columns are simply absent from the build, and `_build_one` raises `RuntimeError: features.yaml columns absent from build: [...]`, aborting the whole cadence.

**Fixed in `_030`** (branch `daily-preds-fundamentals-inference`): `_build_one` now reads the cell's `spec["features"]["candidates"]` token and branches. Technical cells (`all`) are unchanged (byte-identical incremental cache). Fundamentals cells (`FUNDAMENTALS_TOKENS = {"all_fundamentals","all_fundamentals2"}`) load `load_fundamentals_panel(...)` and **full-build** the matrix directly with `families=token, fund_df=fund_df` (mirroring the training build). The shared `feat_cache` key gained the token so fund + technical cells of the same universe never collide. First fund candidate: `sp500_up_40pct_200d_dd20pct_w2ffundtune` (`deployed=False`).

**Why / how to apply:**
- Before adding ANY new `/daily-predictions` candidate, check its `features.candidates`. Technical (`all`) → wire straight into `CELLS`. Fundamentals → the `_030` branch handles it, but note the two caveats below. Macro (`all_macro`, F17) is **still unsupported** by inference — would need the same treatment (and macro is not promoted anyway, `[[project-gbdt-macro-features-f17]]`).
- **Correctness guard:** the per-cell self-check reproduces `predictions/test.csv` to `<1e-4`; a faithful fresh fund build proves itself (w2ffundtune → 2.97e-08). If it aborts, the build diverged (lookbacks/exclude/fund-join mismatch) — don't emit.
- **Caveat 1 — no incremental fund cache:** fund cells full-build the ~7y warmup slice each run (sp500 ~2–3 min). A fund incremental cache is a `docs/gbdt/V1.8_TBD.md` optimization.
- **Caveat 2 — freshness:** the daily seed refreshes only `us_equities`. Keep a fund candidate current by rebuilding the valuation panel separately (`scripts.valuation.build_valuation_panel`) as the `us_fundamentals` cache grows — weekly suffices (F18 is point-in-time on `filed_date`, ~quarterly).

Memo: `docs/backtests/_030_fundamentals_daily_candidate.md`. Related: `[[feedback-daily-predictions-consensus]]`, `docs/gbdt/_279`/`_280` (why w2ffundtune is the top fundamental model, and why F18 is not promoted).
