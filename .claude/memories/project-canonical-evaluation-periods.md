# Canonical evaluation periods

The fixed, regime-corrected window set to use for **all** model training, fine-tuning, and backtesting (US/sp500; NSE cells land on the nearest NSE trading days). Set by the project owner 2026-07-13. Supersedes ad-hoc anchors — the V1.4 D2 `train_start: 2019-01-01` default and the deployed champion's trailing split — for any NEW training / fine-tuning / backtest work.

| Window | Period (inclusive) | Sole purpose |
|--------|--------------------|--------------|
| **train** | 2015-01-01 → 2022-03-29 | model training (fit) |
| **val** | 2022-03-30 → 2023-06-30 (1 yr 3 mo) | feature selection + early stopping |
| **eval** | 2023-07-01 → 2024-06-30 (1 yr) | hyperparameter tuning |
| **test** | 2024-07-01 → 2025-06-30 (1 yr) | final model evaluation + comparison |
| **backtest** | 2025-07-01 → 2026-06-30 (1 yr) | backtesting ONLY |

**Why these windows.** Back-extended to 2015 to de-bias the COVID-rally regime skew: a 2019-anchored train is almost entirely the 2020–2022 rally, so target base rates run 3–5× hotter in train than in test (the `_285` regime correction — see [[project-gbdt-macro-features-f17]] neighbourhood; nifty500 `_285` memo `docs/gbdt/_285_nifty500_f18in_sweep.md`). This window set derives from `_285` (train 2015-01-01→2022-03-29, train_rows 1787) but **shrinks val to end 2023-06-30** (1 yr 3 mo) and re-lays eval/test on clean fiscal-year boundaries, adding a **dedicated held-out `backtest` window (2025-07→2026-06)**.

**Role discipline is the whole point — never bleed roles across windows:**
- `train` — fit only.
- `val` — feature selection + early stopping (NOT hyperparameter tuning).
- `eval` — hyperparameter tuning (NOT final comparison).
- `test` — final model evaluation + cross-model comparison (report R-p@K / AUC / Brier here).
- `backtest` — strategy backtesting ONLY; a separate, later, never-touched window, so backtest returns are on data no model-selection step ever saw. Do NOT compute R-p@K model-comparison on it, and do NOT use `test` for backtesting.

**How to apply.** Dates are fixed calendar boundaries, not row-count-derived. When writing a `date_aligned` spec, pick `train_start`/row counts (or explicit boundaries) that reproduce these windows for the run's snapshot; verify the realized `segment_dates` in `metrics.json` match the table before trusting a run. See CLAUDE.md "Canonical evaluation periods" bullet (the summary) and `docs/gbdt/EXPERIMENT_SPEC.md` § `split`.
