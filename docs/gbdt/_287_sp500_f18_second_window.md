# _287 — sp500 F18 second-window sweep: the 100d win does NOT replicate

**Verdict: F18 (US fundamentals) does not robustly help sp500 at 100d — the
nifty500 `_285` "F18 helps at 100d" result was a window/market artifact.**
The champions stand; no promotion. This is the independent second-market
replication test (task #28), and it fails — consistent with the fragile-edge
story of the whole fundamentals program (F17-macro `_264`, F18 `_278`/`_280`
all failed their second windows).

## Setup

Regime-corrected **mirror of the nifty500 `_285` A/B** (memo `_286`/spec-gen
`scripts/gbdt/gen_sp500_f18_regime_sweep.py`): 17 sp500 cells × {rfbase, rffund},
date_aligned train_start **2015**, **calendar2** tokens (`all_calendar2` vs
`all_fundamentals_calendar2`), xgboost default HP, single fit — so each cell's
`fund − base` delta is a clean F18 read. US valuation panel built for this run
(953 tickers, 842/953 PE coverage, `--start 2013` to cover the 2015 train start).
34/34 arms, 0 failures. Test window snapshot-end 2025-07-01.

## Result — fund-minus-base by horizon (test, macro-avg over cells)

| horizon | Δ AUC | Δ R-p@1 | Δ R-p@3 | Δ R-p@10 |
|---|--:|--:|--:|--:|
| 5d | +0.004 | −0.038 | −0.019 | −0.036 |
| 10d | −0.001 | −0.055 | −0.007 | −0.013 |
| 25d | +0.002 | −0.006 | −0.021 | +0.011 |
| 50d | +0.001 | −0.008 | −0.039 | −0.002 |
| **100d** | **−0.009** | **−0.070** | **−0.000** | **+0.004** |
| 200d | −0.027 | +0.087 | +0.029 | −0.020 |

**100d cells** (base_rate): 20/100d (0.218) ΔAUC −0.030 / Δrp1 +0.046 / Δrp3 +0.020
(marginal, AUC drops); 40/100d (0.047) Δrp1 **−0.230**; 50/100d (0.023) Δrp3 −0.017.
The most reliable 100d cell (20/100d — the direct nifty500 twin) is marginal at
best; the other two are negative → mean ≈ neutral-to-negative.

## Reading

- **No robust F18 win at 100d.** The nifty500 100d effect did not carry to sp500.
- **Weak, inconsistent 200d signal** (Δrp1 +0.087, Δrp3 +0.029) — but only 2 cells,
  comes WITH an AUC (−0.027) + R-p@10 (−0.020) cost, on tiny base rates. Not
  adoptable; flag, don't ship.
- Magnitudes are small (±0.02–0.09) on a single test window; high-threshold cells
  (40/50%) carry base rates 0.002–0.09 → R-p@K is noise-dominated there.

## Disposition

Task #28 answered: **F18 is contextually additive but not a robust, adoptable
signal** — reaffirms `[[project-gbdt-macro-features-f17]]`-style "not robust"
reads. No champion swap, not wired into `/daily-predictions` (the existing
`sp500_40_200` fund candidate stays `deployed=False`, forward-comparison only).
Fundamentals program parked pending a materially different feature/label design.
