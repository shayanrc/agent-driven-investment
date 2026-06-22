# _263 — Macro-lattice sweep (sp500): does F17 macro help across the grid?

> **Update (2026-06-22) — superseded by `_264`.** These results are on a single
> (trailing 2026-Q1) window. `_264` re-ran this exact matched lattice on an independent
> **date-aligned** window (2024-H2): the broad top-of-book help did **NOT replicate**
> (8/12 cells flip sign at R-p@1; the headline `+20%/50d` +0.30 → −0.11), and `_264`
> also closed the 5 long-horizon cells that were Q=0 here. Net verdict: F17 is
> contextually additive but **not a robust edge — not deployed.** The "still gated on
> multi-window validation" caveat below is now resolved (validation failed). Read `_264`.

**Headline:** Sweeping the matched base-vs-`+macro` A/B across the sp500 cell lattice
generalizes the `_262` two-cell win: **macro (F17) is broadly additive at the TOP of
the ranking.** Across the 12 testable cells it helps **R-Precision@1 in 10/12**
(mean +0.07, median +0.09), **@5 in 9/12** (mean +0.04), and dilutes to roughly
neutral by **@10 (5/12 help, mean +0.01)**. Strongest at **+20%** (e.g. +20%/50d
@1 +0.30). The macro edge concentrates on the single highest-conviction pick and
fades as K grows — consistent with regime context sharpening the top of the book.

Heatmap: `results/gbdt/data/_263_macro_lattice_heatmap.png`.

## Setup

- **17 canonical sp500 cells**, each run base (`candidates: all`) vs `+macro`
  (`all_macro`) under an **identical fixed config** (xgboost, `min_child_weight=10`,
  single fit) — only `features.candidates` differs, so each cell's `macro − base`
  delta is a clean read of the macro contribution (the `_260` per-arm-HP confound is
  avoided by construction). Trailing split, snapshot 2026-06-20.
- Tooling: `scripts/gbdt/{gen_macro_sweep_specs.py, run_macro_sweep.sh,
  aggregate_macro_sweep.py}`. 24 registry rows (12 cells × 2 arms) added.

## Results — testable cells (macro − base; raw R-Precision + base rate)

| cell | base_rate | @1 base→macro | @5 base→macro | @10 base→macro |
|---|---|---|---|---|
| +10%/5d | 0.049 | 0.211→0.232 (+0.02) | 0.215→0.227 (+0.01) | 0.215→0.221 (+0.01) |
| +10%/10d | 0.117 | 0.200→0.322 (+0.12) | 0.282→0.336 (+0.05) | 0.298→0.294 (−0.00) |
| +10%/25d | 0.231 | 0.293→0.373 (+0.08) | 0.277→0.325 (+0.05) | 0.273→0.311 (+0.04) |
| +10%/50d | 0.279 | 0.320→0.220 (**−0.10**) | 0.300→0.264 (−0.04) | 0.316→0.258 (−0.06) |
| +20%/5d | 0.009 | 0.084→0.145 (+0.06) | 0.137→0.138 (+0.00) | 0.228→0.179 (−0.05) |
| +20%/10d | 0.028 | 0.281→0.427 (+0.15) | 0.229→0.260 (+0.03) | 0.233→0.248 (+0.02) |
| +20%/25d | 0.088 | 0.280→0.373 (+0.09) | 0.325→0.419 (+0.09) | 0.312→0.393 (+0.08) |
| +20%/50d | 0.150 | 0.140→0.440 (**+0.30**) | 0.344→0.420 (+0.08) | 0.318→0.424 (+0.11) |
| +40%/25d | 0.024 | 0.467→0.307 (**−0.16**) | 0.285→0.271 (−0.01) | 0.318→0.300 (−0.02) |
| +40%/50d | 0.057 | 0.600→0.640 (+0.04) | 0.412→0.484 (+0.07) | 0.422→0.398 (−0.02) |
| +50%/25d | 0.013 | 0.113→0.226 (+0.11) | 0.166→0.246 (+0.08) | 0.338→0.343 (+0.01) |
| +50%/50d | 0.038 | 0.540→0.680 (+0.14) | 0.380→0.484 (+0.10) | 0.457→0.502 (+0.05) |

Aggregate: **@1** helps 10 / hurts 2; **@5** helps 9 / hurts 2 / neutral 1;
**@10** helps 5 / hurts 4 / neutral 3. Monotone dilution with K.

## Untestable cells (Q=0) — a real limitation

5 long-horizon cells returned **Q=0** (empty test window) at this snapshot:
`+20%/100d`, `+40%/100d`, `+40%/200d`, `+50%/100d`, `+50%/200d`. Under the trailing
split the **complete-label boundary** (`snapshot − H` trading days) falls *before* the
eval/test boundary for H ∈ {100, 200}, so there are no labelable test days. These cells
are **not evaluated** here — testing them needs a `date_aligned` split or an earlier
snapshot. (Shown as blank/· in the heatmap.)

## Reconciliation + verdict

- `_262` (2 champion cells, matched HP): macro beats the champion. `_263` (12-cell
  lattice, same matched recipe): macro helps **broadly** at @1/@5 — **not a two-cell
  fluke.**
- The two regressions (+10%/50d, +40%/25d) are isolated; no systematic harm region.
- **The benefit is a top-of-book effect** (@1 ≫ @5 ≫ @10). For a top-K strategy that
  concentrates in the highest ranks (the `_020` finding), this is the useful regime.

**Verdict:** strong, broad positive signal for macro at the top of the sp500 lattice —
the most encouraging macro result so far. Still **gated for deployment** on: (a)
**multi-window validation** (this is one snapshot), (b) the **untestable long-horizon
cells** (re-run on `date_aligned` to close the grid), (c) `mcw=10` is fixed (not each
cell's optimum), (d) HY-OAS is still a proxy.

## Artifacts

- Heatmap: `results/gbdt/data/_263_macro_lattice_heatmap.{png,svg}`
- Per-cell data: `results/gbdt/data/_263_macro_lattice_data.json` (+ `_raw.json`)
- Registry: 24 rows (`*_swbase` / `*_swmacro`) in `results/gbdt/data/r_precision_at_k.csv`
- Aggregator: `scripts/gbdt/aggregate_macro_sweep.py`
