# _278 — window-2 confirmation: the F18 long-horizon edge fails; the backend survives

**Question (V1.7_TBD §5).** Do the `_274`–`_277` findings on the top-3 F18 cells
(+40%/200d, +20%/100d, +20%/50d) replicate on an independent window — the F18
increment, the `_275` ffundtune candidate (R-p@1 0.58), and the `_277` CatBoost
advantage?

**Design.** Window 2 = **date_aligned `train_start: 2019-07-01`** → test
**2025-01-24 → 2025-06-17** (Q=100), zero overlap with window 1's test
(2024-07-26 → 2024-12-16). A literal trailing split gives Q=0 at H=100/200 (`_263`),
hence a second date-aligned window (the `_264` precedent). Same snapshot
(2026-07-02), matrices warm. 17 arms — a full **backend × features factorial** per
cell per window: xgb base / xgb fund / cb base / cb fund (defaults), plus the
`_275` ffundtune config (141-col FS list + λ4.5, replayed via a scripted
agent-protocol replica — val argmin confirmed on iter 1, so the artifact is the
exact config under test) and cb depth-4 on +20%/100d (the `_277` every-K winner).
The 6 cb-base arms complete the factorial on BOTH windows (they were the missing
attribution cell in `_277`). All single fits at defaults unless stated; window-1
xgb numbers from `_274`–`_276`, cb-fund from `_277`.

**Regime note.** The windows differ materially: +40%/200d base_rate 0.119 (w1) →
0.221 (w2); +20%/50d 0.128 → 0.140. Cross-window comparisons of raw R-p@K are
confounded by base rate — the meaningful reads are **within-window arm deltas**,
compared *across* windows for sign stability.

## Results — test windows, raw values (base_rate for reference)

**+40%/200d** (w1 base 0.1189 | w2 base 0.2213):

| arm | window | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xgb base | w1 | 0.6528 | 0.1032 | 0.140 | 0.293 | 0.296 | 0.306 | 0.309 |
| xgb fund | w1 | 0.6784 | **0.1013** | 0.330 | 0.327 | 0.338 | **0.364** | **0.367** |
| xgb fund tuned (`_275`) | w1 | 0.6581 | 0.1015 | 0.580 | **0.460** | **0.414** | 0.359 | 0.336 |
| cb base | w1 | **0.6982** | 0.1020 | **0.750** | 0.437 | 0.382 | 0.310 | 0.317 |
| cb fund (defaults) | w1 | 0.6924 | 0.1024 | 0.650 | 0.310 | 0.228 | 0.307 | 0.320 |
| cb fund depth-4 | w1 | 0.6977 | 0.1026 | 0.500 | 0.290 | 0.252 | 0.307 | 0.354 |
| xgb base | w2 | 0.7275 | 0.1580 | 0.580 | 0.550 | 0.550 | 0.539 | 0.487 |
| xgb fund | w2 | 0.7040 | 0.1585 | **0.620** | 0.543 | 0.516 | 0.479 | 0.434 |
| xgb fund tuned cfg | w2 | 0.7268 | 0.1557 | **0.620** | **0.603** | 0.572 | 0.523 | 0.459 |
| cb base | w2 | **0.7765** | **0.1470** | 0.600 | 0.600 | **0.598** | **0.579** | 0.549 |
| cb fund (defaults) | w2 | 0.7651 | 0.1508 | 0.590 | 0.543 | 0.550 | 0.576 | **0.550** |

**+20%/100d** (w1 base 0.2311 | w2 base 0.2323):

| arm | window | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xgb base | w1 | 0.5928 | 0.1853 | 0.330 | 0.310 | 0.328 | 0.328 | 0.318 |
| xgb fund | w1 | 0.5745 | 0.1875 | 0.550 | 0.397 | 0.370 | 0.341 | 0.331 |
| cb base | w1 | 0.6384 | **0.1841** | 0.260 | **0.593** | **0.562** | **0.454** | 0.397 |
| cb fund (defaults) | w1 | 0.6523 | 0.1859 | 0.270 | 0.533 | 0.456 | 0.403 | 0.384 |
| cb fund depth-4 | w1 | **0.6565** | 0.1852 | **0.660** | 0.510 | 0.456 | 0.441 | **0.425** |
| xgb base | w2 | 0.6162 | 0.1741 | 0.440 | 0.430 | 0.438 | 0.434 | 0.440 |
| xgb fund | w2 | 0.6326 | 0.1718 | 0.350 | 0.393 | 0.416 | 0.425 | 0.427 |
| cb base | w2 | 0.7254 | 0.1622 | 0.420 | 0.470 | 0.502 | 0.513 | 0.483 |
| cb fund (defaults) | w2 | **0.7372** | **0.1613** | **0.530** | **0.560** | **0.546** | **0.518** | **0.508** |
| cb fund depth-4 | w2 | 0.7282 | 0.1620 | 0.520 | 0.527 | 0.528 | 0.493 | 0.485 |

**+20%/50d** (w1 base 0.1277 | w2 base 0.1404; the `_275` tuned arm ≡ xgb fund —
iter-0 revert):

| arm | window | AUC | test Brier | R-p@1 | R-p@3 | R-p@5 | R-p@10 | R-p@20 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| xgb base | w1 | 0.6878 | 0.1141 | 0.160 | 0.300 | 0.310 | 0.314 | 0.308 |
| xgb fund | w1 | 0.6866 | 0.1134 | **0.440** | **0.393** | 0.340 | 0.317 | 0.292 |
| cb base | w1 | **0.7357** | 0.1123 | 0.260 | 0.303 | 0.336 | **0.375** | **0.376** |
| cb fund (defaults) | w1 | 0.7335 | 0.1126 | 0.160 | 0.307 | 0.310 | 0.362 | **0.376** |
| cb fund depth-4 | w1 | 0.7299 | **0.1117** | 0.170 | 0.343 | **0.350** | 0.357 | 0.373 |
| xgb base | w2 | 0.6729 | 0.1155 | 0.430 | 0.380 | 0.386 | 0.393 | 0.410 |
| xgb fund | w2 | 0.6854 | 0.1149 | 0.350 | 0.397 | 0.401 | 0.422 | 0.421 |
| cb base | w2 | **0.8200** | 0.1019 | **0.560** | 0.467 | **0.463** | 0.454 | 0.443 |
| cb fund (defaults) | w2 | 0.8198 | **0.1019** | 0.540 | **0.503** | **0.463** | **0.457** | **0.457** |

## Findings

1. **The F18 increment on these cells is window-specific — it fails confirmation.**
   On xgboost, the `_274` fbase→ffund deltas that defined the "F18-edge band" do
   not replicate: on w2 the book delta is negative at @3–@20 on +40%/200d, negative
   at every K on +20%/100d, and mixed on +20%/50d (@1 −0.08). On CatBoost (the
   factorial's new cells) the increment is negative-to-nil on 5 of 6 cell-windows
   (+40%/200d both windows, +20%/50d both windows ≈ tie, +20%/100d w1 mixed) — only
   +20%/100d w2 shows fund clearly helping (every K). This is the `_264` macro
   pattern repeating for F18: **contextually additive, not robust**. The
   `_272`/`_273` champion validation (+50%/50d) is untouched — it passed its own
   two-window test; what dies here is the long-horizon extension.
2. **The CatBoost backend advantage is the robust finding — and it does not need
   F18.** A CatBoost arm holds the best AUC on **all 6 cell-windows** (cb-base
   itself beats every xgb arm on AUC 6/6, up to +0.135 on +20%/50d w2's 0.820) and
   the best Brier on 5 of 6. On +20%/100d a cb arm beats every xgb arm at **every
   K on both windows** (w1: depth-4; w2: defaults). On +20%/50d w2 both cb arms
   beat both xgb arms at every K. Technical-only CatBoost at pure defaults even
   posts the best top-book ever recorded on +40%/200d (w1 @1 **0.750**, 6.3× base).
3. **The `_275` ffundtune candidate partially survives — as the best xgb arm.** Its
   config (fixed 141-col list + λ4.5) replicates direction on w2: best xgb book
   (@1 0.620 / @3 0.603 / @5 0.572, best xgb Brier). The `_276` suspicion that it
   was pure val-window selection is NOT borne out — but cb-base matches/beats it
   from @5 down and on AUC/Brier, so the xgb-vs-xgb question is mooted by the
   backend result.
4. **Which cb arm wins flips between windows** (+20%/100d: depth-4 on w1, defaults
   on w2; +40%/200d: fund@1 on w1, base broadly on w2) — consistent with `_276`:
   fine config selection inside a backend is window noise. The robust prescription
   is the coarsest one: **CatBoost, defaults, technical features**.

## Caveats

Two windows, Q=100 each, same universe — sign-stability across two regimes
(2024-H2, 2025-H1), not a general proof. The w2 regime is richer (base rates up,
every arm's raw numbers up); within-window deltas are the honest read. cb-base's
w1 @1 0.750 is a single-window top-1 (the metric `_276` showed is noisiest).
No champion cell was touched.

## Verdict

- **V1.7_TBD §5 is RESOLVED — negative for F18.** No long-horizon F18 candidate
  cells: the increment doesn't replicate. F18 remains validated only where
  `_272`/`_273` left it (the +50%/50d champion).
- **New lead: CatBoost-base (defaults, technical-only) on long-horizon cells.**
  Robust two-window AUC/Brier/deep-book advantage, no F18 dependency, no tuning.
  §6 (cb pass over the deployed champion cells) is upgraded — it's now the
  highest-value cheap experiment in the module.
- No champion change, no `/daily-predictions` change (human decisions, `_019`).

Registry: 17 rows (`*_w2*` ×11 + `*_cbbase`/`*_w2cbbase` ×6). Specs:
`configs/gbdt/experiments/*_w2*.yaml`, `*_cbbase.yaml`; driver:
`scripts/gbdt/run_f18_w2_confirmation.sh`. Prior: `_277` (cb control), `_276`
(xgb agent loop), `_275` (default loop), `_274` (lattice), `_272`/`_273`
(champion A/Bs). Plan: `V1.7_fundamentals_features_plan.md`.
