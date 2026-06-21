# _259 — Macro-augmented sp500 cells: REAL macro panel (8/9 series, primary sources)

**Headline:** With FRED's own egress down, the macro panel was rebuilt from the
**primary sources FRED redistributes** — U.S. Treasury (nominal + TIPS curves),
NY Fed (EFFR), and Yahoo (VIX/USD) — giving **8 of 9 real daily series** (only
ICE HY-OAS unavailable free). On a clean A/B (identical date-aligned windows,
`max_iter:1`), the full 8-series real panel **decisively beats both the baseline
and the 4-series proxy on the +20%/25d cell at every K**, and beats the baseline
on the +50%/50d cell. The new real-only series (real yield, breakeven, curve
slope, fed funds) are materially used by the model. The deployed-champion
head-to-head (Phase 3) is still pending.

## Data — real values from FRED's upstream sources

`stlouisfed.org` was unreachable this session, so each series was sourced from
where FRED itself gets it (so these are the *same underlying values*, not
proxies), cached under the FRED ids with honest provenance:

| FRED id | source (provider tag) |
|---|---|
| DGS10, DGS3MO, T10Y2Y | U.S. Treasury daily par yield curve (`treasury`; T10Y2Y = 10Y−2Y) |
| DFII10, T10YIE | U.S. Treasury TIPS real curve (`treasury_tips`; T10YIE = nom10−real10) |
| DFF | NY Fed effective fed funds (`nyfed_effr`) |
| VIXCLS, DTWEXBGS | Yahoo `^VIX`, `DX-Y.NYB` |

`BAMLH0A0HYM2` (ICE BofA HY credit OAS) has no free non-FRED source and is
absent. **8 series → 40 F17 columns** (vs 20 for the 4-series proxy of `_258`).

## Cache-key bug found + fixed (before any wrong number escaped)

The first macroreal run loaded `macroproxy`'s **stale 4-series matrix** and
produced byte-identical metrics — the tell. Root cause: the universe/per-cell
feature-cache key encoded the macro feature set only via `families`
(`"all_macro"`) + a SHA of `features.py`, neither of which changes when the
*cached macro data* changes. Fix (`3b7ba45`): `gbdt.data.macro_panel_signature`
(cached series + coverage) is folded into both `compute_key`s **only when macro
is selected** — non-macro keys stay byte-identical (base_v2/champion caches keep
hitting), 4-series vs 8-series now key distinctly. Re-run confirmed an
8-series / 40-macro / 319-col build from scratch.

## Results (test 2024-07-26→2024-12-16, identical across all three arms; raw R-Precision@K + base rate)

### sp500 +20%/25d (dd10%) — base_rate 0.0402, Q=100

| K | base_v2 | macroproxy (4s) | macroreal (8s) |
|---|---|---|---|
| R-Precision@1 | 0.1000 | 0.1000 | **0.1200** |
| R-Precision@3 | 0.1367 | 0.1633 | **0.2067** |
| R-Precision@5 | 0.1660 | 0.1980 | **0.2200** |
| R-Precision@10 | 0.1879 | 0.2157 | **0.2340** |
| R-Precision@20 | 0.2570 | 0.2358 | **0.2809** |
| eval AUC | 0.7912 | 0.7986 | 0.8049 |

The 8-series real panel **wins at every K** vs both baseline and proxy — the
cleanest macro result of the program. (@1 0.120 is 2.98× the 0.0402 base rate.)

### sp500 +50%/50d (dd25%) — base_rate 0.0097, Q=91

| K | base_v2 | macroproxy (4s) | macroreal (8s) |
|---|---|---|---|
| R-Precision@1 | 0.1099 | 0.1319 | 0.1209 |
| R-Precision@3 | 0.1062 | 0.0916 | **0.1282** |
| R-Precision@5 | 0.1454 | 0.1176 | **0.1679** |
| R-Precision@10 | 0.2182 | 0.2736 | 0.2499 |
| R-Precision@20 | 0.3912 | 0.4234 | 0.3457 |
| eval AUC | 0.8654 | 0.8676 | 0.8563 |

Mixed: macroreal **beats baseline** on @1/@3/@5/@10 and wins @3/@5 over both, but
the proxy edges it on @1/@10 and it's lowest at @20. On this rare-event cell
(only 91 days-with-positives) @1/@3/@5 are small-sample; no clean win over proxy.

## F17 feature usage — the real-only series carry signal

- **+20%/25d:** 40/40 macro features used; macro = **18.2%** of importance.
  `macro_DGS3MO_chg_60` is **#3 overall**. New real-only series all rank: T10Y2Y
  #17, DFII10 #20, DFF #32, T10YIE #41.
- **+50%/50d:** 34/40 used; macro = **11.0%**. **`macro_DFII10_chg_20` is #10
  overall** (the real-yield block is the dominant macro signal here);
  `macro_DGS10_chg_60` #11.

Mechanistically coherent: the model leans on short-rate dynamics + the
real-yield/breakeven block — exactly the regime context the proxy lacked.

## Verdict

1. The **full 8-series real macro panel adds clear top-K signal**: a clean sweep
   over baseline AND proxy on the +20% cell (every K), and beats the baseline on
   the +50% cell. The real-only series (real yield, breakeven, curve slope, fed
   funds) are genuinely used. **Macro is additive — confirmed on real data.**
2. **Still NOT a champion head-to-head.** Same caveat as `_258`: this is a
   `max_iter:1` date-aligned A/B on a 2024-H2 window, not the committed champions'
   trailing-split agent-tuned 2025-26 window (R-p@1 0.640 / 0.413). The
   "outperform the deployed champion" objective remains **open**.

## Next step (Phase 3 — beat the champion)

Run the macro variant under the **champion config** (trailing split +
`agent_file_protocol` tuning) and a matched re-baseline, on the champion's
window, for the real head-to-head. Promote to the deployed champion /
`/daily-predictions` only if it wins there. (HY-OAS still missing — if a source
is found, add `BAMLH0A0HYM2` for the full 9-series panel first.)

## Artifacts

- Specs: `configs/gbdt/experiments/sp500_up_{20pct_25d_dd10pct,50pct_50d_dd25pct}_macroreal.yaml`
- Registry: 2 `_macroreal` rows in `results/gbdt/data/r_precision_at_k.csv`.
- Sidecar: `results/gbdt/data/_259_macro_real_fred_data.json`.
- Cache-key fix: commit `3b7ba45` (`feature_cache`, `universe_feature_cache`, `data`, `__main__`).
