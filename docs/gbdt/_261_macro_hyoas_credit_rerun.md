# _261 — Closing the HY-OAS gap (credit proxy) + +50% re-run: it doesn't help

**Headline:** The one macro axis missing from `_259`/`_260` was credit spread
(`BAMLH0A0HYM2`, ICE HY OAS). There is **no free source for the real series**, so
it was approximated with a **`-log(HYG/IEF)` credit proxy** and the +50%/50d cell
(where macro helped in `_260`) was re-run with the full **9-series** panel under
the identical champion regime + window. **Adding the credit axis did NOT help — it
slightly regressed the cell** (R-Precision@1 0.360 → 0.320). The credit proxy is
used by the model but at low importance; the breakeven + nominal-yield block still
dominates. The `_260` verdict stands: **macro does not beat the champion**, and
the credit axis is not the missing piece for this cell.

## Setup

- **HY-OAS gap:** `BAMLH0A0HYM2` (ICE BofA US HY Index OAS) is ICE-proprietary,
  redistributed free only by FRED (down). Confirmed unavailable elsewhere free:
  DBnomics returns 0 results, Nasdaq Data Link's legacy `ML/HYOAS` is now key-gated
  (403), ICE is paywalled. **Proxy used:** `-log(HYG_adjclose / IEF_adjclose)` —
  high-yield corp ETF vs duration-matched Treasuries (removes the rate component,
  isolates credit; rises with credit stress, tracking the OAS *direction*). Seeded
  under the FRED id with `yahoo_credit_proxy` provenance. Panel → **9 series / 45
  F17 columns**.
- **Re-run:** `sp500_up_50pct_50d_dd25pct_macrochamp9` — champion regime (trailing
  split, default auto loop, `max_iter 5`, n_iter=3), `--snapshot-end 2026-06-20`.
  Built the 9-series matrix from scratch (324 cols), **identical trailing test
  window 2026-01-22→2026-04-02, base_rate 0.0384, Q=50** as the `_260` 8-series run
  — directly comparable.

## Result (test segment; raw R-Precision@K + base rate 0.0384)

| K | champbase (no macro) | macrochamp 8-series (`_260`) | macrochamp9 9-series (+credit) | committed champion (ref) |
|---|---|---|---|---|
| R-Precision@1 | 0.1200 | **0.3600** | 0.3200 | 0.6400 |
| R-Precision@3 | 0.1600 | **0.3133** | 0.2733 | — |
| R-Precision@5 | 0.2040 | **0.3360** | 0.2600 | — |
| R-Precision@10 | 0.2165 | **0.3050** | 0.2908 | 0.3460 |
| R-Precision@20 | 0.3132 | 0.4153 | 0.4017 | — |
| test AUC | 0.8562 | 0.8644 | 0.8525 | — |

The 9-series panel is **worse than the 8-series at every K** (R-p@1 −0.04, R-p@10
−0.01). It still beats the no-macro champbase, but adding the credit proxy moved
it *away* from, not toward, the committed champion's 0.640.

## Credit-proxy feature usage

Total macro gain rose 0.195 → **0.222** (the model leaned on macro slightly more),
but the credit-proxy features rank **low**: `macro_BAMLH0A0HYM2_chg_60` is #33
(0.58% gain), the other four transforms #63/#95/#131/#143 (~1.7% combined). The
macro signal is still carried by **breakeven + nominal yield** —
`macro_T10YIE_chg_60` (#4), `macro_T10YIE_level` (#5), `macro_DGS10_chg_60` (#6),
`macro_VIXCLS_z_120` (#7). So the credit axis is consumed but adds mostly
low-value structure that, net, slightly hurt top-1 generalization.

## Verdict

- **Closing the HY-OAS gap did not help.** The credit proxy is a real, used signal
  but a minor one for this cell, and the net effect was a small regression. Credit
  spread is **not** the missing piece for +50%/50d.
- The `_260` conclusion is **unchanged**: macro — now with all 9 axes including
  credit — does not beat the deployed champion (still 0.320 vs the champion's
  0.640, with the same tuning-mode + window confounds). **Keep F17 opt-in; do not
  promote.**

## Caveats

- This is a **proxy**, not the real `BAMLH0A0HYM2`. The real ICE OAS could behave
  differently — but the proxy tracks OAS direction, so the weak result is
  suggestive that credit spread isn't the decisive missing signal here. If FRED
  egress returns, re-seed the real series and re-confirm.
- vs-champion comparison remains confounded (default-auto vs agent-tuned; different
  trailing window) — read the credit effect off the matched 8-series→9-series delta.

## Artifacts

- Spec: `configs/gbdt/experiments/sp500_up_50pct_50d_dd25pct_macrochamp9.yaml`
- Registry: `sp500_up_50pct_50d_dd25pct_macrochamp9` row in `results/gbdt/data/r_precision_at_k.csv`.
- Sidecar: `results/gbdt/data/_261_macro_hyoas_credit_data.json`.
- Credit proxy cached as `FRED:BAMLH0A0HYM2` (provider `yahoo_credit_proxy`, `-log(HYG/IEF)`).
