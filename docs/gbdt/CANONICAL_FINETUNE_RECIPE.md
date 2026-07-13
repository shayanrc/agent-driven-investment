# Canonical-Periods Fine-Tune — agent recipe (v2, updated through #52)

Step-by-step procedure for fine-tuning a gbdt cell on the canonical evaluation periods.
Supersedes the v1 recipe. Worked examples #49–#52 + mistake-guards at the end.

## 0. Conventions (fixed)
- **Windows** (explicit-boundary `date_aligned`): train 2015-01-01 · val_start 2022-03-30 ·
  eval_start 2023-07-01 · test_start 2024-07-01 · test_end 2025-06-30. Backtest 2025-07→2026-06.
- **Role discipline:** train = fit; **val = feature selection + early stopping**;
  **eval = HP tuning**; **test = final evaluation ONLY** (never fit/select on it).
- Every fit: XGBoost, 500 trees, early stopping **explicitly on val AUC**
  (`EarlyStopping(metric_name="auc", data_name="val", maximize=True)`), uniqueness weights.
- **Judge top-K cells by the test BOOK — R-p@K — NOT test AUC.** (FS-pruned models had
  higher AUC but a worse book; #50.)
- Tooling: `scripts/gbdt/{fs_iterative_canon,hp_one_canon,final_fit_canon}.py`, all
  cell-parameterized via `scripts/gbdt/canon_cells.py` (`CELL=<id>`, `FEATS=all|rN|locked|csv`,
  `HP=<json>`). Prep cache makes each hp_one fit ~15–60 s.

## 1. Base matrix + the CONTROLLED baseline (the bar)
- Build the base matrix: `python -m gbdt experiment <cell>_canon_base.yaml` (fast via the
  universe feature cache; ~40 s sp500/nasdaq, ~11 min russell cold).
- **DO NOT use the runner's `metrics.json` as the baseline bar.** The default loop applies
  matched-sweep HP + calibration and on rare cells sits at a weak/degenerate point — it is a
  DIFFERENT, usually weaker model. (nasdaq: runner rp@3 0.379 vs proper 0.462; sp500_50:
  runner rp@3 0.262 vs proper 0.323 — comparing to the runner metric made c9 look like a
  clean win when it isn't.)
- **The bar = `final_fit CELL=<id> FEATS=all HP={d6,mcw1,ss1,cs1,g0,eta0.05}`** on test —
  the same code path as the FT candidates, differing only in HP. Record its full R-p@K.

## 2. Diagnose the cell BEFORE tuning
- **Prevalence** (build_target reports it): rare (<~4%) vs common (≥~5%).
- **eval↔test agreement:** compare the baseline's eval R-p@3 to its test R-p@3. If they
  invert (e.g. weak eval, strong test — happens on every cell so far), **eval is an
  unreliable HP oracle** → do NOT trust eval-greedy choices; select on **val**, confirm on test.
- **Base top-of-book headroom:** is baseline test R-p@1 low (room) or already high (maxed)?

## 3. Pick the tuning path from the diagnosis
- **COMMON event + low base @1 → deep trees + row/col bagging on the FULL feature set.**
  Sweep depth {6,8,10} × subsample {0.7,0.85} × colsample {0.7,1.0}, mcw=1, gamma=0.
  Bagging strength is itself a knob (russell: ss0.7+cs0.7 > ss0.85). This is the winner for
  #50 and #52 (beat baseline at EVERY K, decisively at @1).
- **RARE event + high base @1 → the default model is usually the book champion.** Regularized
  FTs (mcw10, deep+bagging) trade the book for @1. Try mcw {5,10} + light bagging, but EXPECT
  the base to win @3–@20; only adopt an FT if it beats the baseline BOOK, not just @1. (#49
  base beat c9 at @3–@20; #51 base stood.)
- **Feature selection:** run the FS trajectory for the record, but it is usually **inert at
  default HP** (the greedy fit ignores redundant features — rounds 279→~90 come out identical).
  Do NOT let eval drive the feature count down (that anti-selects the test book — the #50
  attempt-1 error). Count is a val/FS-role decision; when in doubt keep all 279.
- **eta is a dead knob** (rank-invariant for R-p@K/AUC); **gamma usually hurts** — leave both.

## 4. Select on val, confirm on test (minimize test looks)
- Run candidates through `hp_one` (val+eval). **Select the FT by val** (val@1/@3 + val book),
  because eval is unreliable. val correctly predicted the winners in #50/#52 and the @1
  sacrifice in #51.
- Take the 1–2 val-best configs to `final_fit` on test. Compare to the controlled baseline's
  full R-p@K. **Adopt the FT only if it beats the baseline BOOK (R-p@3–@20), not just @1 or AUC.**
- When several test configs were viewed, note the multiple-comparison risk. **Model selection
  ENDS on `test`** — do NOT defer the config choice to the backtest window; per the canonical role
  discipline the backtest is never-touched by model selection, and using it to pick a config
  contaminates the only clean forward-OOS window. The backtest is a downstream strategy/deploy
  check only (§6).

## 5. Save + record
- `final_fit` saves model.pkl + predictions/{val,eval,test,backtest}.csv + final_summary.json
  + spec.yaml + metrics.json (the last three are backtest scaffolding for run_fresh_oos).
- Write `hp/EXPLORATION.md`: windows/prevalence, the controlled baseline, the path tried,
  the test table, the verdict.

## 6. Backtest — strategy evaluation + the deploy cut (NOT a model-selection arbiter)
- `scripts.backtests.run_fresh_oos --cell <ft_dir> --predictions <ft_dir>/predictions/backtest.csv
   --out <o> --name <n> --selection-mode rank --rank-by raw --sizing-mode equal`. The ft dir needs
  `spec.yaml` (horizon) + `predictions/val.csv` (calibrator) + `metrics.json` — final_fit now writes them.
- **Use `--sizing-mode equal`, not `rank_kelly`.** The Kelly gate sets the per-pick win prob to
  the cell's eval R-p@K, which measures P(threshold move) NOT P(+10%/-5% exit) — it under-shoots
  breakeven (0.333) and zeroes out all trades. Equal-weight top-K is the fair signal backtest.
- **test R-p@K does NOT perfectly predict backtest return** (russell_40_100's FT won every K on
  test but lagged the backtest, +21% vs NDX +32%) — different regimes. That is a reason for humility,
  NOT a licence to pick the model on the backtest: per the canonical role discipline the backtest is
  **never touched by model selection**. Use it for the **deploy cut** (target-hits > DD-stops, a
  strategy-simulation metric) and a sanity read — the model config is already fixed by `test`.
  Backtesting both the chosen and alternative config is fine for *understanding*, but the config
  choice stays on the test book (#49's baseline-over-c9 is decided on test, not its +156.4%).
- Gross only (no costs — downstream). Benchmark is ^NDX in the harness for all cells (a reference).

## Worked examples
| cell | prev | base@1 | winner | test R-p@1/3/5/10/20 (winner) | vs baseline |
|---|---|---|---|---|---|
| #49 sp500 +50%/50d | 0.9% | 0.311 | baseline all/d6 (test-book winner) | 0.311/0.323/0.338/0.406/0.484 | base wins the book; c9 (sp500_50_c9) wins @1 only |
| #50 sp500 +20%/25d | 4.8% | 0.253 | 279f·d8·ss0.85 | 0.321/0.313/0.311/0.299/0.330 | **FT wins EVERY K** |
| #51 nasdaq +40%/50d | 3.0% | 0.564 | baseline all/d6 | 0.564/0.462/0.500/0.719/0.956 | base stands (FT trades @1) |
| #52 russell +40%/100d | 8% | 0.208 | 279f·d8·ss0.7·cs0.7 | 0.332/0.351/0.396/0.362/0.381 | **FT wins EVERY K** |
| #53 russell +50%/200d | 12% | 0.516 | 279f·d8·ss0.7·cs0.7 | 0.396/0.511/0.526/0.459/0.408 | FT wins @3–@20; loses @1 |
| #54 sp500 +40%/200d F18 | 17% | 0.628 | baseline all/d6 (292f incl F18) | 0.628/0.517/0.505/0.454/0.395 | base wins EVERY K |

## The real determinant: base @1 headroom (refines "rare vs common")
Deep+bagging **redistributes probability mass from the spiky @1 toward the deeper book.**
Whether that's a NET win depends on the base's @1 headroom, NOT prevalence alone:
- base @1 LOW (~0.21–0.25: sp500_20, russell_40_100) → FT wins EVERY K. Adopt.
- base @1 HIGH (~0.56–0.63: nasdaq, sp500_F18) → the sharp top is already maxed; bagging
  only dilutes it → base wins. Keep base.
- base @1 MID (~0.31–0.52) → a trade: FT lifts @3–@20, loses @1 (#53, and #49 in reverse via
  mcw). Judgment call **on test** — adopt the FT for a top-K (K>1) strategy where the book matters
  (its test book wins); keep base if @1 concentration is what the strategy sizes on. Decide on the
  test book, NOT the backtest window.
Prevalence correlates (common cells tend to have low base @1) but is only a proxy; the base
@1 you measure in Step 1 is the real signal. Always TRY deep+bagging on common cells, but
confirm on test at every K before adopting.

## Mistakes made (guard against these)
1. **Compared to the runner's `metrics.json` baseline** (weaker/different model) → a fine-tune
   looked like a clean win when the controlled baseline actually beat it (#49 c9). ALWAYS use
   the `final_fit all/d6` control.
2. **Let eval drive the feature count** down to 12f (eval-AUC peak) → anti-selected the test
   book (#50 attempt-1). Count is a val decision; eval is unreliable on these cells.
3. **Judged by AUC / eval, not the test book** → the eval-greedy and higher-AUC models had
   worse books. Judge by test R-p@K.
4. **Forgot `FEATS=all` handling / dir-creation** in the scripts → small bugs; fixed. When
   generalizing scripts, syntax-check + a back-compat resolve() check before running.
5. **Re-fitting overwrote the saved winner** (final_fit clobbers the ft dir each run) → always
   RE-SAVE the chosen config last, and capture intermediate test numbers from stdout/logs.
6. **Ran two heavy `final_fit`s in parallel → OOM-kill** (38 GB box; russell d8-deep ~15 GB +
   sp500 fit ~13 GB, both peaking = kill; the killed run left the STALE baseline summary, which
   read back as a fake "identical" result). Serialize large fits; a SIGKILL leaves no traceback
   and a stale summary — verify the saved `hp`/`best_iter` matches what you launched. Watch
   `/proc/meminfo MemAvailable` (not `free -m` — the shell alias errors); keep >2 GB headroom.
7. **Long-horizon test labels need the cache seeded forward.** A 200d cell's test_end
   (2025-06-30) needs prices to ~2026-04-15. The russell cache was stale at 2025-11-19 →
   seed `data_pipelines seed --domain us_equities --universe <u> --start <> --end 2026-07-06`
   first (idempotent). After seeding, verify the base build's preflight "covered through" date
   and the final_fit test row count (815×250 = fully labeled).
