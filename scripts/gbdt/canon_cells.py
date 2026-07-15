"""Cell registry for the canonical-periods fine-tunes (#49-#54).

Single source of truth mapping a CELL id -> (universe, threshold_pct,
horizon_days, max_drawdown, feature-token, experiment-name stem). The canon
fine-tune scripts (fs_iterative_canon / hp_one_canon / final_fit_canon) import
`resolve()` so they are universe- and token-agnostic instead of hardcoding
sp500 + {50, 20}.

The canonical explicit-boundary split + de-biasing gate + snapshot are the same
for every cell (per CLAUDE.md canonical evaluation periods + the _285/_287
regime correction); only the (universe, target, feature-token) differ.
"""
from datetime import date

# Canonical explicit-boundary date_aligned split, shared by every canon cell.
SPLIT = dict(train_start=date(2015, 1, 1), val_start=date(2022, 3, 30),
             eval_start=date(2023, 7, 1), test_start=date(2024, 7, 1),
             test_end=date(2025, 6, 30))
SNAP = "2026-07-06"          # --snapshot-end for load_panel
MIN_ROWS_PER_TICKER = 2591   # _285/_287 de-biasing gate (kept ~2015-present tickers)

CELLS = {
    "50": dict(universe="sp500", thr=50, hor=50, dd=0.25, token="all",
               stem="sp500_up_50pct_50d_dd25pct"),
    "20": dict(universe="sp500", thr=20, hor=25, dd=0.10, token="all",
               stem="sp500_up_20pct_25d_dd10pct"),
    "nasdaq_40_50": dict(universe="nasdaq100", thr=40, hor=50, dd=0.20, token="all",
                         stem="nasdaq100_up_40pct_50d_dd20pct"),
    "russell_40_100": dict(universe="russell1000", thr=40, hor=100, dd=0.20, token="all",
                           stem="russell1000_up_40pct_100d_dd20pct"),
    "russell_50_200": dict(universe="russell1000", thr=50, hor=200, dd=0.25, token="all",
                           stem="russell1000_up_50pct_200d_dd25pct"),
    "sp500_40_200_f18": dict(universe="sp500", thr=40, hor=200, dd=0.20,
                             token="all_fundamentals",
                             stem="sp500_up_40pct_200d_dd20pct_f18"),
    # V1.10 nifty500 canonical-scan finetunes (task #55): the strongest fund win
    # (50%/200d ffund, +0.341 test R-p@1) + the top technical champion (30%/50d
    # fbase, AUC 0.669). NSE calendar + all_*_calendar2 tokens flow through resolve().
    "nifty_50_200_ffund": dict(universe="nifty500", thr=50, hor=200, dd=0.25,
                               token="all_fundamentals_calendar2",
                               stem="nifty500_up_50pct_200d_dd25pct_ffund"),
    "nifty_30_50_fbase": dict(universe="nifty500", thr=30, hor=50, dd=0.15,
                              token="all_calendar2",
                              stem="nifty500_up_30pct_50d_dd15pct_fbase"),
    # V1.11 (task #60): the two deploy-shaped every-K fund wins from the _289 scan
    # (common base rate + ~+0.12 test R-p@1) — the natural next finetunes after the
    # rare 50%/200d. Both all_fundamentals_calendar2 (F18-IN de-confounded).
    "nifty_10_25_ffund": dict(universe="nifty500", thr=10, hor=25, dd=0.05,
                              token="all_fundamentals_calendar2",
                              stem="nifty500_up_10pct_25d_dd5pct_ffund"),
    "nifty_20_100_ffund": dict(universe="nifty500", thr=20, hor=100, dd=0.10,
                               token="all_fundamentals_calendar2",
                               stem="nifty500_up_20pct_100d_dd10pct_ffund"),
    # #56 PILOT (champion-matrix row 1): the deployed +20%/25d/dd10% champion cell
    # replicated across the other universes, matched token `all` (same as the sp500
    # champion) so the universe is the only variable. Transfer test, not a champion swap.
    "nasdaq_20_25": dict(universe="nasdaq100", thr=20, hor=25, dd=0.10, token="all",
                         stem="nasdaq100_up_20pct_25d_dd10pct"),
    "russell_20_25": dict(universe="russell1000", thr=20, hor=25, dd=0.10, token="all",
                          stem="russell1000_up_20pct_25d_dd10pct"),
    "nifty_20_25": dict(universe="nifty500", thr=20, hor=25, dd=0.10, token="all",
                        stem="nifty500_up_20pct_25d_dd10pct"),
}


def resolve(cell):
    """Return the cell dict augmented with base/ft experiment names."""
    if cell not in CELLS:
        raise SystemExit(f"unknown CELL={cell!r}; known: {sorted(CELLS)}")
    c = dict(CELLS[cell])
    c["cell"] = cell
    c["base"] = f"{c['stem']}_canon_base"
    c["ft"] = f"{c['stem']}_canon_ft"
    return c
