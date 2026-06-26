"""Regenerate the canonical back-test performance registry CSV.

The back-test analog of ``scripts/gbdt/regenerate_r_precision_at_k_csv.py``:
produces ``results/backtests/data/backtest_summary.csv`` — one row per back-test
run, in the extended schema documented in ``docs/backtests/CONVENTIONS.md``
(§ "Registry CSV schema"). Rows come from two sources:

  1. **Run dirs (authoritative).** Per-run artifact dirs holding ``summary.json``
     + ``equity_curve.csv`` [+ ``picks.csv``] written by ``run_backtest_cell.py``.
     Fully (re)built from artifacts: risk metrics from the equity curve, exposure
     /turnover/exit-triggers/benchmark identity from ``summary.json``. The 10
     ``_024`` cells live here (manifest ``_024_RUNS``); future runs append.

  2. **Legacy carry.** The pre-artifact ``_001``–``_023`` curated rows, carried
     from the existing CSV and schema-migrated: the ``*_ndx_bh`` → ``idx_bh_*``
     rename (those columns held each universe's *actual* index — SPX/NIFTY — under
     an NDX name; verified mislabel), ``benchmark_index`` + ``excess_return_total``
     + ``calmar_strategy`` + parsed strategy-config columns derived from the
     existing data, and every run-level column (risk/exposure/turnover/
     faithfulness) left NaN. **Append-only** per CONVENTIONS — old rows are never
     rewritten beyond the approved rename + derivable additions.

Idempotent: re-running reads the (already-migrated) CSV for legacy rows and
rebuilds the run-dir rows from artifacts, so artifacts are the source of truth
for everything they cover. Default sort is chronological (run_timestamp, id).

    uv run python -m scripts.backtests.regenerate_backtest_performance_csv [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "results/backtests/data/backtest_summary.csv"
RUN_DIR_024 = REPO / "results/backtests/_024_oos_top10"
RP_REGISTRY = REPO / "results/gbdt/data/r_precision_at_k.csv"
TRADING_DAYS = 252.0

# Benchmark index by universe (mirrors run_backtest_cell.INDEX_BY_UNIVERSE) — used
# to set benchmark_index on legacy rows that predate the geometry.index field.
_BENCH_BY_UNIVERSE = {
    "nasdaq100": ("NDX", False, ""),
    "sp500": ("SPX", False, ""),
    "russell1000": ("SPX", True, "^RUI"),
    "nifty500": ("NIFTY500", False, ""),
    "nifty50": ("NIFTY50", False, ""),
}

# --- _024 run manifest: (cell, short_name, selfcheck_status, max_abs_diff, universe_delta)
# faithfulness parsed from the OOS batch logs (/tmp/batch_oos_bt.log, batch_oom2.log,
# test_v14p1_chunked.log). russell cells reproduce exactly (PASS, no membership change);
# sp500 cells warn on +9 recent listings (CRWD/DDOG/UBER/MRNA/CTVA/DOW/FOX/FOXA/VRT).
# sp500_50_cbagent's exact diff was not retained in a log → NaN, status known.
_024_RUNS = [
    ("russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1", "024_r1k_40pct_100d_v14p1",  "PASS",                 1.49e-08, "+0/-0"),
    ("russell1000_up_40pct_200d_dd20pct_aligned_cbagent",     "024_r1k_40pct_200d_cbagent", "PASS",                9.71e-17, "+0/-0"),
    ("russell1000_up_50pct_200d_dd25pct_aligned_agent",       "024_r1k_50pct_200d_agent",   "PASS",                2.98e-08, "+0/-0"),
    ("russell1000_up_40pct_200d_dd20pct_aligned_agent",       "024_r1k_40pct_200d_agent",   "PASS",                2.97e-08, "+0/-0"),
    ("russell1000_up_50pct_200d_dd25pct_aligned_cbagent",     "024_r1k_50pct_200d_cbagent", "PASS",                9.71e-17, "+0/-0"),
    ("sp500_up_50pct_200d_dd25pct_aligned_cbagent",           "024_sp500_50pct_200d_cbagent", "WARN_UNIVERSE_GROWTH", float("nan"), "+9/-0"),
    ("sp500_up_50pct_200d_dd25pct_aligned_agent",             "024_sp500_50pct_200d_agent",   "WARN_UNIVERSE_GROWTH", 4.78e-02, "+9/-0"),
    ("russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1", "024_r1k_50pct_200d_v14p1",   "PASS",                2.98e-08, "+0/-0"),
    ("sp500_up_40pct_200d_dd20pct_aligned_cbagent",           "024_sp500_40pct_200d_cbagent", "WARN_UNIVERSE_GROWTH", 4.37e-02, "+9/-0"),
    ("sp500_up_40pct_200d_dd20pct_aligned_agent",             "024_sp500_40pct_200d_agent",   "WARN_UNIVERSE_GROWTH", 7.15e-02, "+9/-0"),
]
_024_COMMIT = "3d27c380c1d4f33d50bb9033800b61b52bea78ca"  # origin/main tip where _024 landed
_024_RUN_TS = "2026-06-24"

# Canonical column order. The first 39 (with the 4 *_ndx_bh → idx_bh_* renames) are
# the legacy schema; the rest are the leaderboard extension, grouped by purpose.
LEGACY_COLS = [
    "id", "name", "prediction_source", "prediction_module", "cell_or_preset",
    "model_artifact_path", "calibrator_class", "sizer_class", "sizer_c", "strategy_class",
    "K", "oos_start", "oos_end", "n_picks", "n_unique_tickers", "initial_cash",
    "final_equity_strategy", "total_return_strategy", "cagr_strategy", "max_dd_strategy",
    "sharpe_strategy_gross",
    "idx_bh_final_equity", "idx_bh_total_return", "idx_bh_cagr", "idx_bh_max_dd",
    "final_equity_ew_basket", "total_return_ew_basket", "cagr_ew_basket", "max_dd_ew_basket",
    "final_equity_ew_topk_no_kelly", "total_return_ew_topk_no_kelly",
    "cagr_ew_topk_no_kelly", "max_dd_ew_topk_no_kelly",
    "r_precision_at_1_realized", "r_precision_at_3_realized", "r_precision_at_5_realized",
    "commit_sha", "run_timestamp", "notes",
]
NEW_COLS = [
    # identity / provenance
    "experiment", "runner",
    # benchmark identity + relative
    "benchmark_index", "benchmark_is_proxy", "benchmark_proxy_for", "excess_return_total",
    # dates
    "train_start", "comparison_end", "data_vintage", "n_days", "mtm_truncated",
    # strategy config
    "selection_mode", "sizing_mode", "selection_bound", "rank_by",
    "regime_gate", "cost_bps", "prob_weight_alpha", "vol_window",
    # risk (active-window, from equity_curve.csv)
    "calmar_strategy", "vol_annual_strategy", "sortino_strategy",
    "ulcer_index_strategy", "worst_day_strategy", "sharpe_active_strategy",
    # exposure & turnover
    "avg_gross_exposure", "pct_days_invested", "n_entries", "n_exits", "n_trims",
    "open_at_end", "n_exit_target", "n_exit_dd", "n_exit_horizon",
    # data vintage / faithfulness
    "universe_n", "selfcheck_status", "selfcheck_max_abs_diff", "universe_delta",
    # quality
    "caveat",
    # deployment status
    "daily_preds", "comment",
]
COLUMN_ORDER = LEGACY_COLS + NEW_COLS

_NDX_RENAME = {
    "final_equity_ndx_bh": "idx_bh_final_equity",
    "total_return_ndx_bh": "idx_bh_total_return",
    "cagr_ndx_bh": "idx_bh_cagr",
    "max_dd_ndx_bh": "idx_bh_max_dd",
}

# benchmark_index → cache ticker (mirrors run_backtest_cell.INDEX_BY_UNIVERSE).
_BENCH_TICKER = {"NDX": "INDEX:^NDX", "SPX": "INDEX:^SPX",
                 "NIFTY500": "NIFTY:500", "NIFTY50": "NIFTY:50"}

# Legacy rows whose `idx_bh_*` are authoritative — carried from the original curated
# CSV, computed by the original run over its exact (now-unrecorded) `comparison_end`.
# They are never recomputed and get no reconstructed `comparison_end` (we won't claim
# a window we can't reproduce exactly — a recompute matches only to ~2e-5, since the
# curated values are 4-dp-rounded). All OTHER legacy rows' `idx_bh_*` are
# reconstruction-based and get a pinned `comparison_end` (frozen → reproducible).
_AUTHORITATIVE_LEGACY_IDS = frozenset(range(1, 10))

# Cells deployed to the /daily-predictions forward-OOS cadence (the two validated
# sp500 champions, scored daily with the SMA200 regime gate). Sets the `daily_preds`
# flag (per model — all of a cell's rows) + seeds the `comment` column. See
# docs/backtests/_019_forward_oos_pipeline.md.
_DEPLOYED_CELLS = frozenset({
    "sp500_up_50pct_50d_dd25pct_agentloop",
    "sp500_up_20pct_25d_dd10pct_agentloop",
})
_DEPLOY_COMMENT = "Deployed to /daily-predictions (SMA200 regime gate)"
_UNDEPLOYED_COMMENT = "Not deployed to /daily-predictions"


def _universe_of(cell_or_preset: str) -> str:
    s = str(cell_or_preset)
    for u in ("nasdaq100", "sp500", "russell1000", "nifty500", "nifty50"):
        if u in s:
            return u
    return ""


def _clean_experiment(cell_or_preset: str) -> str:
    """Strip the descriptive ' (published test window)'/'(inferred fresh-OOS)' suffix
    so the value is the bare cell name — the join key into r_precision_at_k.csv."""
    return re.sub(r"\s*\(.*\)\s*$", "", str(cell_or_preset)).strip()


def _parse_benchmark_label(idx_label: str) -> tuple[str, bool, str]:
    """geometry.index → (index, is_proxy, proxy_for). 'SPX (proxy: ^RUI uncached)'
    → ('SPX', True, '^RUI'); 'SPX' → ('SPX', False, '')."""
    m = re.match(r"^([A-Za-z0-9]+)\s*(?:\(proxy:\s*(\S+).*\))?$", str(idx_label).strip())
    if not m:
        return str(idx_label), False, ""
    base, proxy = m.group(1), m.group(2)
    return base, proxy is not None, (proxy or "")


def _parse_config(strategy_class: str, sizer_class: str) -> dict:
    """Decompose the free-text strategy_class/sizer_class into structured columns.
    One-time legacy parse (never used on artifact rows, which read summary['config'])."""
    sc, sz = str(strategy_class), str(sizer_class)
    blob = f"{sc} {sz}"
    selection = "rank" if ("rank" in sc or "selection_mode=rank" in sc) else "breakeven"
    if "prob_weight" in blob:
        sizing = "prob_weight"
    elif "Kelly" in sz or "kelly" in sz:
        sizing = "kelly"
    elif "equal" in blob:
        sizing = "equal"
    else:
        sizing = ""
    regime = "none"
    if (m := re.search(r"regime_ma=(\d+)", sz)):
        regime = f"SMA{m.group(1)}"
    elif (m := re.search(r"regime=(\w+)", sz)):
        regime = m.group(1)
    elif "SMA-gate" in sc:
        regime = "SMA"
    elif (m := re.search(r"\+(\w+)-gate", sc)):
        regime = m.group(1)
    cost = 0.0
    if (m := re.search(r"cost=(\d+)bps", sz)):
        cost = float(m.group(1))
    alpha = np.nan
    if (m := re.search(r"α=(\d+)", sz)):
        alpha = float(m.group(1))
    elif "prob_weight" in sizing:
        alpha = 1.0
    return {"selection_mode": selection, "sizing_mode": sizing, "regime_gate": regime,
            "cost_bps": cost, "prob_weight_alpha": alpha}


def _risk_metrics(equity_csv: Path, active_start: pd.Timestamp | None) -> dict:
    """Risk metrics from the daily equity curve, restricted to the active window
    (>= test_start) so the flat pre-buffer doesn't deflate vol / distort Sharpe."""
    eq = pd.read_csv(equity_csv, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    if active_start is not None:
        eq = eq[eq.index >= active_start]
    r = eq.pct_change().dropna()
    n_days = int(len(eq))
    sd = float(r.std(ddof=1)) if len(r) > 1 else np.nan
    down = r[r < 0]
    dd = eq / eq.cummax() - 1.0
    return {
        "n_days": n_days,
        "vol_annual_strategy": sd * np.sqrt(TRADING_DAYS) if sd == sd else np.nan,
        "sharpe_active_strategy": (float(r.mean()) / sd * np.sqrt(TRADING_DAYS)
                                   if sd and sd == sd and sd > 0 else np.nan),
        "sortino_strategy": (float(r.mean()) / float(down.std(ddof=1)) * np.sqrt(TRADING_DAYS)
                             if len(down) > 1 and down.std(ddof=1) > 0 else np.nan),
        "ulcer_index_strategy": float(np.sqrt((dd ** 2).mean())) if len(dd) else np.nan,
        "worst_day_strategy": float(r.min()) if len(r) else np.nan,
    }


def _pct_days_invested(picks_csv: Path, eq_index: pd.DatetimeIndex,
                       end_date: pd.Timestamp) -> float:
    """Fraction of active-window trading days with ≥1 open position, reconstructed
    from picks.csv entry/exit events (trims don't open/close; open-at-end runs to
    end_date)."""
    if not picks_csv.exists() or len(eq_index) == 0:
        return np.nan
    pk = pd.read_csv(picks_csv, parse_dates=["date"])
    pk = pk[pk["kind"].isin(["entry", "exit"])].sort_values(["ticker", "step"])
    intervals, open_start = [], {}
    for _, e in pk.iterrows():
        t, k, d = e["ticker"], e["kind"], e["date"]
        if k == "entry":
            open_start.setdefault(t, d)
        elif k == "exit" and t in open_start:
            intervals.append((open_start.pop(t), d))
    for t, d0 in open_start.items():
        intervals.append((d0, end_date))
    if not intervals:
        return 0.0
    invested = pd.Series(False, index=eq_index)
    for a, b in intervals:
        invested |= (eq_index >= a) & (eq_index <= b)
    return float(invested.mean())


def _mtm_truncated(test_end: pd.Timestamp, comparison_end: pd.Timestamp,
                   horizon: int) -> bool:
    """True if the mark-to-market end was clipped before the full horizon resolved."""
    full = pd.bdate_range(test_end, periods=horizon + 1)[-1]
    return bool(comparison_end < full)


def row_from_run(run_dir: Path, *, id: int, name: str, selfcheck_status: str,
                 max_abs_diff: float, universe_delta: str,
                 rp_lookup: dict) -> dict:
    """Build a full-schema row from a run dir's artifacts (summary.json +
    equity_curve.csv + picks.csv)."""
    s = json.loads((run_dir / "summary.json").read_text())
    cfg, geo, win = s["config"], s["geometry"], s["window"]
    strat, bench = s["strategy"], s.get("benchmarks", {})
    turn, trig = s.get("turnover", {}), s.get("exit_triggers", {})
    cell = s["cell"]
    idx_bh = bench.get("index_bh", {})
    ewb, ewt = bench.get("ew_basket", {}), bench.get("ew_topk_no_kelly", {})
    bidx, is_proxy, proxy_for = _parse_benchmark_label(geo.get("index", ""))

    test_start = pd.Timestamp(win["test_start"])
    test_end = pd.Timestamp(win["test_end"])
    comparison_end = pd.Timestamp(win["comparison_end"])
    risk = _risk_metrics(run_dir / "equity_curve.csv", test_start)
    pct_inv = _pct_days_invested(run_dir / "picks.csv",
                                 _active_index(run_dir / "equity_curve.csv", test_start),
                                 comparison_end)
    total_ret = strat.get("total_return")
    idx_ret = idx_bh.get("total_return")
    max_dd = strat.get("max_dd")
    cagr = strat.get("cagr")
    rp = rp_lookup.get(_clean_experiment(cell), {})

    return {
        "id": id, "name": name, "prediction_source": "gbdt", "prediction_module": "gbdt",
        "cell_or_preset": cell, "experiment": _clean_experiment(cell),
        "model_artifact_path": f"results/gbdt/experiments/{cell}/",
        "calibrator_class": "BetaBinomialBucketed", "runner": "run_backtest_cell",
        "sizer_class": _sizer_label(cfg), "sizer_c": cfg.get("fractional_c"),
        "strategy_class": _strategy_label(cfg),
        "selection_mode": cfg.get("selection_mode"), "sizing_mode": cfg.get("sizing_mode"),
        "selection_bound": cfg.get("selection_bound"), "rank_by": cfg.get("rank_by"),
        "K": cfg.get("K"), "regime_gate": "none", "cost_bps": 0.0,
        "prob_weight_alpha": cfg.get("prob_weight_alpha"), "vol_window": cfg.get("vol_window"),
        "train_start": rp.get("train_start"),
        "oos_start": win["test_start"], "oos_end": win["test_end"],
        "comparison_end": win["comparison_end"], "data_vintage": win["data_end"],
        "n_days": risk["n_days"],
        "mtm_truncated": _mtm_truncated(test_end, comparison_end, int(geo.get("horizon", 0))),
        "n_picks": np.nan, "n_unique_tickers": turn.get("unique_tickers"),
        "n_entries": turn.get("entries"), "n_exits": turn.get("exits"),
        "n_trims": turn.get("trims"), "open_at_end": turn.get("open_at_end"),
        "n_exit_target": trig.get("target", 0), "n_exit_dd": trig.get("DD", 0),
        "n_exit_horizon": trig.get("horizon", 0),
        "initial_cash": strat.get("start"),
        "final_equity_strategy": strat.get("end"), "total_return_strategy": total_ret,
        "cagr_strategy": cagr, "max_dd_strategy": max_dd,
        "sharpe_strategy_gross": np.nan,
        "calmar_strategy": (cagr / abs(max_dd) if max_dd else np.nan),
        "vol_annual_strategy": risk["vol_annual_strategy"],
        "sortino_strategy": risk["sortino_strategy"],
        "ulcer_index_strategy": risk["ulcer_index_strategy"],
        "worst_day_strategy": risk["worst_day_strategy"],
        "sharpe_active_strategy": risk["sharpe_active_strategy"],
        "avg_gross_exposure": strat.get("gross_exposure_avg"), "pct_days_invested": pct_inv,
        "benchmark_index": bidx, "benchmark_is_proxy": is_proxy,
        "benchmark_proxy_for": proxy_for,
        "idx_bh_final_equity": idx_bh.get("end"), "idx_bh_total_return": idx_ret,
        "idx_bh_cagr": idx_bh.get("cagr"), "idx_bh_max_dd": idx_bh.get("max_dd"),
        "excess_return_total": (total_ret - idx_ret
                                if total_ret is not None and idx_ret is not None else np.nan),
        "final_equity_ew_basket": ewb.get("end"), "total_return_ew_basket": ewb.get("total_return"),
        "cagr_ew_basket": ewb.get("cagr"), "max_dd_ew_basket": ewb.get("max_dd"),
        "final_equity_ew_topk_no_kelly": ewt.get("end"),
        "total_return_ew_topk_no_kelly": ewt.get("total_return"),
        "cagr_ew_topk_no_kelly": ewt.get("cagr"), "max_dd_ew_topk_no_kelly": ewt.get("max_dd"),
        "r_precision_at_1_realized": np.nan, "r_precision_at_3_realized": np.nan,
        "r_precision_at_5_realized": np.nan,
        "universe_n": np.nan,
        "selfcheck_status": selfcheck_status, "selfcheck_max_abs_diff": max_abs_diff,
        "universe_delta": universe_delta,
        "caveat": bool(risk["n_days"] < 120 or (strat.get("gross_exposure_avg") or 0) < 0.4),
        "commit_sha": _024_COMMIT, "run_timestamp": _024_RUN_TS,
        "notes": ("OOS test_end+1→2026-06-01, mark-to-market@data_end (horizon-truncated → "
                  "realized R-p NaN); _024 top-10."
                  + (" exact self-check diff not retained (status known)."
                     if max_abs_diff != max_abs_diff else "")),
    }


def _active_index(equity_csv: Path, start: pd.Timestamp) -> pd.DatetimeIndex:
    eq = pd.read_csv(equity_csv, index_col=0, parse_dates=True)
    return eq.index[eq.index >= start]


def _sizer_label(cfg: dict) -> str:
    sm = cfg.get("sizing_mode")
    return {"equal": "rank/equal-weight", "kelly": "DiscreteBoundedLossKelly",
            "prob_weight": "prob_weight (bet∝p)"}.get(sm, sm or "")


def _strategy_label(cfg: dict) -> str:
    return f"TopKDailyKellyLabelExit({cfg.get('selection_mode')},{cfg.get('sizing_mode')})"


def migrate_legacy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Schema-migrate the legacy curated rows: the approved *_ndx_bh → idx_bh_*
    rename + derivable new columns (benchmark identity, excess return, calmar,
    parsed config, clean experiment). Run-level columns stay NaN. Idempotent."""
    df = df.rename(columns={k: v for k, v in _NDX_RENAME.items() if k in df.columns})
    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        uni = _universe_of(d.get("cell_or_preset", ""))
        bidx, is_proxy, proxy_for = _BENCH_BY_UNIVERSE.get(uni, ("", False, ""))
        d.setdefault("experiment", _clean_experiment(d.get("cell_or_preset", "")))
        d["experiment"] = _clean_experiment(d.get("cell_or_preset", ""))
        d["runner"] = d.get("runner") if isinstance(d.get("runner"), str) else "legacy_survey"
        d["benchmark_index"] = bidx
        d["benchmark_is_proxy"] = is_proxy
        d["benchmark_proxy_for"] = proxy_for
        tr, ir = d.get("total_return_strategy"), d.get("idx_bh_total_return")
        d["excess_return_total"] = (tr - ir if pd.notna(tr) and pd.notna(ir) else np.nan)
        cg, mdd = d.get("cagr_strategy"), d.get("max_dd_strategy")
        d["calmar_strategy"] = (cg / abs(mdd) if pd.notna(cg) and pd.notna(mdd) and mdd else np.nan)
        cfg = _parse_config(d.get("strategy_class", ""), d.get("sizer_class", ""))
        for k, v in cfg.items():
            d[k] = v
        rows.append(d)
    return pd.DataFrame(rows)


def _backfill_index_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Pin + freeze legacy index benchmark metrics. For each legacy row with no
    stored `comparison_end` (and not an authoritative original), reconstruct the
    benchmark window `comparison_end = min(test_end + horizon bdays, cache data-end)`
    — the same window `run_backtest_cell` benchmarks on — and STORE it, so later
    regens read the pinned window and reproduce the values byte-for-byte (no
    cache-vintage drift). `idx_bh_*` fields are filled only where NULL (existing
    values, incl. authoritative index returns, are never overwritten); for the
    already-backfilled rows that just means `comparison_end` is now persisted next
    to its values. Rows in `_AUTHORITATIVE_LEGACY_IDS`, rows already carrying a
    `comparison_end`, and rows without a resolvable index/horizon are left untouched.
    Cache-only (`_cache_read`) — no network."""
    mask = (df["comparison_end"].isna() & ~df["id"].isin(_AUTHORITATIVE_LEGACY_IDS)
            & df["benchmark_index"].isin(_BENCH_TICKER)
            & df["oos_start"].notna() & df["oos_end"].notna())
    if not mask.any():
        return df, 0
    try:
        from scripts.backtests import benchmarks as bm
        from scripts.backtests.run_cell5_bayesian_kelly import INITIAL_CASH, _load_closes
    except Exception:
        return df, 0
    need = df[mask]
    lo = pd.to_datetime(need.oos_start).min() - pd.Timedelta(days=20)
    hi = pd.to_datetime(need.oos_end).max() + pd.Timedelta(days=600)
    closes = {}
    for bidx in need.benchmark_index.unique():
        s = _load_closes([_BENCH_TICKER[bidx]], lo, hi).get(_BENCH_TICKER[bidx])
        if s is not None and len(s):
            closes[bidx] = s.sort_index()
    n = 0
    for i, r in need.iterrows():
        s = closes.get(r.benchmark_index)
        m = re.search(r"_(\d+)d_", f"_{r.experiment}_")
        if s is None or not m:
            continue
        H = int(m.group(1))
        ts, te = pd.Timestamp(r.oos_start), pd.Timestamp(r.oos_end)
        after = s.index[s.index > te]
        ce = min(after[H - 1] if len(after) >= H else s.index.max(), s.index.max())
        try:
            _, mm, _ = bm.buy_and_hold(s, ts, ce, INITIAL_CASH)
        except Exception:
            continue
        df.at[i, "comparison_end"] = str(ce.date())  # pin the window
        for col, key in (("idx_bh_final_equity", "end"), ("idx_bh_total_return", "total_return"),
                         ("idx_bh_cagr", "cagr"), ("idx_bh_max_dd", "max_dd")):
            if pd.isna(df.at[i, col]):
                df.at[i, col] = mm[key]
        if pd.isna(df.at[i, "excess_return_total"]) and pd.notna(df.at[i, "total_return_strategy"]):
            df.at[i, "excess_return_total"] = (df.at[i, "total_return_strategy"]
                                               - df.at[i, "idx_bh_total_return"])
        n += 1
    return df, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REGISTRY))
    ap.add_argument("--dry-run", action="store_true", help="print summary, don't write")
    args = ap.parse_args()

    # r_precision_at_k.csv lookup (experiment → its registry dates), for train_start.
    rp_lookup = {}
    if RP_REGISTRY.exists():
        rp = pd.read_csv(RP_REGISTRY)
        key = "experiment" if "experiment" in rp.columns else rp.columns[0]
        for _, r in rp.iterrows():
            rp_lookup.setdefault(str(r[key]), r.to_dict())

    legacy = migrate_legacy_df(pd.read_csv(args.out))
    # Drop any prior _024 run rows (idempotent rebuild from artifacts).
    legacy = legacy[legacy.get("runner", pd.Series([""] * len(legacy))) != "run_backtest_cell"]
    legacy, n_bf = _backfill_index_metrics(legacy)
    if n_bf:
        print(f"pinned comparison_end (froze the index benchmark window) for {n_bf} legacy rows")

    next_id = int(legacy["id"].max()) + 1
    run_rows = []
    for i, (cell, name, status, diff, delta) in enumerate(_024_RUNS):
        rd = RUN_DIR_024 / cell
        if not (rd / "summary.json").exists():
            print(f"  WARN missing artifacts for {cell}; skipping")
            continue
        run_rows.append(row_from_run(rd, id=next_id + i, name=name,
                                     selfcheck_status=status, max_abs_diff=diff,
                                     universe_delta=delta, rp_lookup=rp_lookup))

    out = pd.concat([legacy, pd.DataFrame(run_rows)], ignore_index=True)
    out = out.reindex(columns=COLUMN_ORDER)
    # Deployment status: daily_preds is derived (per model); comment is seeded with
    # the deployment status where empty and otherwise preserved (manual comments survive).
    out["daily_preds"] = out["experiment"].isin(_DEPLOYED_CELLS)
    out["comment"] = [
        c if isinstance(c, str) and c.strip()
        else (_DEPLOY_COMMENT if dp else _UNDEPLOYED_COMMENT)
        for c, dp in zip(out["comment"], out["daily_preds"])
    ]
    out = out.sort_values(["run_timestamp", "id"], kind="stable").reset_index(drop=True)

    print(f"legacy rows: {len(legacy)} | _024 run rows: {len(run_rows)} | total: {len(out)}")
    print(f"columns: {len(out.columns)} (was 39)")
    if args.dry_run:
        print("\n[dry-run] _024 rows preview:")
        cols = ["name", "benchmark_index", "total_return_strategy", "excess_return_total",
                "calmar_strategy", "avg_gross_exposure", "pct_days_invested",
                "selfcheck_status", "mtm_truncated"]
        print(out[out.runner == "run_backtest_cell"][cols].to_string(index=False))
        return
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
