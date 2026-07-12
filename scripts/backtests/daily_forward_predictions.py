"""Daily forward-prediction pipeline — the `_019` forward-OOS cadence.

Idempotent · self-gating · backfilling. Designed to run unattended (systemd
user timer) OR on demand via the `/daily-predictions` skill, on a machine that
is NOT always on. Because it backfills, a multi-day gap (machine asleep over a
weekend / holiday / vacation) just catches up on the next run.

Tracks the two deployed sp500 champions (``deployed=True``) plus higher-ranked
registry candidates (``deployed=False``) across sp500/russell1000/nasdaq100 —
one unified, model-keyed forward log with universe-aware gate columns.

Pipeline (in order):
  1. disk pre-flight (>= 10 G free, per the FS-wedge guard)
  2. seed each tracked universe to ``--end`` (idempotent; warm-cache tail fetch).
     ``--no-seed`` skips this and scores cache-only (backfills / offline).
  3. find the last snapshot already in the forward log (per model)
  4. self-gate: if the cache has not advanced past the last log, exit as a no-op
  5. INCREMENTAL inference (``infer_fresh_predictions --since``, ~5x cheaper),
     one shared feature build per universe → fresh CSV (gitignored scratch)
  6. regime gate (SMA200 on the cell's universe index — ^SPX / ^NDX) per new date
  7. append the top-K per (date, model) to the committed forward log
     (idempotent: a (snapshot_date, model) already present is never re-appended)
  8. optionally ``git`` commit the log (``--commit``)

The full per-(date,ticker) CSVs stay gitignored (regenerable); the compact
top-K **forward log** under ``results/backtests/data/`` is the durable,
checked-in record you grow over time and later join against realized outcomes.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from scripts.backtests.run_backtest_cell import INDEX_BY_UNIVERSE
from scripts.backtests.run_cell5_bayesian_kelly import _load_closes
from scripts.backtests.regime_signals import risk_on_sma
from scripts.backtests.infer_fresh_predictions import build_scores_multi, self_check

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "results/backtests/data/forward_predictions_log.csv"
SCRATCH = REPO / "results/backtests/_019_fwd_oos"  # gitignored fresh CSVs
TOP_K = 10            # rows logged per (date, model)
MIN_FREE_GB = 10.0
REGIME_MA = 200       # SMA window for the deployment gate
SEED_JOBS = 8         # parallel fetch workers for the seed (network I/O-bound; V1.6 S2)

# Cells tracked by the cadence. ``deployed=True`` are the two validated sp500
# champions wired into /daily-predictions (the signal record). ``deployed=False``
# are higher-ranked registry candidates tracked for forward comparison — NOT
# deployed (their registry ``daily_preds`` stays False); a later promotion is a
# separate decision. ``backfill_from`` floors a cell's FIRST-run backfill, clamped
# to >= the cell's gbdt test_end (pre-test_end dates are in-sample, NOT forward-OOS).
# Candidate test windows end 2024-10 (russell_50_200), 2026-03 (nasdaq_40_50),
# 2025-05 (russell_40_100): January 2026 is genuine OOS for the two russell cells
# (backfilled from 2026-01-01), but IN-sample for nasdaq_40_50 — its backfill_from
# of 2026-01-01 therefore clamps to its 2026-03-13 OOS start. Champions (test_end
# 2026-03/04) stay ``None`` -> test_end (their OOS legitimately begins then).
CELLS = {
    "sp500_50":       {"cell": "results/gbdt/experiments/sp500_up_50pct_50d_dd25pct_agentloop",
                       "deployed": True,  "backfill_from": None},
    "sp500_20":       {"cell": "results/gbdt/experiments/sp500_up_20pct_25d_dd10pct_agentloop",
                       "deployed": True,  "backfill_from": None},
    "russell_50_200": {"cell": "results/gbdt/experiments/russell1000_up_50pct_200d_dd25pct_aligned_agent_v14p1",
                       "deployed": False, "backfill_from": "2026-01-01"},
    "nasdaq_40_50":   {"cell": "results/gbdt/experiments/nasdaq100_up_40pct_50d_dd20pct_agentloop_mix",
                       "deployed": False, "backfill_from": "2026-01-01"},
    "russell_40_100": {"cell": "results/gbdt/experiments/russell1000_up_40pct_100d_dd20pct_aligned_agent_v14p1",
                       "deployed": False, "backfill_from": "2026-01-01"},
    # SPLIT-ADJUSTED champion candidates (V5 fix). The two deployed sp500 champions
    # reproduced on split-adjusted prices (champmatch single-fit: candidates=all,
    # mcw=10, train_start 2019, one fit). Tracked deployed=False to forward-compare
    # the corrected-price config against the deployed (unadjusted-trained) champions
    # ahead of a swap decision. NOTE: the split fix moves the champions negligibly
    # (ΔAUC ±0.004, #37), so these track ≈ their deployed counterparts. test_end
    # 2024-12-16 → backfill from 2026-01-01 is genuine OOS. Artifacts are the
    # V5-validated adjusted build (self-check faithful); a champion swap is separate.
    "sp500_50_adj": {"cell": "results/gbdt/experiments/sp500_up_50pct_50d_dd25pct_aligned_champmatch",
                     "deployed": False, "backfill_from": "2026-01-01"},
    "sp500_20_adj": {"cell": "results/gbdt/experiments/sp500_up_20pct_25d_dd10pct_aligned_champmatch",
                     "deployed": False, "backfill_from": "2026-01-01"},
    # (removed sp500_40_200 — the F18 fundamentals candidate failed two-window
    # replication, _279/_280; the technical champions dominate it on the fresh-OOS
    # backtest, so it's demoted out of the tracked comparison set.)
}

# Unified v2 schema. Gate columns are universe-aware (``gate_index`` names the
# index the regime gate uses — ^SPX for sp500/russell1000, ^NDX for nasdaq100);
# ``p_calibrated`` carries the model's native probability (the inference path
# returns p_raw; isotonic recalibration is monotonic so top-K ranks are identical).
LOG_COLUMNS = [
    "snapshot_date", "cell", "model", "universe", "threshold_pct", "horizon_days",
    "rank", "ticker", "p_calibrated", "base_rate",
    "gate_index", "gate_close", "gate_sma200", "regime_on",
    "deployed", "logged_at",
]


def _preflight_disk() -> None:
    free_gb = shutil.disk_usage(REPO).free / 1e9
    if free_gb < MIN_FREE_GB:
        sys.exit(f"[ABORT] only {free_gb:.1f} G free (< {MIN_FREE_GB} G) — "
                 "refusing to run (FS-wedge guard).")
    print(f"[preflight] disk OK: {free_gb:.0f} G free")


def _seed(universes: list[str], end: str) -> None:
    for uni in universes:
        print(f"[seed] refreshing {uni} universe → {end} ...")
        r = subprocess.run(
            [sys.executable, "-m", "data_pipelines", "seed", "--domain", "us_equities",
             "--universe", uni, "--start",
             str((pd.Timestamp(end) - pd.Timedelta(days=20)).date()), "--end", end,
             "--jobs", str(SEED_JOBS)],
            cwd=REPO, capture_output=True, text=True)
        tail = r.stdout.strip().splitlines()[-1:] or [r.stderr.strip()[-200:]]
        print(f"[seed] {uni}: {tail[0]}")
        if r.returncode != 0:
            sys.exit(f"[ABORT] seed failed for {uni} (rc={r.returncode}).")


def _last_logged(model: str) -> pd.Timestamp | None:
    if not LOG.exists():
        return None
    df = pd.read_csv(LOG, parse_dates=["snapshot_date"])
    sub = df[df["model"] == model]
    return sub["snapshot_date"].max() if len(sub) else None


# Liquid bellwethers whose EOD bar lands with the broad market. We gate on the
# STOCK panel, NOT the index: the index EOD/quote can lead the constituent stock
# bars intraday, so gating on ^SPX would falsely trigger a ~5 min/cell no-op
# inference on the very day before stock EOD finalizes. Max over a few mega-caps
# is robust to any single name being halted.
_GATE_TICKERS = ("NASDAQ:AAPL", "NASDAQ:MSFT", "NYSE:JPM")


def _cache_max_date(universe: str) -> pd.Timestamp | None:
    """Latest COMPLETE stock bar (max over liquid bellwethers, EXCLUDING the
    current calendar day whose bar may be an in-progress intraday partial). A
    cheap pre-gate so a run with no genuinely-new complete stock data skips the
    inference entirely. ``date < today`` (string compare on the 'YYYY-MM-DD
    00:00:00' column) drops today's bar and keeps prior complete days."""
    import sqlite3
    qs = ",".join("?" * len(_GATE_TICKERS))
    today = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    con = sqlite3.connect(REPO / "data/processed.db")
    try:
        row = con.execute(
            f"SELECT MAX(date) FROM us_equities_data "
            f"WHERE ticker IN ({qs}) AND date < ?",
            (*_GATE_TICKERS, today),
        ).fetchone()
    finally:
        con.close()
    return pd.Timestamp(row[0]) if row and row[0] else None


def _cell_meta(cell_dir: Path) -> tuple[str, int, int, float]:
    spec = yaml.safe_load((cell_dir / "spec.yaml").read_text())["target"]
    test = pd.read_csv(cell_dir / "predictions" / "test.csv")
    base_rate = float(test["y_true"].mean()) if "y_true" in test else float("nan")
    return (spec["universe"], int(spec["threshold_pct"]), int(spec["horizon_days"]),
            base_rate)


def _warmup_start(since: pd.Timestamp) -> str:
    """~7y trailing slice start before `since` (mirrors infer_fresh_predictions's
    --since): comfortably exceeds the 1600-td eligibility floor so the kept-ticker
    set + cross-sectional features match the full build, while skipping deep
    history that contributes nothing to ≤200d rolling features."""
    return str((since - pd.Timedelta(days=2700)).date())


def _regime_series(universe: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Series:
    idx_tk = INDEX_BY_UNIVERSE[universe][0]
    s = _load_closes([idx_tk], lo - pd.Timedelta(days=420), hi + pd.Timedelta(days=2))[idx_tk]
    on = risk_on_sma(s, REGIME_MA)
    return s, on


def run(end: str, commit: bool, no_seed: bool = False, deployed_only: bool = False) -> int:
    _preflight_disk()
    # Daily fast path (--deployed-only): score ONLY the deployed sp500 champions
    # and skip the 3 comparison candidates, so the daily seed + build stay on the
    # sp500 universe (the ~1,006-ticker russell1000 seed + its two ~10-min builds
    # are the bulk of the ~60-min full cadence). A separate weekly run refreshes the
    # candidates; their forward series is weekly-sampled + backfilled, fine for
    # comparison-tracking. See docs/gbdt/V1.6 § seed track + _019.
    cells = {m: c for m, c in CELLS.items() if c["deployed"]} if deployed_only else CELLS
    if deployed_only:
        print(f"[tier] deployed-only: scoring {', '.join(cells)} "
              "(candidates skipped — the weekly run refreshes them).")
    metas = {model: _cell_meta(REPO / m["cell"]) for model, m in cells.items()}
    universes = sorted({metas[model][0] for model in cells})
    if no_seed:
        print(f"[seed] skipped (--no-seed); scoring cache-only for {', '.join(universes)}.")
    else:
        _seed(universes, end)

    # Per-universe warmup anchor: the warmup slice must be deep enough to faithfully
    # rebuild the OLDEST tracked test window in that universe (self_check reproduces
    # predictions/test.csv). We anchor to the earliest test_start among the universe's
    # cells — NOT to `since`, which for a candidate can sit years after its test window
    # (a too-shallow warmup shifts long-lookback/cross-sectional features and the
    # reproduction diverges → abort). One anchor per universe also keeps the shared
    # per-universe feature build (same warmup ⇒ one build).
    test_dates = {model: pd.read_csv(REPO / m["cell"] / "predictions" / "test.csv",
                                     usecols=["date"], parse_dates=["date"])["date"]
                  for model, m in cells.items()}
    uni_anchor: dict[str, pd.Timestamp] = {}
    for model in cells:
        u, ts = metas[model][0], test_dates[model].min()
        uni_anchor[u] = min(uni_anchor.get(u, ts), ts)
    uni_warmup = {u: _warmup_start(a) for u, a in uni_anchor.items()}

    # Pass 1 — cheap pre-gate: decide which cells actually advanced past their last
    # logged snapshot, so the shared build below covers exactly those cells.
    todo: list[dict] = []
    for model, m in cells.items():
        cell_dir = REPO / m["cell"]
        universe, thr, hor, base_rate = metas[model]
        since = _last_logged(model)
        if since is None:  # first ever run for this model → start after its test window
            since = test_dates[model].max()
            bf = m.get("backfill_from")
            if bf is not None:  # floor the first-run backfill (candidates → April)
                since = max(since, pd.Timestamp(bf) - pd.Timedelta(days=1))
        cache_max = _cache_max_date(universe)
        if cache_max is not None and cache_max <= since:
            print(f"[{model}] cache latest {cache_max.date()} not past last log "
                  f"{since.date()} — no-op (skipping inference).")
            continue
        todo.append(dict(model=model, cell_dir=cell_dir, since=since, universe=universe,
                         thr=thr, hor=hor, base_rate=base_rate, deployed=m["deployed"],
                         warmup=uni_warmup[universe]))

    if not todo:
        print("[done] no new snapshots — log unchanged (no-op).")
        return 0

    # Shared inference: ONE load_panel + build_feature_matrix per
    # (universe, slice, alignment). The two sp500 champions share a universe and
    # (logged together) a `since`, so they collapse to ONE feature build instead
    # of two — ~halves inference wall-time. Each cell's per-cell self-check below
    # still guards faithfulness independently, so the sharing can only abort
    # loudly on a mismatch, never emit bad scores.
    print(f"[infer] shared feature build for {len(todo)} cell(s): "
          f"{', '.join(t['model'] for t in todo)} (incremental from {todo[0]['warmup']}) ...")
    specs = [(t["cell_dir"], t["warmup"]) for t in todo]
    scores_by_cell = build_scores_multi(specs, end, align_panel=True)

    new_rows: list[dict] = []
    logged_at = str(pd.Timestamp.now())
    today = pd.Timestamp.now().normalize()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # Pass 2 — per-cell: self-check, write the regenerable scratch CSV, regime-gate,
    # and build the top-K log rows for complete days only.
    for t in todo:
        model, cell_dir, since = t["model"], t["cell_dir"], t["since"]
        universe, thr, hor, base_rate = t["universe"], t["thr"], t["hor"], t["base_rate"]
        deployed = t["deployed"]
        scores = scores_by_cell[str(cell_dir)].copy()
        scores["date"] = pd.to_datetime(scores["date"])
        self_check(scores, cell_dir, incremental=True, label=model)

        fresh = scores[scores["date"] > since].copy()
        fresh["p_calibrated"] = fresh["p_raw"]  # native isotonic pass-through on this cell
        fresh = fresh.sort_values(["date", "ticker"]).reset_index(drop=True)
        fresh.to_csv(SCRATCH / f"{model}_fresh.csv", index=False)  # gitignored, regenerable

        # Never log the current day's bar — it may be an in-progress intraday
        # partial (low volume, mid-session). Only complete days (strictly before
        # today) enter the forward log; today's bar is logged on the next run
        # after it finalizes. Belt-and-suspenders with the _cache_max_date gate.
        fresh = fresh[fresh["date"] < today]
        if fresh.empty:
            print(f"[{model}] no new COMPLETE trading days since {since.date()} "
                  f"(today's bar, if any, is excluded as in-progress) — nothing to log.")
            continue

        gate_idx_tk = INDEX_BY_UNIVERSE[universe][0]
        gate, regime = _regime_series(universe, fresh["date"].min(), fresh["date"].max())
        reg_map = regime.reindex(sorted(set(regime.index) | set(fresh["date"]))).ffill().to_dict()
        gate_map = gate.reindex(sorted(set(gate.index) | set(fresh["date"]))).ffill().to_dict()
        gma_map = gate.rolling(REGIME_MA).mean().reindex(
            sorted(set(gate.index) | set(fresh["date"]))).ffill().to_dict()

        for d, day in fresh.groupby("date"):
            top = day.sort_values("p_calibrated", ascending=False).head(TOP_K)
            ro = reg_map.get(pd.Timestamp(d))
            for rank, r in enumerate(top.itertuples(), 1):
                new_rows.append({
                    "snapshot_date": str(pd.Timestamp(d).date()), "cell": cell_dir.name,
                    "model": model, "universe": universe,
                    "threshold_pct": thr, "horizon_days": hor, "rank": rank,
                    "ticker": r.ticker, "p_calibrated": round(float(r.p_calibrated), 6),
                    "base_rate": round(base_rate, 6),
                    "gate_index": gate_idx_tk,
                    "gate_close": round(float(gate_map.get(pd.Timestamp(d), float("nan"))), 2),
                    "gate_sma200": round(float(gma_map.get(pd.Timestamp(d), float("nan"))), 2),
                    "regime_on": bool(ro) if ro == ro else True,  # NaN→risk-on
                    "deployed": deployed,
                    "logged_at": logged_at,
                })
        print(f"[{model}] logged {fresh['date'].nunique()} new day(s) "
              f"[{fresh['date'].min().date()}..{fresh['date'].max().date()}] × top-{TOP_K}.")

    if not new_rows:
        print("[done] no new snapshots — log unchanged (no-op).")
        return 0

    add = pd.DataFrame(new_rows)[LOG_COLUMNS]
    if LOG.exists():
        old = pd.read_csv(LOG)
        # idempotent: drop any (snapshot_date, model) already present
        present = set(zip(old["snapshot_date"].astype(str), old["model"]))
        add = add[~add.apply(lambda r: (r["snapshot_date"], r["model"]) in present, axis=1)]
        out = pd.concat([old, add], ignore_index=True)
    else:
        out = add
    if add.empty:
        print("[done] all computed snapshots already logged — no-op (idempotent).")
        return 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(LOG, index=False)
    print(f"[done] appended {len(add)} rows ({add['snapshot_date'].nunique()} day(s)) "
          f"→ {LOG.relative_to(REPO)} (now {len(out)} rows).")

    if commit:
        subprocess.run(["git", "add", "-f", str(LOG)], cwd=REPO, check=False)
        msg = (f"backtests: forward-prediction log — append "
               f"{sorted(add['snapshot_date'].unique())[0]}"
               f"..{sorted(add['snapshot_date'].unique())[-1]}")
        rc = subprocess.run(["git", "commit", "-m", msg], cwd=REPO,
                            capture_output=True, text=True)
        print(f"[commit] {'ok: ' + msg if rc.returncode == 0 else 'skipped (' + rc.stdout.strip()[-120:] + ')'}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily forward-prediction cadence (_019).")
    ap.add_argument("--end", default=str(pd.Timestamp.today().date()),
                    help="as-of date (default: today). Data through the latest "
                         "available trading day ≤ end is used.")
    ap.add_argument("--commit", action="store_true",
                    help="git-commit the appended forward log locally (no push).")
    ap.add_argument("--no-seed", action="store_true",
                    help="skip the data-refresh seed and score cache-only — for "
                         "backfilling over already-cached history, or running offline.")
    ap.add_argument("--deployed-only", action="store_true",
                    help="DAILY FAST PATH: seed + score ONLY the deployed sp500 "
                         "champions, skipping the 3 comparison candidates and their "
                         "heavier russell1000/nasdaq100 universes. A separate weekly run "
                         "refreshes the candidates. Cuts the seed (no ~1,006-ticker "
                         "russell fetch) and the build (no russell double-build).")
    args = ap.parse_args()
    sys.exit(run(args.end, args.commit, args.no_seed, args.deployed_only))


if __name__ == "__main__":
    main()
