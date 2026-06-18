"""plot_actions — render a single-window back-test's action chart.

One chart per back-test run directory (anything produced by
``run_backtest_cell.py``): the strategy equity curve, the universe-index
buy-and-hold curve, and every buy/sell marked on the strategy curve with the
ticker as a horizontal label (entries stack upward, exits stack downward).
Exit labels carry the trigger suffix ``·t`` (target) / ``·D`` (drawdown-stop) /
``·h`` (horizon) / ``·b`` (breakeven).

Inputs (all written by the runner): ``summary.json`` (geometry + window),
``equity_curve.csv`` (strategy equity), ``picks.csv`` (the per-event log).
The index curve is recomputed the same way the runner does its benchmark
(``benchmarks.buy_and_hold`` over the back-test window) because the index
equity series is not persisted to disk.

CLI::

    uv run python -m scripts.backtests.plot_actions <run_dir> [<run_dir> ...]
    uv run python -m scripts.backtests.plot_actions <run_dir> --out path.png

Importable: ``run_backtest_cell.py`` calls ``plot_actions(out)`` so every new
single-window back-test emits ``figs/actions.png`` as a standard artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.backtests import benchmarks as bm
from scripts.backtests.run_cell5_bayesian_kelly import INITIAL_CASH, _load_closes

# Exit-trigger → one-char suffix on the sell label.
_TRIG = {"target": "t", "DD": "D", "horizon": "h", "breakeven": "b"}

# Benchmark-key prefix → index ticker, for older summaries (_001–_003) that
# predate the `geometry` block and name the index benchmark e.g. ``ndx_bh``.
_BH_TICKER = {"ndx": "INDEX:^NDX", "spx": "INDEX:^SPX",
              "nifty500": "NIFTY:500", "nifty50": "NIFTY:50", "index": "INDEX:^SPX"}


def _resolve_index(summary: dict) -> tuple[str | None, str, float | None]:
    """(index ticker, label, buy-hold total_return) across summary schemas.

    Newer runs (run_backtest_cell) carry ``geometry.universe`` + a ``index_bh``
    benchmark; older runs (_001–_003) carry only a ``<idx>_bh`` benchmark key.
    """
    benchmarks = summary.get("benchmarks", {})
    geom = summary.get("geometry")
    if geom and "universe" in geom:
        from scripts.backtests.run_backtest_cell import INDEX_BY_UNIVERSE  # lazy: avoid cycle

        if geom["universe"] in INDEX_BY_UNIVERSE:
            ticker, label = INDEX_BY_UNIVERSE[geom["universe"]]
            bh = benchmarks.get("index_bh", {}).get("total_return")
            return ticker, label, bh
    # Older schema: scan for the index buy-hold key (``*_bh`` that isn't ``ew_*``).
    for key, m in benchmarks.items():
        if key.endswith("_bh") and not key.startswith("ew"):
            prefix = key[:-3]
            return _BH_TICKER.get(prefix), prefix.upper(), m.get("total_return")
    return None, "index", None


def _window_span(window: dict) -> tuple[str, str]:
    """(start, end) date strings across window schemas (test_* / fresh_* / start)."""
    start = window.get("test_start") or window.get("fresh_start") or window.get("start")
    end = window.get("comparison_end") or window.get("fresh_end") or window.get("test_end")
    return start, end


def _index_equity(summary: dict, t_start: pd.Timestamp, t_end: pd.Timestamp):
    """Index buy-and-hold equity over the back-test window, or None if uncached."""
    ticker, label, bh = _resolve_index(summary)
    if ticker is None:
        return None, label, bh
    try:
        closes = _load_closes([ticker], t_start - pd.Timedelta(days=10), t_end)
        series = closes.get(ticker)
        if series is None or len(series) == 0:
            return None, label, bh
        eq, _, _ = bm.buy_and_hold(series, t_start, t_end, INITIAL_CASH)
        return eq, label, bh
    except Exception:
        return None, label, bh


def plot_actions(run_dir: Path | str, out: Path | str | None = None) -> Path:
    """Render ``<run_dir>/figs/actions.png`` (or ``out``) and return its path."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    window, strat = summary["window"], summary["strategy"]
    cfg = summary.get("config", {})
    out = Path(out) if out else run_dir / "figs" / "actions.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    start_str, end_str = _window_span(window)
    t_start, t_end = pd.Timestamp(start_str), pd.Timestamp(end_str)

    eq = pd.read_csv(run_dir / "equity_curve.csv", index_col=0, parse_dates=True).iloc[:, 0]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(eq.index, eq.values, lw=2, color="#1f4e79",
            label=f"strategy {strat['total_return'] * 100:+.0f}%  (DD {strat['max_dd'] * 100:.0f}%)")

    ie, idx_label, bh = _index_equity(summary, t_start, t_end)
    if ie is not None:
        lab = f"{idx_label} buy-hold" + (f" {bh * 100:+.0f}%" if bh is not None else "")
        ax.plot(ie.index, ie.values, lw=1.3, color="#888", ls="--", label=lab)
    ax.axhline(INITIAL_CASH, color="gray", lw=0.6, ls=":")

    # picks.csv is empty (no header) for 0-trade runs — the strategy sat in cash;
    # render equity + index anyway (the flat cash line is itself the finding).
    try:
        pk = pd.read_csv(run_dir / "picks.csv", parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pk = pd.DataFrame()
    if not pk.empty:
        pk = pk[pk["kind"].isin(["entry", "exit"])].copy()
    if not pk.empty:
        eqd = eq.reindex(eq.index.union(pk["date"].unique())).ffill()
        pk["y"] = pk["date"].map(eqd)
        ent = pk[pk["kind"] == "entry"]
        ex = pk[pk["kind"] == "exit"]
        ax.scatter(ent["date"], ent["y"], marker="^", s=46, color="#1a9850",
                   zorder=5, edgecolor="white", lw=0.5)
        ax.scatter(ex["date"], ex["y"], marker="v", s=46, color="#d73027",
                   zorder=5, edgecolor="white", lw=0.5)
        # Horizontal labels stacked vertically: entries upward, exits downward.
        fs = 6 if len(pk) > 40 else 7.5
        step = fs + 2.5
        for kind, df, sgn, col in [("entry", ent, 1, "#1a7a3a"), ("exit", ex, -1, "#b2182b")]:
            for _, grp in df.groupby("date"):
                for i, (_, r) in enumerate(grp.iterrows()):
                    tk = str(r["ticker"]).split(":")[-1]
                    lab = tk
                    if kind == "exit" and isinstance(r["trigger"], str):
                        lab = f"{tk}·{_TRIG.get(r['trigger'], '')}"
                    ax.annotate(lab, (r["date"], r["y"]),
                                xytext=(0, sgn * (9 + i * step)), textcoords="offset points",
                                ha="center", va="bottom" if sgn > 0 else "top",
                                fontsize=fs, color=col, rotation=0)

    name = summary.get("cell", run_dir.name)
    cfg_bits = []
    if cfg.get("selection_mode") or cfg.get("sizing_mode"):
        cfg_bits.append(f"{cfg.get('selection_mode', '?')}/{cfg.get('sizing_mode', '?')}")
    if cfg.get("K") is not None:
        cfg_bits.append(f"K={cfg['K']}")
    if cfg.get("fractional_c") is not None:
        cfg_bits.append(f"c={cfg['fractional_c']}")
    cfg_str = ("  ·  " + " ".join(cfg_bits)) if cfg_bits else ""
    title = (f"{name}{cfg_str}  ·  OOS {start_str} → {end_str}\n"
             f"▲ buy   ▼ sell (·t target  ·D drawdown-stop  ·h horizon)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_ylabel("equity ($)")
    ax.set_xlabel("date")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", help="back-test run directory(ies)")
    ap.add_argument("--out", default=None, help="output path (only valid for a single run_dir)")
    args = ap.parse_args()
    if args.out and len(args.run_dirs) > 1:
        ap.error("--out is only valid with a single run_dir")
    for rd in args.run_dirs:
        try:
            p = plot_actions(rd, args.out)
            print(f"[plot_actions] {rd} → {p}")
        except FileNotFoundError as e:
            print(f"[plot_actions] SKIP {rd}: missing {e.filename}")
        except Exception as e:  # noqa: BLE001 — one bad run dir shouldn't abort the batch
            print(f"[plot_actions] SKIP {rd}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
