"""V0 — universe-eligibility scan.

For each universe registered in ``configs/gbdt/default.yaml`` (nifty50,
nifty100, nifty_midcap_150, nifty500, sp500, nasdaq100, russell1000),
count how many constituent tickers have at least ``min_rows=1600`` rows of
cached daily history — the threshold the v1 runner applies via
``configs/gbdt/default.yaml::split.min_rows_per_ticker``.

This is the "row gate" that decides per-universe participation in the
single trailing-anchor fold the runner carves (``src/gbdt/train.py::carve_single_fold``;
``n_folds: 1`` in the default config). The metric is *not* per-fold-index — a
multi-fold sliding regime is not yet implemented in v1 — but per-universe
the share of tickers that survive the 1600-row gate is the leading
explanation for the participation-starvation pattern observed in
experiments 1–3.

Output: ``results/gbdt/v0_investigations/universe_eligibility.png``
(grouped bar chart, kept-vs-excluded per universe, threshold annotated).

Re-runnable as:

    uv run python -m scripts.gbdt.v0_universe_eligibility

The script reads ``configs/gbdt/default.yaml`` for the universe list and
uses ``gbdt.data.ensure_universe_cached`` for the row-count check — the
same code path the production runner uses, so the numbers here cannot drift
from the runner's per-experiment exclusion report.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from gbdt.data import ensure_universe_cached, resolve_universe

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = REPO_ROOT / "configs/gbdt/default.yaml"
OUTPUT_PNG = REPO_ROOT / "results/gbdt/v0_investigations/universe_eligibility.png"

MIN_ROWS = 1600  # mirrors split.min_rows_per_ticker in configs/gbdt/default.yaml

# Override knob for the data root the gbdt cache reader resolves against.
# ``gbdt.data._data_root`` builds ``<repo_root>/data/processed.db``; set
# ``GBDT_V0_DATA_REPO_ROOT`` to redirect the lookup (e.g. when the committed
# data/processed.db is unreadable on the host filesystem and a sibling cache
# lives elsewhere). Leave unset to use the repo's own ``data/`` directory.
DATA_REPO_ROOT_ENV = "GBDT_V0_DATA_REPO_ROOT"


def _load_universes(cfg_path: Path) -> dict[str, dict]:
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("universes") or {}


def _count_universe(
    name: str, repo_root: Path
) -> tuple[int, int, str]:
    """Return ``(kept, excluded, note)`` for one universe.

    ``note`` is empty on success, or a short reason when the universe could
    not be evaluated (missing constituent YAML, no cached tickers, etc.).
    """
    try:
        tickers = resolve_universe(name, repo_root=repo_root)
    except FileNotFoundError as exc:
        return 0, 0, f"constituent YAML missing ({exc})"
    except Exception as exc:  # noqa: BLE001
        return 0, 0, f"resolve failed: {exc}"

    statuses = ensure_universe_cached(
        tickers,
        start=None,
        end=None,
        min_rows=MIN_ROWS,
        repo_root=repo_root,
        cache_only=True,
    )
    kept = sum(1 for s in statuses.values() if s.kept)
    excluded = sum(1 for s in statuses.values() if not s.kept)
    return kept, excluded, ""


def _plot(rows: list[tuple[str, int, int, str]], out_path: Path) -> None:
    """Bar chart: kept vs excluded per universe, threshold annotated."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r[0] for r in rows]
    kept = [r[1] for r in rows]
    excluded = [r[2] for r in rows]
    totals = [k + e for k, e in zip(kept, excluded)]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = list(range(len(names)))
    bar_k = ax.bar(x, kept, label=f"eligible (>= {MIN_ROWS} rows)", color="#3a7d44")
    bar_e = ax.bar(x, excluded, bottom=kept, label="excluded", color="#c44536")

    # Annotate kept / total over each bar.
    for i, (k, t) in enumerate(zip(kept, totals)):
        label = f"{k} / {t}" if t > 0 else "n/a"
        ax.text(i, t + max(totals) * 0.01 + 1, label,
                ha="center", va="bottom", fontsize=9)

    # Annotate any universes that could not be evaluated.
    for i, r in enumerate(rows):
        note = r[3]
        if note:
            ax.text(i, max(totals) * 0.5 if max(totals) else 1,
                    "no constituent\nYAML", ha="center", va="center",
                    fontsize=8, color="#666", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("tickers")
    ax.set_title(
        f"gbdt v1 universe eligibility — tickers meeting min_rows={MIN_ROWS}\n"
        f"(single trailing-anchor fold; n_folds=1 in configs/gbdt/default.yaml)"
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    override = os.environ.get(DATA_REPO_ROOT_ENV)
    data_repo_root = Path(override) if override else REPO_ROOT
    db_path = data_repo_root / "data" / "processed.db"
    if not db_path.exists():
        raise FileNotFoundError(
            f"cache db missing at {db_path}; seed the cache with "
            f"data_pipelines, or set {DATA_REPO_ROOT_ENV} to a directory "
            f"whose 'data/processed.db' is readable."
        )

    universes = _load_universes(DEFAULT_CFG)
    if not universes:
        raise RuntimeError(f"no universes block found in {DEFAULT_CFG}")

    print(f"data_root = {db_path.parent}")
    print(f"min_rows  = {MIN_ROWS}")
    print(f"universes = {list(universes.keys())}")
    print()

    rows: list[tuple[str, int, int, str]] = []
    for name in universes:
        kept, excluded, note = _count_universe(name, data_repo_root)
        rows.append((name, kept, excluded, note))
        total = kept + excluded
        line = f"  {name:24s} kept={kept:5d}  excluded={excluded:5d}  total={total:5d}"
        if note:
            line += f"   [{note}]"
        print(line)

    _plot(rows, OUTPUT_PNG)
    print()
    print(f"chart -> {OUTPUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
