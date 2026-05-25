"""Universe panel loader for gbdt v1.

Loads a NIFTY 50 (or other registered NSE universe) OHLCV panel via
``data_pipelines.fetch()``, keyed by ticker. Handles:

- Resolving a preset name → ticker list (reading the universe YAML registered
  under ``configs/data_pipelines/domains/nse_equities/universe_<name>.yaml``).
- Self-service registration: writing a fresh universe YAML on first use so
  a spec for ``nifty500`` runs end-to-end without manual infra edits.
- Per-ticker cache fetch via ``data_pipelines.fetch()``.
- Aligning per-stock OHLCV frames into a long-format ``(date, ticker)``
  MultiIndex panel.
- Dropping tickers below ``min_rows`` with a structured exclusion report.

Walk-forward boundary discipline (CLAUDE.md C6) is *not* enforced here —
this layer ships the raw aligned panel; ``train.py`` carves segments in
time order downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

# Side effect: register the NSE domain so ``data_pipelines.fetch("NSE:...")``
# resolves. The us_equities domain is registered too so that an accidental US
# ticker in a spec fails downstream with a clean error instead of UnknownDomain.
import data_pipelines.domains.nse_equities  # noqa: F401
import data_pipelines.domains.us_equities  # noqa: F401
from data_pipelines import fetch as _dp_fetch


# Repo-relative roots. Resolved against the CWD unless caller passes
# ``repo_root`` explicitly (useful in tests).
_DEFAULT_UNIVERSE_DIR = Path("configs/data_pipelines/domains/nse_equities")
_DEFAULT_GBDT_CONFIG = Path("configs/gbdt/default.yaml")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TickerStatus:
    """Per-ticker outcome from :func:`ensure_universe_cached`."""

    ticker: str
    rows: int
    kept: bool
    reason: str = ""


@dataclass
class UniversePanel:
    """Loaded universe panel + per-ticker status."""

    universe: str
    panel: pd.DataFrame                  # long-format, MultiIndex (date, ticker)
    index_series: pd.DataFrame           # OHLCV for the index ticker
    annualization_factor: int
    statuses: list[TickerStatus] = field(default_factory=list)

    @property
    def tickers_kept(self) -> list[str]:
        return [s.ticker for s in self.statuses if s.kept]

    @property
    def tickers_excluded(self) -> list[str]:
        return [s.ticker for s in self.statuses if not s.kept]


# ---------------------------------------------------------------------------
# Universe registration / resolution
# ---------------------------------------------------------------------------


def _universe_yaml_path(name: str, repo_root: Path | None = None) -> Path:
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    return base / _DEFAULT_UNIVERSE_DIR / f"universe_{name}.yaml"


def register_universe(
    name: str,
    tickers: list[str],
    *,
    repo_root: Path | None = None,
    indices: list[str] | None = None,
) -> Path:
    """Write a universe YAML so future calls can ``resolve_universe(name)``.

    Matches the existing ``universe_nifty50.yaml`` schema. The caller is
    responsible for the ``universes::<name>`` block in
    ``configs/gbdt/default.yaml`` (the gbdt loader resolves the universe YAML
    via ``source:`` there) — typically the orchestrator pre-flight does that
    step separately.
    """
    path = _universe_yaml_path(name, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "universe": name,
        "listed_at": str(date.today()),
        "indices": indices or [],
        "tickers": list(tickers),
    }
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def resolve_universe(
    name: str, *, repo_root: Path | None = None
) -> list[str]:
    """Read the universe YAML and return the ticker list.

    Raises ``FileNotFoundError`` with a hint when the preset isn't registered.
    """
    path = _universe_yaml_path(name, repo_root)
    if not path.exists():
        raise FileNotFoundError(
            f"universe {name!r} is not registered "
            f"(no file at {path}); call register_universe({name!r}, [...]) first"
        )
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    tickers = doc.get("tickers") or []
    if not tickers:
        raise ValueError(f"universe YAML {path} has no tickers")
    return list(tickers)


def _read_gbdt_default(repo_root: Path | None = None) -> dict:
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    path = base / _DEFAULT_GBDT_CONFIG
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def universe_metadata(
    name: str, *, repo_root: Path | None = None
) -> dict:
    """Return the ``universes::<name>`` block from gbdt default.yaml.

    Falls back to NIFTY-style defaults if the gbdt config doesn't pre-register
    the preset (typical for self-service universes added on first use).
    """
    cfg = _read_gbdt_default(repo_root)
    block = (cfg.get("universes") or {}).get(name)
    if block is not None:
        return block
    return {
        "source": str(_DEFAULT_UNIVERSE_DIR / f"universe_{name}.yaml"),
        "index_ticker": "INDEX:^NSEI",
        "ticker_prefix": "NSE:",
        "annualization_factor": 250,
    }


# ---------------------------------------------------------------------------
# Cache check + ticker filter
# ---------------------------------------------------------------------------


def ensure_universe_cached(
    tickers: Iterable[str],
    start: str | date | None,
    end: str | date | None,
    min_rows: int = 1600,
    *,
    repo_root: Path | None = None,
) -> dict[str, TickerStatus]:
    """For each ticker, call ``data_pipelines.fetch()`` and verify row count.

    Returns a dict ``ticker -> TickerStatus``. Tickers below ``min_rows`` are
    flagged ``kept=False`` with a descriptive reason; the caller decides what
    to do (the panel loader drops them, the agent surface logs them).
    """
    if start is None:
        start = "1990-01-01"
    if end is None:
        end = date.today().isoformat()
    statuses: dict[str, TickerStatus] = {}
    for ticker in tickers:
        try:
            df = _dp_fetch(ticker, start, end, data_root=_data_root(repo_root))
        except Exception as exc:
            statuses[ticker] = TickerStatus(
                ticker=ticker, rows=0, kept=False, reason=f"fetch failed: {exc}",
            )
            continue
        n = int(len(df))
        if n < min_rows:
            statuses[ticker] = TickerStatus(
                ticker=ticker, rows=n, kept=False,
                reason=f"only {n} rows; need >= {min_rows}",
            )
        else:
            statuses[ticker] = TickerStatus(ticker=ticker, rows=n, kept=True)
    return statuses


def _data_root(repo_root: Path | None) -> str:
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    return str(base / "data")


# ---------------------------------------------------------------------------
# Panel build
# ---------------------------------------------------------------------------


def load_panel(
    universe: str,
    start: str | date | None = None,
    end: str | date | None = None,
    *,
    min_rows: int = 1600,
    repo_root: Path | None = None,
) -> UniversePanel:
    """Load a universe's OHLCV panel as a long-format MultiIndex DataFrame.

    Returns a :class:`UniversePanel` with:
    - ``panel``: ``MultiIndex(date, ticker)`` frame with canonical OHLCV cols
      (``open, high, low, close, adj_close, volume``).
    - ``index_series``: a flat-index DataFrame for the index ticker
      (e.g. ``^NSEI``) with one row per date; used by F1/F5/F9/F9b families.
    - ``annualization_factor`` (250 for NIFTY).
    - per-ticker ``statuses`` documenting kept / excluded outcomes.
    """
    meta = universe_metadata(universe, repo_root=repo_root)
    tickers = resolve_universe(universe, repo_root=repo_root)

    statuses = ensure_universe_cached(
        tickers, start, end, min_rows=min_rows, repo_root=repo_root,
    )

    kept = [t for t, s in statuses.items() if s.kept]
    if not kept:
        raise RuntimeError(
            f"universe {universe!r} produced no usable tickers; "
            f"statuses={statuses}"
        )

    frames: list[pd.DataFrame] = []
    for t in kept:
        df = _dp_fetch(
            t,
            start or "1990-01-01",
            end or date.today().isoformat(),
            data_root=_data_root(repo_root),
        ).copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = t
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel.set_index(["date", "ticker"]).sort_index()

    idx_ticker = meta.get("index_ticker", "INDEX:^NSEI")
    index_df = _dp_fetch(
        idx_ticker,
        start or "1990-01-01",
        end or date.today().isoformat(),
        data_root=_data_root(repo_root),
    ).copy()
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values("date").set_index("date")

    return UniversePanel(
        universe=universe,
        panel=panel,
        index_series=index_df,
        annualization_factor=int(meta.get("annualization_factor", 250)),
        statuses=list(statuses.values()),
    )
