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
        "index_ticker": "NIFTY:50",
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
    cache_only: bool = True,
) -> dict[str, TickerStatus]:
    """For each ticker, verify cache row count meets ``min_rows``.

    With ``cache_only=True`` (default) we read directly from the SQLite cache
    — no provider calls. This is what the experiment runner uses: gbdt v1
    assumes the universe has already been seeded; the data_pipelines
    refresh flow is a separate concern (the ``/gbdt-experiment`` skill's
    pre-flight handles a cold-pull when needed).

    With ``cache_only=False`` we go through ``data_pipelines.fetch()``,
    which will detect and try to fill gaps against the provider chain. Use
    that only when you want a fresh seed.
    """
    if start is None:
        start = "1990-01-01"
    statuses: dict[str, TickerStatus] = {}
    for ticker in tickers:
        try:
            if cache_only:
                df = _cache_read(ticker, start, end, repo_root=repo_root)
            else:
                ticker_end = end if end is not None else _cache_last_date(
                    ticker, repo_root=repo_root,
                ) or date.today().isoformat()
                df = _dp_fetch(
                    ticker, start, ticker_end,
                    data_root=_data_root(repo_root),
                )
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


def _cache_read(
    ticker: str,
    start: str | date,
    end: str | date | None,
    *,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Read ``ticker`` rows from the SQLite cache directly. Returns the
    canonical OHLCV DataFrame (``date, open, high, low, close, adj_close,
    volume``). Raises if the ticker isn't cached.
    """
    import sqlite3
    if ticker.startswith(("NSE:", "BSE:", "INDEX:", "NIFTY:")):
        table = "nse_equities_data"
    else:
        table = "us_equities_data"
    db = Path(_data_root(repo_root)) / "processed.db"
    if not db.exists():
        raise FileNotFoundError(f"cache db missing at {db}")
    start_s = start if isinstance(start, str) else start.isoformat()
    end_s = (end if isinstance(end, str)
             else (end.isoformat() if end is not None else "2099-01-01"))
    con = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(
            f"SELECT date, open, high, low, close, adj_close, volume "
            f"FROM {table} WHERE ticker = ? AND date >= ? AND date <= ? "
            f"ORDER BY date",
            con,
            params=(ticker, start_s, end_s),
        )
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _cache_last_date(ticker: str, *, repo_root: Path | None = None) -> str | None:
    """Look up the last cached trading date for ``ticker`` via the
    data_pipelines SQLite cache; returns None if not cached.

    Probes ``<domain>_meta.range_end`` first (cheap), falls back to MAX(date)
    on the per-domain data table. Routes to the right domain table by ticker
    prefix.
    """
    try:
        import sqlite3
        db = Path(_data_root(repo_root)) / "processed.db"
        if not db.exists():
            return None
        if ticker.startswith(("NSE:", "BSE:", "INDEX:", "NIFTY:")):
            domain = "nse_equities"
        else:
            domain = "us_equities"
        con = sqlite3.connect(str(db))
        try:
            cur = con.execute(
                f"SELECT range_end FROM {domain}_meta WHERE ticker = ?",
                (ticker,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])[:10]
            cur = con.execute(
                f"SELECT MAX(date) FROM {domain}_data WHERE ticker = ?",
                (ticker,),
            )
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0])[:10]
        finally:
            con.close()
    except Exception:
        return None
    return None


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
    cache_only: bool = True,
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
        cache_only=cache_only,
    )

    kept = [t for t, s in statuses.items() if s.kept]
    if not kept:
        raise RuntimeError(
            f"universe {universe!r} produced no usable tickers; "
            f"statuses={statuses}"
        )

    def _load_one(t: str) -> pd.DataFrame:
        if cache_only:
            return _cache_read(t, start or "1990-01-01", end, repo_root=repo_root)
        ticker_end = (end if end is not None
                      else _cache_last_date(t, repo_root=repo_root)
                      or date.today().isoformat())
        return _dp_fetch(
            t, start or "1990-01-01", ticker_end,
            data_root=_data_root(repo_root),
        )

    frames: list[pd.DataFrame] = []
    for t in kept:
        df = _load_one(t).copy()
        df["date"] = pd.to_datetime(df["date"])
        df["ticker"] = t
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel.set_index(["date", "ticker"]).sort_index()

    idx_ticker = meta.get("index_ticker", "INDEX:^NSEI")
    index_df = _load_one(idx_ticker).copy()
    index_df["date"] = pd.to_datetime(index_df["date"])
    index_df = index_df.sort_values("date").set_index("date")

    return UniversePanel(
        universe=universe,
        panel=panel,
        index_series=index_df,
        annualization_factor=int(meta.get("annualization_factor", 250)),
        statuses=list(statuses.values()),
    )
