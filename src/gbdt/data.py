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

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Default cache-freshness tolerance in *calendar* days. A 14-trading-day
# threshold ~ 20 calendar days; we keep it slightly tight so a stale cache
# can't sneak past a long weekend or single holiday. Overridable via the
# ``staleness_days`` kwarg or the spec's ``data.staleness_days`` key — see
# PR #8 review (Minor 1).
DEFAULT_STALENESS_DAYS = 20

# Side effect: register the NSE domain so ``data_pipelines.fetch("NSE:...")``
# resolves. The us_equities domain is registered too so that an accidental US
# ticker in a spec fails downstream with a clean error instead of UnknownDomain.
import data_pipelines.domains.nse_equities  # noqa: F401
import data_pipelines.domains.us_equities  # noqa: F401
import data_pipelines.domains.fred_macro  # noqa: F401  (registers FRED: for macro features)
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
    # Cache freshness telemetry. Populated for kept tickers only; ``None``
    # when the per-domain meta has no range_end and the data table is empty.
    cache_last_date: str | None = None
    cache_age_days: int | None = None
    is_stale: bool = False
    # Rows dropped at cache read because every OHLCV value was NaN — see
    # PR #8 review (Minor 4).
    nan_rows_dropped: int = 0


@dataclass
class UniversePanel:
    """Loaded universe panel + per-ticker status."""

    universe: str
    panel: pd.DataFrame                  # long-format, MultiIndex (date, ticker)
    index_series: pd.DataFrame           # OHLCV for the index ticker
    annualization_factor: int
    statuses: list[TickerStatus] = field(default_factory=list)
    # Aggregate freshness/NaN telemetry surfaced to ``metrics.json::data``.
    stale_tickers: list[str] = field(default_factory=list)
    staleness_days_threshold: int = DEFAULT_STALENESS_DAYS

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
    """Default on-disk path for a universe YAML.

    Used by :func:`register_universe` (write side) and as a *fallback only*
    when the gbdt default.yaml's ``universes::<name>::source`` is absent.
    Read-side lookups should go through :func:`universe_metadata` and consult
    the ``source`` key — US universes live under ``us_equities/``, not
    ``nse_equities/``.
    """
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    return base / _DEFAULT_UNIVERSE_DIR / f"universe_{name}.yaml"


def register_universe(
    name: str,
    tickers: list[str],
    *,
    repo_root: Path | None = None,
    indices: list[str] | None = None,
    listed_at: str | date | None = None,
) -> Path:
    """Write a universe YAML so future calls can ``resolve_universe(name)``.

    Matches the existing ``universe_nifty50.yaml`` schema. The caller is
    responsible for the ``universes::<name>`` block in
    ``configs/gbdt/default.yaml`` (the gbdt loader resolves the universe YAML
    via ``source:`` there) — typically the orchestrator pre-flight does that
    step separately.

    ``listed_at`` is optional. When omitted the field is skipped entirely so
    that re-registering the same universe is deterministic (no `date.today()`
    creeping into diffs every regeneration — PR #8 review, Nit 7). Pass an
    explicit ISO date string (or :class:`datetime.date`) to record one.
    """
    path = _universe_yaml_path(name, repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"universe": name}
    if listed_at is not None:
        payload["listed_at"] = (
            listed_at if isinstance(listed_at, str) else listed_at.isoformat()
        )
    payload["indices"] = indices or []
    payload["tickers"] = list(tickers)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _resolve_universe_yaml_path(
    name: str, *, repo_root: Path | None = None
) -> Path:
    """Locate the universe YAML for ``name``.

    Resolution order:
    1. If the gbdt ``universes::<name>::source`` is set, treat it as
       repo-relative and use it. This is how US universes (which live under
       ``configs/data_pipelines/domains/us_equities/``) resolve.
    2. Otherwise fall back to the NSE convention in ``_DEFAULT_UNIVERSE_DIR``
       (so a freshly-registered NSE basket works before a ``universes::``
       block is added).
    """
    base = Path(repo_root) if repo_root is not None else Path.cwd()
    cfg = _read_gbdt_default(repo_root)
    block = (cfg.get("universes") or {}).get(name)
    if block is not None:
        source = block.get("source")
        if source:
            return base / source
    return _universe_yaml_path(name, repo_root)


def resolve_universe(
    name: str, *, repo_root: Path | None = None
) -> list[str]:
    """Read the universe YAML and return the ticker list.

    Raises ``FileNotFoundError`` with a hint when the preset isn't registered.
    """
    path = _resolve_universe_yaml_path(name, repo_root=repo_root)
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

    Hard-fails when no block exists — the silent fallback to NIFTY-style
    defaults was the cause of the exp-2 benchmark-misalignment bug (NIFTY:50
    used for nifty100 because nifty100 wasn't registered). The pre-flight
    flow in ``.claude/skills/gbdt-experiment/SKILL.md`` § "Universe
    self-service" is responsible for registering the block before the runner
    sees the spec.

    Schema (all keys required — see
    ``docs/data_pipelines/universe_yaml_spec.md`` for the full registry-block
    contract):

    - ``source``: repo-relative path to the universe YAML (ticker list).
      Constituents in the referenced YAML are always stored fully-prefixed
      (``"NSE:RELIANCE"``, ``"NASDAQ:AAPL"``, ``"NYSE:JPM"``); the registry
      block carries no prefix metadata.
    - ``index_ticker``: benchmark identifier in the data_pipelines cache
      (e.g. ``"NIFTY:100"``, ``"INDEX:^SPX"``). Drives all F1/F5/F9/F9b
      macro features.
    - ``annualization_factor``: trading days per year (``250`` for NSE,
      ``252`` for US). Propagated into the feature builder.
    """
    cfg = _read_gbdt_default(repo_root)
    block = (cfg.get("universes") or {}).get(name)
    if block is None:
        raise KeyError(
            f"Universe {name!r} referenced in spec but no "
            f"'universes::{name}' block in configs/gbdt/default.yaml. "
            f"Add one with source, index_ticker, "
            f"annualization_factor."
        )
    return block


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
    staleness_days: int = DEFAULT_STALENESS_DAYS,
    reference_date: str | date | None = None,
) -> dict[str, TickerStatus]:
    """For each ticker, verify cache row count meets ``min_rows`` and report
    freshness telemetry.

    With ``cache_only=True`` (default) we read directly from the SQLite cache
    — no provider calls. This is what the experiment runner uses: gbdt v1
    assumes the universe has already been seeded; the data_pipelines
    refresh flow is a separate concern (the ``/gbdt-experiment`` skill's
    pre-flight handles a cold-pull when needed).

    With ``cache_only=False`` we go through ``data_pipelines.fetch()``,
    which will detect and try to fill gaps against the provider chain. Use
    that only when you want a fresh seed.

    ``staleness_days`` (default ``DEFAULT_STALENESS_DAYS`` ≈ 14 trading days)
    is the calendar-day tolerance for cache freshness. If a kept ticker's
    cache max-date is older than ``reference_date - staleness_days`` we set
    ``status.is_stale = True`` and emit a logged warning — we never abort
    the run on staleness alone (PR #8 review, Minor 1). ``reference_date``
    defaults to the requested ``end`` (or today when ``end`` is open).
    """
    if start is None:
        start = "1990-01-01"

    ref = reference_date
    if ref is None:
        ref = end if end is not None else date.today()
    ref_d = ref if isinstance(ref, date) else date.fromisoformat(str(ref)[:10])

    statuses: dict[str, TickerStatus] = {}
    for ticker in tickers:
        try:
            if cache_only:
                df, nan_dropped = _cache_read(
                    ticker, start, end, repo_root=repo_root, return_nan_count=True,
                )
            else:
                ticker_end = end if end is not None else _cache_last_date(
                    ticker, repo_root=repo_root,
                ) or date.today().isoformat()
                df = _dp_fetch(
                    ticker, start, ticker_end,
                    data_root=_data_root(repo_root),
                )
                nan_dropped = 0
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
                nan_rows_dropped=int(nan_dropped),
            )
            continue
        # Kept: stamp freshness telemetry.
        last_iso = _cache_last_date(ticker, repo_root=repo_root)
        age_days: int | None = None
        is_stale = False
        if last_iso is not None:
            try:
                last_d = date.fromisoformat(last_iso)
                age_days = (ref_d - last_d).days
                is_stale = age_days > staleness_days
            except ValueError:
                age_days = None
        if is_stale:
            logger.warning(
                "stale cache: %s last_date=%s age=%dd > threshold=%dd",
                ticker, last_iso, age_days, staleness_days,
            )
        statuses[ticker] = TickerStatus(
            ticker=ticker, rows=n, kept=True,
            cache_last_date=last_iso,
            cache_age_days=age_days,
            is_stale=is_stale,
            nan_rows_dropped=int(nan_dropped),
        )
    return statuses


_OHLCV_COLS = ("open", "high", "low", "close", "adj_close", "volume")


def _cache_read(
    ticker: str,
    start: str | date,
    end: str | date | None,
    *,
    repo_root: Path | None = None,
    return_nan_count: bool = False,
):
    """Read ``ticker`` rows from the SQLite cache directly. Returns the
    canonical OHLCV DataFrame (``date, open, high, low, close, adj_close,
    volume``). Raises if the ticker isn't cached.

    Rows where every OHLCV value is NaN (a known data-pipelines adapter
    quirk on some NSE tickers — see ``docs/data_pipelines/V2_TBD.md``) are
    dropped silently here and reported in logs + ticker status. Set
    ``return_nan_count=True`` to receive ``(df, n_dropped)`` instead of
    just ``df`` (PR #8 review, Minor 4).
    """
    import sqlite3
    # Route to per-domain table. NSE domain owns NSE:/BSE:/NIFTY:; us_equities
    # domain owns NYSE:/NASDAQ:/INDEX: (the latter is for US benchmarks like
    # INDEX:^SPX — NSE indices use the NIFTY: prefix, not INDEX:).
    if ticker.startswith(("NSE:", "BSE:", "NIFTY:")):
        table = "nse_equities_data"
    else:
        table = "us_equities_data"
    db = Path(_data_root(repo_root)) / "processed.db"
    if not db.exists():
        raise FileNotFoundError(f"cache db missing at {db}")
    # Cached dates carry a 'YYYY-MM-DD 00:00:00' time component, so a bare
    # 'date <= end' string-compares the end-day row ('…16 00:00:00') as GREATER
    # than the bare end string ('…16') and silently drops it — an off-by-one
    # that hides the most recent day from every caller (load_panel, fresh
    # inference, …). Use a half-open interval [start_day, end_day + 1) on
    # normalized day boundaries: 'date >= start_day' includes the start day
    # (its '…00:00:00' sorts after the bare date), and 'date < (end_day + 1)'
    # includes the full end day regardless of the stored time component.
    start_s = pd.Timestamp(start).normalize().strftime("%Y-%m-%d")
    end_excl = ("2100-01-01" if end is None
                else (pd.Timestamp(end).normalize()
                      + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    con = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(
            f"SELECT date, open, high, low, close, adj_close, volume "
            f"FROM {table} WHERE ticker = ? AND date >= ? AND date < ? "
            f"ORDER BY date",
            con,
            params=(ticker, start_s, end_excl),
        )
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    # Drop all-NaN OHLCV rows. We keep rows with at least one non-null OHLCV
    # value (a partial row is still informative for the downstream rolling
    # features, which already use ``min_periods``).
    n_before = len(df)
    ohlcv_present = [c for c in _OHLCV_COLS if c in df.columns]
    if ohlcv_present:
        df = df.dropna(subset=ohlcv_present, how="all").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        logger.warning(
            "dropped %d all-NaN OHLCV rows from %s cache", n_dropped, ticker,
        )
    if return_nan_count:
        return df, int(n_dropped)
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
        # See _cache_read for the domain ownership map; INDEX: is us_equities.
        if ticker.startswith(("NSE:", "BSE:", "NIFTY:")):
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
# FRED macro panel (cache-only) — feeds the F17 macro feature family
# ---------------------------------------------------------------------------


def _cache_read_fred(
    series_id: str,
    start: str | date,
    end: str | date | None,
    *,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Read one FRED series (the ``fred_macro`` domain) from the SQLite cache,
    **cache-only** — never hits the network (experiments run sandboxed). Returns
    a ``(date, value)`` DataFrame. Uses the same half-open ``[start, end+1)``
    interval as :func:`_cache_read` so the end day's bar is never dropped.
    """
    import sqlite3
    db = Path(_data_root(repo_root)) / "processed.db"
    if not db.exists():
        raise FileNotFoundError(f"cache db missing at {db}")
    start_s = pd.Timestamp(start).normalize().strftime("%Y-%m-%d")
    end_excl = ("2100-01-01" if end is None
                else (pd.Timestamp(end).normalize()
                      + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    con = sqlite3.connect(str(db))
    try:
        df = pd.read_sql_query(
            "SELECT date, value FROM fred_macro_data "
            "WHERE ticker = ? AND date >= ? AND date < ? ORDER BY date",
            con, params=(f"FRED:{series_id}", start_s, end_excl),
        )
    finally:
        con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_macro_panel(
    series: Iterable[str],
    start: str | date | None,
    end: str | date | None,
    *,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    """Date-indexed DataFrame, one column per FRED series id (cache-only).

    Series with no cached rows are skipped with a warning (the F17 builder
    degrades gracefully on an absent series). Raises only if NONE of the
    requested series are cached — that means the FRED panel was never seeded.
    """
    cols: dict[str, pd.Series] = {}
    missing: list[str] = []
    for sid in series:
        df = _cache_read_fred(sid, start or "1990-01-01", end, repo_root=repo_root)
        if len(df) == 0:
            missing.append(sid)
            continue
        cols[sid] = df.set_index("date")["value"]
    if missing:
        logger.warning(
            "load_macro_panel: %d FRED series not cached (skipped): %s",
            len(missing), missing,
        )
    if not cols:
        raise RuntimeError(
            "load_macro_panel: none of the requested FRED series are cached — "
            "seed them first, e.g. `uv run python -m data_pipelines seed "
            "--domain fred_macro --universe macro --start 2003-01-01 --end <today>`"
        )
    return pd.DataFrame(cols).sort_index()


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
    staleness_days: int = DEFAULT_STALENESS_DAYS,
) -> UniversePanel:
    """Load a universe's OHLCV panel as a long-format MultiIndex DataFrame.

    Returns a :class:`UniversePanel` with:
    - ``panel``: ``MultiIndex(date, ticker)`` frame with canonical OHLCV cols
      (``open, high, low, close, adj_close, volume``).
    - ``index_series``: a flat-index DataFrame for the index ticker
      (e.g. ``^NSEI``) with one row per date; used by F1/F5/F9/F9b families.
    - ``annualization_factor`` (250 for NIFTY).
    - per-ticker ``statuses`` documenting kept / excluded outcomes (and
      cache-age + NaN-row-drop telemetry per PR #8 review).
    - ``stale_tickers``: subset of kept tickers whose cache max-date is
      older than ``staleness_days`` (default ≈ 14 trading days). Stale
      tickers are *not* dropped — the run continues and the staleness is
      recorded in the artifact for the analyst.
    """
    meta = universe_metadata(universe, repo_root=repo_root)
    tickers = resolve_universe(universe, repo_root=repo_root)

    statuses = ensure_universe_cached(
        tickers, start, end, min_rows=min_rows, repo_root=repo_root,
        cache_only=cache_only, staleness_days=staleness_days,
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

    stale = [s.ticker for s in statuses.values() if s.kept and s.is_stale]
    return UniversePanel(
        universe=universe,
        panel=panel,
        index_series=index_df,
        annualization_factor=int(meta.get("annualization_factor", 250)),
        statuses=list(statuses.values()),
        stale_tickers=stale,
        staleness_days_threshold=staleness_days,
    )
