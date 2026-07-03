"""Per-share metrics + valuation ratios (pure, vectorized).

Given a daily frame carrying ``adj_close`` ($/share), ``shares`` (millions,
split-adjusted to the latest basis) and the TTM flow metrics ``revenue_ttm``,
``net_income_ttm``, ``fcf_ttm`` (all $M), computes:

  market_cap = adj_close × shares               ($M — dimensionally consistent
                                                  with the $M TTM metrics)
  eps_ttm    = net_income_ttm / shares          ($/share)
  rev_ps_ttm = revenue_ttm    / shares
  fcf_ps_ttm = fcf_ttm        / shares
  pe    = market_cap / net_income_ttm           (NaN when earnings ≤ 0)
  ps    = market_cap / revenue_ttm              (NaN when sales ≤ 0)
  p_fcf = market_cap / fcf_ttm                   (NaN when FCF ≤ 0)
  earnings_yield = net_income_ttm / market_cap   (signed, finite — modeling-
  sales_yield    = revenue_ttm    / market_cap    preferred: continuous across
  fcf_yield      = fcf_ttm        / market_cap    the zero-earnings crossing)

Ratio-vs-yield asymmetry is deliberate: a price/earnings ratio is meaningless
(and discontinuous) for a loss-making quarter, so PE/PS/P-FCF are NaN on a
non-positive denominator; the inverse yields carry the sign and stay finite,
which is what a model wants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_COLUMNS = (
    "market_cap", "eps_ttm", "rev_ps_ttm", "fcf_ps_ttm",
    "pe", "ps", "p_fcf", "earnings_yield", "sales_yield", "fcf_yield",
)


def _safe_div(num: pd.Series, den: pd.Series, *, positive_only: bool) -> pd.Series:
    """num/den as float64; result is NaN where den is NaN/0 (and where den ≤ 0
    if ``positive_only``)."""
    num = pd.to_numeric(num, errors="coerce").astype("float64")
    den = pd.to_numeric(den, errors="coerce").astype("float64")
    ok = den.notna() & num.notna() & (den != 0.0)
    if positive_only:
        ok &= den > 0.0
    return pd.Series(np.where(ok, num / den.where(ok, np.nan), np.nan),
                     index=num.index, dtype="float64")


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with the ten ratio columns added (existing columns kept).

    Requires ``adj_close``, ``shares``, ``revenue_ttm``, ``net_income_ttm``,
    ``fcf_ttm``. Rows missing any input yield NaN ratios (propagated by
    ``_safe_div``), never an exception.
    """
    out = df.copy()
    adj = pd.to_numeric(out["adj_close"], errors="coerce").astype("float64")
    sh = pd.to_numeric(out["shares"], errors="coerce").astype("float64")
    # A share count <= 0 is never valid — it's an upstream artifact (SPAC
    # transition quarters report 0 weighted shares; the EDGAR Q4 derivation
    # 4×FY − ΣQ1..3 can go negative on buyback-heavy years). Unmasked, those
    # rows emit PS=0 / negative market cap / sign-flipped EPS. Written back to
    # the output so the panel's shares column never carries the artifact.
    sh = sh.where(sh > 0.0)
    out["shares"] = sh
    ni = pd.to_numeric(out["net_income_ttm"], errors="coerce").astype("float64")
    rev = pd.to_numeric(out["revenue_ttm"], errors="coerce").astype("float64")
    fcf = pd.to_numeric(out["fcf_ttm"], errors="coerce").astype("float64")

    mcap = adj * sh  # $/share × M shares = $M
    out["market_cap"] = mcap
    out["eps_ttm"] = _safe_div(ni, sh, positive_only=False)
    out["rev_ps_ttm"] = _safe_div(rev, sh, positive_only=False)
    out["fcf_ps_ttm"] = _safe_div(fcf, sh, positive_only=False)

    # ratios: NaN on non-positive denominator (a negative PE is misleading)
    out["pe"] = _safe_div(mcap, ni, positive_only=True)
    out["ps"] = _safe_div(mcap, rev, positive_only=True)
    out["p_fcf"] = _safe_div(mcap, fcf, positive_only=True)

    # yields: signed + finite (den = market_cap > 0)
    out["earnings_yield"] = _safe_div(ni, mcap, positive_only=True)
    out["sales_yield"] = _safe_div(rev, mcap, positive_only=True)
    out["fcf_yield"] = _safe_div(fcf, mcap, positive_only=True)
    return out
