"""Backfill pre-2019 NSE annual fundamentals from screener.in (task #32).

Fetches the annual Profit&Loss (+ Balance Sheet for equity capital) table for each
nifty500 ticker, parses per-fiscal-year revenue / net_income / eps / operating_profit /
pbt, converts Rs Crore -> INR millions (x10), and writes a normalized parquet.

NO writes to processed.db and NO valuation-panel rebuild -- parquet + report only.

Politeness: browser UA, ~1.7s sleep between live fetches, raw HTML cached under
/tmp/screener_cache/<SLUG><suffix>.html so re-runs don't re-fetch.
"""
from __future__ import annotations

import os
import re
import sys
import time
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from bs4 import BeautifulSoup

CACHE = Path("/tmp/screener_cache")
CACHE.mkdir(exist_ok=True)
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
SSL = "/etc/ssl/certs/ca-certificates.crt"
SLEEP = 1.7

REPO = Path("/mnt/Workspace/Workspace/wt-nse-valuation")
UNIVERSE = REPO / "configs/data_pipelines/domains/nse_equities/universe_nifty500.yaml"
OUT = REPO / "results/valuation/data/screener_annual_backfill.parquet"

# revenue-analog label priority (banks/NBFC/insurers differ)
REVENUE_LABELS = ["sales", "revenue", "financing profit", "interest earned",
                  "premium", "total income", "net interest income"]
# labels we consider "ambiguous" (used revenue but not the canonical Sales/Revenue)
AMBIGUOUS_REVENUE = ["financing profit", "interest earned", "premium",
                     "total income", "net interest income"]


def slug_of(ticker: str) -> str:
    """NSE symbol -> screener slug. Screener slug == NSE symbol for the vast majority."""
    return ticker


def fetch(slug: str, consolidated: bool) -> tuple[str | None, int]:
    """Return (html_or_None, http_code). Cached to disk."""
    suffix = "" if consolidated else "_std"
    path = CACHE / f"{slug}{suffix}.html"
    if path.exists() and path.stat().st_size > 5000:
        return path.read_text(errors="replace"), 200
    url = (f"https://www.screener.in/company/{slug}/consolidated/"
           if consolidated else f"https://www.screener.in/company/{slug}/")
    env = dict(os.environ, SSL_CERT_FILE=SSL)
    r = subprocess.run(
        ["curl", "-s", "-w", "%{http_code}", "-A", UA, "-o", str(path), url],
        env=env, capture_output=True, text=True,
    )
    code = r.stdout.strip()[-3:]
    try:
        code_i = int(code)
    except ValueError:
        code_i = 0
    time.sleep(SLEEP)
    if code_i != 200 or not path.exists() or path.stat().st_size < 5000:
        if path.exists():
            path.unlink()  # don't cache errors
        return None, code_i
    return path.read_text(errors="replace"), code_i


def _num(s: str):
    s = s.strip().replace(",", "").replace("%", "")
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_pl(html: str):
    """Return (years, rows_dict) where years is ['Mar 2015',...] and rows maps
    lowercased-label -> list[float|None]. None if no P&L section / all-empty."""
    soup = BeautifulSoup(html, "html.parser")
    sec = soup.find("section", id="profit-loss")
    if not sec:
        return None
    tbl = sec.find("table")
    if not tbl or not tbl.find("thead"):
        return None
    years = [th.get_text(strip=True) for th in tbl.find("thead").find_all("th")][1:]
    rows = {}
    any_val = False
    for tr in tbl.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        label = tds[0].get_text(strip=True).rstrip("+").strip().lower()
        vals = [_num(td.get_text(strip=True)) for td in tds[1:]]
        if any(v is not None for v in vals):
            any_val = True
        rows[label] = vals
    if not any_val:
        return None
    return years, rows


def parse_equity_capital(html: str):
    soup = BeautifulSoup(html, "html.parser")
    sec = soup.find("section", id="balance-sheet")
    if not sec:
        return None
    tbl = sec.find("table")
    if not tbl or not tbl.find("thead"):
        return None
    years = [th.get_text(strip=True) for th in tbl.find("thead").find_all("th")][1:]
    for tr in tbl.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        label = tds[0].get_text(strip=True).rstrip("+").strip().lower()
        if label == "equity capital":
            vals = [_num(td.get_text(strip=True)) for td in tds[1:]]
            return dict(zip(years, vals))
    return None


def year_to_fpe(col: str) -> str | None:
    """'Mar 2015' -> '2015-03-31'. Only Mar year-ends handled (screener default)."""
    m = re.match(r"([A-Za-z]{3})\s+(\d{4})", col)
    if not m:
        return None
    mon, yr = m.group(1), int(m.group(2))
    month_end = {"Mar": "03-31", "Dec": "12-31", "Jun": "06-30", "Sep": "09-30",
                 "Jun.": "06-30"}
    if mon not in month_end:
        return None
    return f"{yr}-{month_end[mon]}"


def process(ticker: str):
    """Return (list_of_row_dicts, status_str, source_str). status in
    ok/ok_standalone/no_pl/fetch_fail_<code>."""
    slug = slug_of(ticker)
    html, code = fetch(slug, consolidated=True)
    used_consolidated = True
    parsed = parse_pl(html) if html else None
    if parsed is None:
        # fallback to standalone
        html2, code2 = fetch(slug, consolidated=False)
        if html2 is None:
            if html is None:
                return [], f"fetch_fail_{code}", ""
            return [], f"no_pl_fetch_std_{code2}", ""
        parsed = parse_pl(html2)
        used_consolidated = False
        html = html2
        if parsed is None:
            return [], "no_pl", ""
    years, rows = parsed
    eq = parse_equity_capital(html)

    # pick revenue label
    rev_label = None
    for cand in REVENUE_LABELS:
        for lab in rows:
            if lab == cand or lab.startswith(cand):
                rev_label = lab
                break
        if rev_label:
            break
    ambiguous = rev_label in AMBIGUOUS_REVENUE if rev_label else True

    rev = rows.get(rev_label) if rev_label else None
    net = rows.get("net profit")
    eps = rows.get("eps in rs")
    op = rows.get("operating profit") or rows.get("financing profit")
    pbt = rows.get("profit before tax")

    out_rows = []
    for i, col in enumerate(years):
        fpe = year_to_fpe(col)
        if fpe is None:
            continue

        def g(arr):
            return arr[i] if arr and i < len(arr) and arr[i] is not None else None

        rev_v = g(rev)
        net_v = g(net)
        eps_v = g(eps)
        op_v = g(op)
        pbt_v = g(pbt)
        # all-None row -> skip (screener pads future/empty cols)
        if rev_v is None and net_v is None and eps_v is None:
            continue
        out_rows.append({
            "ticker": ticker,
            "fiscal_period_end": fpe,
            "revenue": rev_v * 10 if rev_v is not None else None,      # Cr -> INR mn
            "net_income": net_v * 10 if net_v is not None else None,
            "operating_profit": op_v * 10 if op_v is not None else None,
            "pbt": pbt_v * 10 if pbt_v is not None else None,
            "eps": eps_v,
            "shares": float("nan"),  # not reliably derivable w/o face value
            "consolidated": used_consolidated,
            "source": "screener_annual",
            "revenue_label": rev_label or "",
            "label_confidence": "ambiguous" if ambiguous else "ok",
        })
    status = "ok" if used_consolidated else "ok_standalone"
    return out_rows, status, ("consolidated" if used_consolidated else "standalone")


def main():
    cfg = yaml.safe_load(UNIVERSE.read_text())
    tickers = [t.split(":", 1)[1] for t in cfg["tickers"]]
    print(f"{len(tickers)} tickers", file=sys.stderr)

    all_rows = []
    status_log = {}
    for n, tk in enumerate(tickers, 1):
        try:
            rows, status, src = process(tk)
        except Exception as e:  # noqa: BLE001
            status_log[tk] = f"parse_error:{type(e).__name__}:{e}"
            print(f"[{n}/{len(tickers)}] {tk} PARSE_ERROR {e}", file=sys.stderr)
            continue
        status_log[tk] = status
        all_rows.extend(rows)
        if n % 25 == 0 or status not in ("ok",):
            print(f"[{n}/{len(tickers)}] {tk} {status} rows={len(rows)}",
                  file=sys.stderr)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["fiscal_period_end"] = pd.to_datetime(df["fiscal_period_end"])
        df = df.sort_values(["ticker", "fiscal_period_end"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"WROTE {OUT} rows={len(df)}", file=sys.stderr)

    # dump status log for the report
    pd.Series(status_log).to_frame("status").to_csv(
        CACHE / "status_log.csv")
    print("STATUS_COUNTS:", file=sys.stderr)
    print(pd.Series(status_log).value_counts().to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
